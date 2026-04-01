"""Walk-Forward Out-of-Sample (OOS) validation engine.

Implements anchored and rolling walk-forward validation for strategy backtests.
Produces per-fold IS/OOS metrics, degradation ratios, and aggregated summaries
designed to detect overfitting.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, median, stdev
from typing import Any, Optional

import numpy as np

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestSummary, BacktestTrade
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from app.utils.math_utils import safe_ratio as _safe_ratio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldWindow:
    """A single walk-forward fold with train/test periods."""
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class FoldResult:
    """Results for a single fold, both IS (train) and OOS (test)."""
    fold_index: int
    candidate: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # IS (in-sample) metrics
    is_trades: int = 0
    is_win_rate: float = 0.0
    is_profit_factor: float = 0.0
    is_expectancy_r: float = 0.0
    is_cumulative_r: float = 0.0
    is_max_drawdown_r: float = 0.0
    is_payoff_ratio: float = 0.0
    # OOS (out-of-sample) metrics
    oos_trades: int = 0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_expectancy_r: float = 0.0
    oos_cumulative_r: float = 0.0
    oos_max_drawdown_r: float = 0.0
    oos_payoff_ratio: float = 0.0
    # Degradation metrics
    expectancy_degradation: float = 0.0  # OOS/IS ratio, <1 = degraded
    pf_degradation: float = 0.0
    wr_degradation: float = 0.0


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward validation report."""
    generated_at: str
    candidate: str
    scheme: str  # "rolling" or "anchored"
    num_folds: int
    total_is_trades: int
    total_oos_trades: int
    # Aggregated IS metrics
    avg_is_expectancy_r: float = 0.0
    avg_is_profit_factor: float = 0.0
    avg_is_win_rate: float = 0.0
    # Aggregated OOS metrics
    avg_oos_expectancy_r: float = 0.0
    avg_oos_profit_factor: float = 0.0
    avg_oos_win_rate: float = 0.0
    cumulative_oos_r: float = 0.0
    max_oos_drawdown_r: float = 0.0
    # Overfitting diagnostics
    avg_expectancy_degradation: float = 0.0
    avg_pf_degradation: float = 0.0
    oos_positive_folds: int = 0
    oos_negative_folds: int = 0
    overfitting_probability: float = 0.0  # % of folds with OOS < 0
    # Consistency
    oos_expectancy_std: float = 0.0
    oos_sharpe_equivalent: float = 0.0  # mean OOS exp / std
    # Per-fold detail
    folds: list[FoldResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["folds"] = [asdict(f) for f in self.folds]
        return result


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def generate_folds(
    *,
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    scheme: str = "rolling",
) -> list[FoldWindow]:
    """Generate walk-forward folds.

    scheme='rolling': train window slides forward.
    scheme='anchored': train always starts from ``start``, expands over time.
    """
    folds: list[FoldWindow] = []
    anchor_start = start
    train_start = start
    train_end = train_start + timedelta(days=train_days)
    index = 1

    while train_end + timedelta(days=test_days) <= end:
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        folds.append(
            FoldWindow(
                index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        index += 1
        if scheme == "anchored":
            train_end = train_end + timedelta(days=step_days)
            train_start = anchor_start
        else:
            train_start = train_start + timedelta(days=step_days)
            train_end = train_start + timedelta(days=train_days)

    return folds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHARPE_CAP = 99.0  # Maximum Sharpe-equivalent when std is exactly zero


def _extract_summary_metrics(summary: BacktestSummary) -> dict[str, float]:
    return {
        "trades": summary.total_trades,
        "win_rate": summary.win_rate,
        "profit_factor": summary.profit_factor,
        "expectancy_r": summary.expectancy_r,
        "cumulative_r": summary.cumulative_r,
        "max_drawdown_r": summary.max_drawdown_r,
        "payoff_ratio": summary.payoff_ratio,
    }


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """Run walk-forward OOS validation for a single strategy candidate."""

    def __init__(
        self,
        *,
        ohlcv_service: OhlcvService,
        strategy_service: StrategyService,
        assumptions: Optional[BacktestAssumptions] = None,
    ) -> None:
        self.ohlcv_service = ohlcv_service
        self.strategy_service = strategy_service
        self.assumptions = assumptions or BacktestAssumptions()

    def validate(
        self,
        *,
        strategy_profile: str,
        symbol: str,
        exchange: str = "binance",
        market_type: str = "perpetual",
        folds: list[FoldWindow],
        min_train_trades: int = 8,
    ) -> WalkForwardReport:
        """Run walk-forward validation across all folds."""
        fold_results: list[FoldResult] = []

        for fold in folds:
            logger.info(
                "Fold %d: train=%s->%s  test=%s->%s",
                fold.index,
                fold.train_start.date(), fold.train_end.date(),
                fold.test_start.date(), fold.test_end.date(),
            )
            result = self._run_fold(
                strategy_profile=strategy_profile,
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                fold=fold,
                min_train_trades=min_train_trades,
            )
            fold_results.append(result)

        return self._aggregate(
            candidate=strategy_profile,
            scheme="rolling",  # caller can override
            folds=fold_results,
        )

    def _run_fold(
        self,
        *,
        strategy_profile: str,
        symbol: str,
        exchange: str,
        market_type: str,
        fold: FoldWindow,
        min_train_trades: int,
    ) -> FoldResult:
        """Run a single IS/OOS fold."""
        service = BacktestService(
            ohlcv_service=self.ohlcv_service,
            strategy_service=self.strategy_service,
            assumptions=self.assumptions,
        )

        # IS (train) run
        is_report = service.run(
            exchange=exchange,
            market_type=market_type,
            symbols=[symbol],
            strategy_profiles=[strategy_profile],
            start=fold.train_start,
            end=fold.train_end,
        )

        # OOS (test) run
        oos_report = service.run(
            exchange=exchange,
            market_type=market_type,
            symbols=[symbol],
            strategy_profiles=[strategy_profile],
            start=fold.test_start,
            end=fold.test_end,
        )

        # Extract metrics
        is_m = self._summarize(is_report)
        oos_m = self._summarize(oos_report)

        # Degradation ratios
        exp_degrad = _safe_ratio(oos_m["expectancy_r"], is_m["expectancy_r"])
        pf_degrad = _safe_ratio(oos_m["profit_factor"], is_m["profit_factor"])
        wr_degrad = _safe_ratio(oos_m["win_rate"], is_m["win_rate"])

        return FoldResult(
            fold_index=fold.index,
            candidate=strategy_profile,
            train_start=fold.train_start.isoformat(),
            train_end=fold.train_end.isoformat(),
            test_start=fold.test_start.isoformat(),
            test_end=fold.test_end.isoformat(),
            is_trades=int(is_m["trades"]),
            is_win_rate=round(is_m["win_rate"], 2),
            is_profit_factor=round(is_m["profit_factor"], 4),
            is_expectancy_r=round(is_m["expectancy_r"], 4),
            is_cumulative_r=round(is_m["cumulative_r"], 4),
            is_max_drawdown_r=round(is_m["max_drawdown_r"], 4),
            is_payoff_ratio=round(is_m["payoff_ratio"], 4),
            oos_trades=int(oos_m["trades"]),
            oos_win_rate=round(oos_m["win_rate"], 2),
            oos_profit_factor=round(oos_m["profit_factor"], 4),
            oos_expectancy_r=round(oos_m["expectancy_r"], 4),
            oos_cumulative_r=round(oos_m["cumulative_r"], 4),
            oos_max_drawdown_r=round(oos_m["max_drawdown_r"], 4),
            oos_payoff_ratio=round(oos_m["payoff_ratio"], 4),
            expectancy_degradation=round(exp_degrad, 4),
            pf_degradation=round(pf_degrad, 4),
            wr_degradation=round(wr_degrad, 4),
        )

    @staticmethod
    def _summarize(report) -> dict[str, float]:
        """Extract top-level summary from a backtest report."""
        if report.overall:
            return _extract_summary_metrics(report.overall[0])
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "cumulative_r": 0.0,
            "max_drawdown_r": 0.0,
            "payoff_ratio": 0.0,
        }

    @staticmethod
    def _aggregate(
        *,
        candidate: str,
        scheme: str,
        folds: list[FoldResult],
    ) -> WalkForwardReport:
        """Build aggregated report from per-fold results."""
        num_folds = len(folds)
        if num_folds == 0:
            return WalkForwardReport(
                generated_at=datetime.now(timezone.utc).isoformat(),
                candidate=candidate,
                scheme=scheme,
                num_folds=0,
                total_is_trades=0,
                total_oos_trades=0,
                folds=[],
            )

        total_is_trades = sum(f.is_trades for f in folds)
        total_oos_trades = sum(f.oos_trades for f in folds)

        avg_is_exp = mean(f.is_expectancy_r for f in folds)
        avg_is_pf = mean(f.is_profit_factor for f in folds)
        avg_is_wr = mean(f.is_win_rate for f in folds)

        oos_exps = [f.oos_expectancy_r for f in folds]
        avg_oos_exp = mean(oos_exps)
        avg_oos_pf = mean(f.oos_profit_factor for f in folds)
        avg_oos_wr = mean(f.oos_win_rate for f in folds)

        cumulative_oos_r = sum(f.oos_cumulative_r for f in folds)

        # Max drawdown across OOS equity curve
        oos_curve = []
        for f in folds:
            oos_curve.append(f.oos_cumulative_r)
        cum_oos = list(np.cumsum(oos_curve))
        if cum_oos:
            peak = np.maximum.accumulate(cum_oos)
            dd = peak - np.array(cum_oos)
            max_oos_dd = float(dd.max())
        else:
            max_oos_dd = 0.0

        # Degradation
        exp_degrads = [f.expectancy_degradation for f in folds if f.is_expectancy_r != 0]
        avg_exp_degrad = mean(exp_degrads) if exp_degrads else 0.0
        pf_degrads = [f.pf_degradation for f in folds if f.is_profit_factor != 0]
        avg_pf_degrad = mean(pf_degrads) if pf_degrads else 0.0

        oos_pos = sum(1 for e in oos_exps if e > 0)
        oos_neg = sum(1 for e in oos_exps if e <= 0)
        overfit_prob = (oos_neg / num_folds) * 100 if num_folds > 0 else 0.0

        oos_std = stdev(oos_exps) if len(oos_exps) >= 2 else 0.0
        # Sharpe equivalent: when std=0 and mean>0, return capped high value
        if oos_std == 0.0:
            oos_sharpe = SHARPE_CAP if avg_oos_exp > 0 else (0.0 if avg_oos_exp == 0 else -SHARPE_CAP)
        else:
            oos_sharpe = avg_oos_exp / oos_std

        return WalkForwardReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            candidate=candidate,
            scheme=scheme,
            num_folds=num_folds,
            total_is_trades=total_is_trades,
            total_oos_trades=total_oos_trades,
            avg_is_expectancy_r=round(avg_is_exp, 4),
            avg_is_profit_factor=round(avg_is_pf, 4),
            avg_is_win_rate=round(avg_is_wr, 2),
            avg_oos_expectancy_r=round(avg_oos_exp, 4),
            avg_oos_profit_factor=round(avg_oos_pf, 4),
            avg_oos_win_rate=round(avg_oos_wr, 2),
            cumulative_oos_r=round(cumulative_oos_r, 4),
            max_oos_drawdown_r=round(max_oos_dd, 4),
            avg_expectancy_degradation=round(avg_exp_degrad, 4),
            avg_pf_degradation=round(avg_pf_degrad, 4),
            oos_positive_folds=oos_pos,
            oos_negative_folds=oos_neg,
            overfitting_probability=round(overfit_prob, 2),
            oos_expectancy_std=round(oos_std, 4),
            oos_sharpe_equivalent=round(oos_sharpe, 4),
            folds=folds,
        )
