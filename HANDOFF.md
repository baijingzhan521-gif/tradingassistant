# 回测交接文档

更新时间：2026-04-01

本文档用于向新同事完整交接当前项目的回测体系。请按顺序阅读，不要跳过任何章节。

---

## 0. 阅读本文档之前

**最重要的一句话：** 这个项目的回测基础设施已经 100% 完成，代码质量已通过 402 个自动化测试。回测支持两种数据模式：

1. **在线模式**：连接 Binance 交易所拉取真实 OHLCV K 线数据（需要能访问 Binance 的网络环境）
2. **离线模式**：使用 `scripts/generate_offline_ohlcv.py` 生成合成数据，**无需任何交易所连接或 API Key**

本文档的目标读者是接手这个项目并需要在本地电脑环境下运行回测的同事。

---

## 1. 环境搭建

### 1.1 Python 版本

**要求：Python 3.11+**

项目根目录有 `.python-version` 文件，标注 `3.11`。

```bash
# 方式一：如果本地已有 python3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 方式二：使用 uv（推荐，自动管理 Python 版本）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

### 1.2 环境变量

```bash
cp .env.example .env
```

重点配置项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_EXCHANGE` | `binance` | 当前只支持 Binance |
| `DEFAULT_MARKET_TYPE` | `perpetual` | 只支持 USDT 永续合约 |
| `CCXT_HTTP_PROXY` | 空 | **如果你的网络不能直连 Binance，必须配置代理** |
| `CCXT_HTTPS_PROXY` | 空 | 同上 |
| `CCXT_SOCKS_PROXY` | 空 | 同上（支持 socks5） |

### 1.3 网络连通性检查

```bash
python scripts/check_exchange_connectivity.py
```

这个脚本会检查：
- Python 版本
- 直连 Binance 是否可达
- 环境变量代理是否生效
- 本地常见代理端口是否在监听
- `ccxt.binanceusdm` 的 `load_markets()` 是否能跑通

**如果这一步不通过，后续所有回测脚本都无法运行。**

### 1.4 验证测试套件

```bash
python -m pytest tests/ --ignore=tests/test_review_smoke.py --ignore=tests/test_workspace_smoke.py -q
```

预期结果：`402 passed`。这两个被忽略的测试文件需要 Playwright 浏览器二进制，跟回测无关。

---

## 2. 项目整体架构

```
tradingassistant/
├── app/                          # 核心应用代码
│   ├── api/                      # FastAPI 路由层
│   ├── backtesting/              # 【回测核心】
│   │   ├── service.py            # BacktestService 主引擎（~1350行）
│   │   ├── walk_forward_validator.py  # Walk-Forward OOS 验证器
│   │   ├── stress_tester.py      # 9场景压力测试器
│   │   ├── portfolio_orchestrator.py  # 多策略组合编排器
│   │   └── diagnostics.py        # 信号诊断收集器
│   ├── indicators/               # 技术指标（12个）
│   │   ├── ema.py                # EMA (21/55/100/200)
│   │   ├── atr.py                # ATR (14周期)
│   │   ├── rsi.py                # RSI (14周期)
│   │   ├── adx.py                # ADX (趋势强度)
│   │   ├── bollinger.py          # 布林带 (20周期/2σ)
│   │   ├── donchian.py           # 唐奇安通道
│   │   ├── regime_classifier.py  # 市场regime分类器（5种regime）
│   │   ├── swings.py             # Swing高低点检测
│   │   ├── divergence.py         # RSI背离
│   │   ├── market_structure.py   # 市场结构分析
│   │   └── candle_profile.py     # K线形态分析
│   ├── strategies/               # 策略集合（70+个文件）
│   │   ├── windowed_mtf.py       # 策略基类（多时间框架窗口）
│   │   ├── trend_following_v1.py # 趋势跟踪（Donchian突破）
│   │   ├── swing_improved_v1.py  # 波段交易（放宽条件版）
│   │   ├── mean_reversion_v1.py  # 均值回归（布林带）
│   │   └── ...                   # 大量研究/消融/变体策略
│   ├── data/                     # 交易所数据层
│   ├── services/                 # 业务服务层
│   ├── utils/                    # 工具函数
│   └── static/                   # 前端静态页面
├── scripts/                      # 回测运行脚本（76个）
├── tests/                        # 测试（402个测试用例）
├── config/strategies/            # 策略YAML配置
└── docs/                         # 文档
    ├── project_memory.md         # 【必读】项目记忆与完整研究结论
    ├── system_design.md          # 系统设计文档
    └── api.md                    # API文档
```

---

## 3. 回测体系详解

### 3.1 核心回测引擎 (`app/backtesting/service.py`)

`BacktestService` 是整个回测系统的核心。它的工作流程是：

1. 从 Binance 拉取历史 OHLCV（带本地 CSV 缓存）
2. 在每根 K 线上评估策略信号
3. 管理头寸生命周期（入场 → 持仓 → 出场）
4. 输出结构化报告（JSON/CSV）

**关键假设参数 (`BacktestAssumptions`)**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `taker_fee_bps` | 5.0 | 单边 taker 手续费（bps） |
| `slippage_bps` | 2.0 | 单边滑点估计（bps） |
| `take_profit_mode` | `"scaled"` | `"scaled"` = TP1/TP2 分批出场；`"fixed_r"` = 固定R全平 |
| `scaled_tp1_r` | None | TP1 目标 R 倍数 |
| `scaled_tp2_r` | None | TP2 目标 R 倍数 |
| `tp1_scale_out` | 0.5 | TP1 出场比例 |
| `move_stop_to_entry_after_tp1` | True | TP1 后移动止损到入场价 |
| `swing_max_hold_bars` | 240 | 最长持仓 K 线数 |
| `trailing_stop_enabled` | False | ATR 追踪止损 |
| `trailing_stop_atr_mult` | 3.0 | 追踪止损 ATR 乘数 |
| `trailing_stop_activation_r` | 1.0 | 追踪止损激活阈值 |
| `leverage` | 1.0 | 杠杆倍数 |
| `max_leverage` | 3.0 | 最大杠杆 |
| `swing_detection_mode` | `"centered"` | swing 检测模式（`centered` 或 `confirmed`） |

**出场原因分类**：`tp1`, `tp2`, `trailing_stop`, `breakeven_after_tp1`, `stop_loss`, `time_stop`, `end_of_test`

### 3.2 Walk-Forward OOS 验证器 (`app/backtesting/walk_forward_validator.py`)

用于检测策略是否过拟合。

**核心流程**：
1. 把历史数据切成多个 fold（训练期 + 测试期）
2. 在每个 fold 的训练期（IS）上运行回测，提取指标
3. 在每个 fold 的测试期（OOS）上运行回测，提取指标
4. 计算 OOS/IS 退化比率（<1 表示退化）
5. 计算过拟合概率（OOS 亏损的 fold 数 / 总 fold 数）

**两种方案**：
- **Rolling（滚动）**：训练窗口大小固定，向前滑动
- **Anchored（锚定）**：训练起点固定，窗口持续扩大

**默认参数**：`365天训练 / 90天测试 / 90天步进`

### 3.3 压力测试器 (`app/backtesting/stress_tester.py`)

9 个递增压力场景：

| 场景 | 手续费倍数 | 滑点倍数 |
|------|-----------|---------|
| baseline | 1× | 1× |
| fees_2x | 2× | 1× |
| fees_3x | 3× | 1× |
| slippage_2x | 1× | 2× |
| slippage_3x | 1× | 3× |
| combined_2x | 2× | 2× |
| combined_3x | 3× | 3× |
| extreme_fees_5x | 5× | 1× |
| extreme_combined | 3× | 5× |

**鲁棒性评分** = 仍盈利的场景数 / 总场景数 × 100%

### 3.4 组合编排器 (`app/backtesting/portfolio_orchestrator.py`)

管理三个策略（`trend_following_v1` + `swing_improved_v1` + `mean_reversion_v1`）的组合配比。

**5 种市场 Regime 的权重配置**：

| Regime | 趋势跟踪 | 波段交易 | 均值回归 | 最大杠杆 |
|--------|---------|---------|---------|---------|
| BULL_TREND | 60% | 30% | 10% | 3.0× |
| BEAR_TREND | 50% | 40% | 10% | 2.0× |
| LOW_VOL_RANGE | 10% | 20% | 70% | 1.5× |
| HIGH_VOL_CHOP | 0% | 10% | 30% | 0.5× |
| TRANSITION | 20% | 50% | 30% | 1.0× |

**三级回撤断路器**：
- Level 1（≥10% DD）：杠杆 × 0.7
- Level 2（≥20% DD）：杠杆 × 0.4
- Level 3（≥25% DD）：杠杆 × 0.0（全停 + 168 根 K 线冷却）

---

## 4. 策略体系

### 4.1 当前主线候选

当前研究主线已收敛到固定日历 regime-switch 方案：

```
2020-01-01 → 2024-03-19：swing_trend_simple_candidate_v2
2024-03-19 → 至今：      swing_trend_long_regime_short_no_reversal_no_aux_v1
                         + be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98
```

关键指标（base / full_2020：2020-01-01 → 2026-03-19）：
- 几何收益：157.85%
- CAGR：16.39%
- PF：1.3445
- 累计 R：45.64R
- 最大回撤：12.20R

### 4.2 三大新策略（组合编排用）

| 策略 | 类型 | 核心逻辑 | 默认杠杆 |
|------|------|----------|---------|
| `trend_following_v1` | 趋势跟踪 | Donchian 突破 + EMA200 方向过滤 + ADX≥40 | 2.0× |
| `swing_improved_v1` | 波段交易 | 多时间框架 regained_fast + held_slow 触发 | 1.5× |
| `mean_reversion_v1` | 均值回归 | 布林带极端区 + RSI 过滤 + 反转 K 线确认 | 1.0× |

### 4.3 独立候选池

- `swing_exhaustion_divergence_ct_block80_v1_btc` — 唯一通过独立收益晋级门的互补 alpha

### 4.4 已冻结/出局的方向

- 日内策略（`intraday_mtf_v1`, `intraday_mtf_v2`）— 无稳定 edge
- 多标的直接复制 — BTC-only 仍优于组合
- confluence gate / level-aware confirmation — 长样本不成立
- axis_distance_z / band_position — 只能作 state_note
- derivatives 方向 alpha — 当前只是 state layer
- carry / basis — research-only
- range-failure — standalone 太差，已出局
- breakout 家族 — one-rule gate 未通过
- neutral-range 家族 — one-rule gate 未通过
- trend-confluence 家族 — one-rule gate 未通过
- level-aware confirmation 家族 — one-rule gate 未通过

---

## 5. 运行回测的具体命令

### 5.0 离线模式：无需交易所连接即可回测（推荐先用这个验证环境）

如果你的网络无法访问 Binance，或者只是想快速验证回测流程，可以使用离线数据生成器：

```bash
# 第一步：生成合成 OHLCV 数据（GBM 模型 + 类 BTC 波动率）
python scripts/generate_offline_ohlcv.py

# 第二步：直接跑回测（自动从缓存读取，不联网）
python scripts/run_backtest.py \
  --symbols "BTC/USDT:USDT" \
  --strategy-profiles "swing_trend_long_regime_gate_v1" \
  --start 2024-03-19 --end 2026-03-19 \
  --take-profit-mode scaled --scaled-tp1-r 1.0 --scaled-tp2-r 3.0 \
  --long-exit-json '{"take_profit_mode":"scaled","scaled_tp1_r":1.0,"scaled_tp2_r":3.0}' \
  --short-exit-json '{"take_profit_mode":"fixed_r","fixed_take_profit_r":1.5}' \
  --output-dir artifacts/backtests/offline_demo
```

**生成器支持的参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | `BTC/USDT:USDT` | 交易对 |
| `--start` | `2024-03-19` | 回测起始日 |
| `--end` | `2026-03-19` | 回测结束日 |
| `--start-price` | `64000.0` | 起始价格 |
| `--annual-drift` | `0.15` | 年化漂移率 |
| `--annual-vol` | `0.65` | 年化波动率（BTC 级别） |
| `--seed` | `42` | 随机种子（相同种子 = 相同数据） |
| `--lookback` | `300` | 预加载 K 线数 |

**生成长样本（2020→2026）：**

```bash
python scripts/generate_offline_ohlcv.py \
  --start 2020-01-01 --end 2026-03-19 \
  --start-price 7200 --seed 42
```

**原理说明：**
- 数据使用 Geometric Brownian Motion (GBM) 生成，包含 regime 切换（牛/熊/震荡）和波动率聚集效应
- CSV 文件保存在 `artifacts/backtests/cache/`，格式与交易所真实数据完全一致
- `BacktestService` 会自动检测缓存命中，跳过 API 调用
- 合成数据的结果**不应被当成真实策略表现**，仅用于验证回测流程和代码正确性

### 5.1 启动 Web 服务（可选，回测不需要）

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
# 主工作台: http://127.0.0.1:8000/
# 复盘页面: http://127.0.0.1:8000/review
```

### 5.2 基础回测（最快验证环境是否就绪）

```bash
python scripts/run_backtest.py \
  --symbols "BTC/USDT:USDT" \
  --strategy-profiles "swing_trend_long_regime_gate_v1" \
  --start 2024-03-19 --end 2026-03-19 \
  --take-profit-mode scaled --scaled-tp1-r 1.0 --scaled-tp2-r 3.0 \
  --long-exit-json '{"take_profit_mode":"scaled","scaled_tp1_r":1.0,"scaled_tp2_r":3.0}' \
  --short-exit-json '{"take_profit_mode":"fixed_r","fixed_take_profit_r":1.5}' \
  --output-dir artifacts/backtests/baseline
```

### 5.3 Walk-Forward OOS 验证（近两年）

```bash
python scripts/run_walk_forward.py \
  --symbol "BTC/USDT:USDT" \
  --start 2024-03-19 --end 2026-03-19 \
  --train-days 365 --test-days 90 --step-days 90 \
  --candidates "swing_trend_long_regime_gate_v1@long_scaled1_3_short_fixed1_5" \
  --output-dir artifacts/backtests/btc_walk_forward_two_year
```

### 5.4 Walk-Forward OOS 验证（长样本，推荐）

```bash
python scripts/run_walk_forward.py \
  --symbol "BTC/USDT:USDT" \
  --start 2020-01-01 --end 2026-03-19 \
  --train-days 365 --test-days 90 --step-days 90 \
  --candidates \
    "swing_trend_long_regime_gate_v1@long_scaled1_3_short_fixed1_5,\
swing_trend_ablation_no_auxiliary_v1@long_scaled1_3_short_fixed1_5,\
swing_trend_simple_candidate_v2@long_scaled1_3_short_fixed1_5" \
  --output-dir artifacts/backtests/btc_entry_candidates_2020_walk_forward
```

### 5.5 多标的组合 Walk-Forward

```bash
python scripts/run_multi_asset_portfolio_walk_forward.py \
  --start 2024-03-19 --end 2026-03-19 \
  --output-dir artifacts/backtests/multi_asset_portfolio_walk_forward
```

### 5.6 使用新的三策略组合回测（编排器 + WF-OOS + 压力测试）

要使用新的 `trend_following_v1 / swing_improved_v1 / mean_reversion_v1` 三策略组合进行回测，可以通过 Python API 直接调用：

```python
from datetime import datetime, timezone
from pathlib import Path
from app.backtesting.service import BacktestService, BacktestAssumptions
from app.backtesting.walk_forward_validator import WalkForwardValidator
from app.backtesting.stress_tester import StressTester
from app.backtesting.portfolio_orchestrator import PortfolioOrchestrator, PortfolioConfig
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService

# 初始化服务
ohlcv = OhlcvService(get_exchange_client_factory())
strategy_svc = StrategyService()

# --- 1) 单策略基线回测 ---
service = BacktestService(
    ohlcv_service=ohlcv,
    strategy_service=strategy_svc,
    assumptions=BacktestAssumptions(
        trailing_stop_enabled=True,
        trailing_stop_atr_mult=3.0,
        trailing_stop_activation_r=1.0,
        leverage=2.0,
    ),
)
report = service.run(
    exchange="binance",
    market_type="perpetual",
    symbols=["BTC/USDT:USDT"],
    strategy_profiles=["trend_following_v1"],
    start=datetime(2024, 3, 19, tzinfo=timezone.utc),
    end=datetime(2026, 3, 19, tzinfo=timezone.utc),
)

# --- 2) Walk-Forward OOS ---
validator = WalkForwardValidator(
    backtest_service=service,
    exchange="binance",
    market_type="perpetual",
)
wf_report = validator.run(
    symbol="BTC/USDT:USDT",
    strategy_profile="trend_following_v1",
    start=datetime(2024, 3, 19, tzinfo=timezone.utc),
    end=datetime(2026, 3, 19, tzinfo=timezone.utc),
    train_days=365,
    test_days=90,
    step_days=90,
    scheme="rolling",
)
print(f"过拟合概率: {wf_report.overfitting_probability}%")
print(f"OOS期望值: {wf_report.avg_oos_expectancy_r}R")

# --- 3) 压力测试 ---
tester = StressTester(
    backtest_service=service,
    exchange="binance",
    market_type="perpetual",
)
stress_report = tester.run(
    symbol="BTC/USDT:USDT",
    strategy_profile="trend_following_v1",
    start=datetime(2024, 3, 19, tzinfo=timezone.utc),
    end=datetime(2026, 3, 19, tzinfo=timezone.utc),
)
print(f"鲁棒性评分: {stress_report.robustness_score}%")

# --- 4) 组合编排 ---
orchestrator = PortfolioOrchestrator(
    ohlcv_service=ohlcv,
    strategy_service=strategy_svc,
    config=PortfolioConfig(),
)
portfolio_report = orchestrator.run(
    exchange="binance",
    market_type="perpetual",
    symbols=["BTC/USDT:USDT"],
    start=datetime(2024, 3, 19, tzinfo=timezone.utc),
    end=datetime(2026, 3, 19, tzinfo=timezone.utc),
)
```

---

## 6. 已有的关键回测结论

> 以下结论全部来自 `docs/project_memory.md`，已在真实 Binance 数据上验证。

### 6.1 主线近两年（2024-03-19 → 2026-03-19）

策略：`swing_trend_long_regime_gate_v1` + `LONG 1R→3R scaled / SHORT 1.5R fixed` + confirmed swing

- 77 笔交易
- PF 1.69
- 期望值 0.28R/笔
- 累计 +21.35R
- 最大回撤 5.46R
- Walk-Forward OOS：4/4 fold 全选中，OOS +10.43R

### 6.2 主线长样本（2020-01-01 → 2026-03-19）

**拉长到 6 年后只剩薄 edge**：
- 272 笔，PF 1.03，期望值 0.02R/笔，累计 +4.84R，最大回撤 25.98R
- Walk-Forward OOS：11/21 有效 fold，OOS +13.11R

### 6.3 regime-switch 主线（最强版本）

`simple_candidate_v2（2020→2024.03）→ challenger_managed（2024.03→至今）`

- 几何收益 157.85%，CAGR 16.39%
- PF 1.34，累计 45.64R，最大回撤 12.20R
- 稳健性：5 个 probe 切点全部通过

### 6.4 核心判断

- **当前 BTC 主线更像 regime-dependent 策略，不像 always-on 通用策略**
- **多标的直接复制没有比 BTC-only 更好**
- **日内策略没有稳定 edge**

---

## 7. 产物输出位置

所有回测脚本默认输出到 `artifacts/backtests/` 目录下（首次运行会自动创建）。

| 产物类型 | 文件格式 | 说明 |
|----------|----------|------|
| 主报告 | `.json` | 完整回测结果，含所有交易记录 |
| 汇总报告 | `.md` | 人类可读的 Markdown 汇总 |
| 交易明细 | `.csv` | 每笔交易的详细数据 |
| OHLCV 缓存 | `.csv` | 缓存在 `artifacts/backtests/cache/`，避免重复拉取 |

---

## 8. 重要文件速查

### 必读文件

| 文件 | 内容 |
|------|------|
| `docs/project_memory.md` | **1875 行完整项目记忆**，包含所有研究结论和判断依据 |
| `app/backtesting/service.py` | 回测引擎核心 |
| `app/backtesting/walk_forward_validator.py` | WF-OOS 验证 |
| `app/backtesting/stress_tester.py` | 压力测试 |
| `app/backtesting/portfolio_orchestrator.py` | 组合编排 |

### 策略文件

| 文件 | 角色 |
|------|------|
| `app/strategies/windowed_mtf.py` | 所有策略的基类 |
| `app/strategies/swing_trend_long_regime_gate_v1.py` | 当前研究主线 |
| `app/strategies/swing_trend_simple_candidate_v2.py` | regime-switch 前段候选 |
| `app/strategies/trend_following_v1.py` | 新趋势跟踪策略 |
| `app/strategies/swing_improved_v1.py` | 新波段策略 |
| `app/strategies/mean_reversion_v1.py` | 新均值回归策略 |

### 核心脚本

| 脚本 | 用途 |
|------|------|
| `scripts/run_backtest.py` | 基础回测 |
| `scripts/run_walk_forward.py` | Walk-Forward OOS 验证 |
| `scripts/run_multi_asset_portfolio_walk_forward.py` | 多标的组合 WF |
| `scripts/run_sizing_walk_forward.py` | 仓位大小 WF |
| `scripts/run_risk_budget_walk_forward.py` | 风险预算 WF |

---

## 9. 注册新策略的流程

如果你要新增策略，需要做 3 步：

1. 在 `app/strategies/` 下创建策略文件，继承 `WindowedMTFStrategy`
2. 在 `app/services/strategy_service.py` 中注册策略类
3. 在 `app/utils/timeframes.py` 中注册三个字典：
   - `STRATEGY_REQUIRED_TIMEFRAMES`
   - `STRATEGY_SUPPLEMENTAL_TIMEFRAMES`
   - `STRATEGY_FETCH_TIMEFRAMES`

---

## 10. 当前代码分支状态

- 当前分支：`copilot/analyze-quant-strategy-performance`
- 最新提交：`7cf4498` - "Add Walk-Forward OOS validator, stress tester, and register new strategies in timeframe maps"
- 工作区：干净（no uncommitted changes）
- 测试状态：402 passed

### 最近两次提交内容

1. **7cf4498** — 新增 WF-OOS 验证器、压力测试器，注册三大新策略到 timeframes 映射
2. **1f6725d** — 代码审查修复：提取 `safe_ratio` 到共享 `math_utils`，使用 `SHARPE_CAP` 常量

---

## 11. 已知限制和注意事项

### 回测本身的限制

- **不是撮合级仿真**：没有订单簿、真实滑点建模、funding rate
- **信号延迟**：信号在当前 bar 收盘形成，下一根 bar 开盘进场
- **单头寸**：同一标的同一策略同一时刻只持有一笔
- **保守处理**：同一根 K 线同时触发止损和止盈时，优先算止损
- **不应把回测数字直接当成实盘承诺**

### 网络限制

- 所有回测数据来自 Binance 永续合约（`binanceusdm`）
- 如果网络不能直连 Binance，需要配置 `CCXT_HTTP_PROXY` / `CCXT_HTTPS_PROXY`
- 首次拉取数据较慢，后续会使用本地 CSV 缓存

### 数据限制

- `HYPE/USDT:USDT` 只从 2025-05-30 开始有数据
- 建议长样本回测从 2020-01-01 开始，不要更早

---

## 12. 下一步建议

根据 `docs/project_memory.md` 中的研究结论，推荐的下一步方向（三选一，不要同时做）：

### 路线 A：把 BTC 主线做成显式部署规则
- 确定何时启用 / 何时停用
- regime-switch 已经有初步验证
- 继续围绕 `exhaustion divergence` 做互补 alpha

### 路线 B：重做多标的路线
- 先做 symbol admission（标的准入），不再直接复制 BTC 规则
- 用三策略组合编排器跑多标的组合验证

### 路线 C：使用 side-permission 外部模型
- 研究基础设施已就绪（`scripts/run_mainline_side_permission_backtest.py`）
- 需要提供真实的 regime / side permission CSV
- 先看 `full_2020` 的完整报告

### 不建议继续的方向

- ❌ 继续在 `intraday_mtf_v2` 上微调
- ❌ 把多标的直接复制当已证明路线
- ❌ 只看近两年给长期主线拍板（默认先看 2020→至今）
- ❌ 继续 confluence gate / level-aware confirmation
- ❌ 把 derivatives 特征硬塞回方向策略
- ❌ 把 carry/basis 当可部署策略

---

## 13. 常用命令速查

```bash
# 激活环境
source .venv/bin/activate

# 运行测试
python -m pytest tests/ --ignore=tests/test_review_smoke.py --ignore=tests/test_workspace_smoke.py -q

# 检查网络连通性
python scripts/check_exchange_connectivity.py

# 启动 Web 服务
uvicorn app.main:app --reload

# ========== 离线回测（不需要交易所连接）==========

# 生成离线数据（两年窗口）
python scripts/generate_offline_ohlcv.py

# 生成离线数据（六年窗口）
python scripts/generate_offline_ohlcv.py --start 2020-01-01 --end 2026-03-19 --start-price 7200

# 用离线数据跑基础回测
python scripts/run_backtest.py --symbols "BTC/USDT:USDT" --strategy-profiles "swing_trend_long_regime_gate_v1" --start 2024-03-19 --end 2026-03-19

# 用离线数据跑多策略回测
python scripts/run_backtest.py --symbols "BTC/USDT:USDT" --strategy-profiles "trend_following_v1,swing_improved_v1,mean_reversion_v1" --start 2024-03-19 --end 2026-03-19

# ========== 在线回测（需要 Binance 连接）==========

# 基础回测（从 Binance 拉数据）
python scripts/run_backtest.py --symbols "BTC/USDT:USDT" --strategy-profiles "swing_trend_long_regime_gate_v1" --start 2024-03-19 --end 2026-03-19

# Walk-Forward OOS
python scripts/run_walk_forward.py --symbol "BTC/USDT:USDT" --start 2020-01-01 --end 2026-03-19

# 多标的组合 WF
python scripts/run_multi_asset_portfolio_walk_forward.py
```

---

## 14. 联系与问题

- 项目完整记忆文档：`docs/project_memory.md`（1875 行，必读）
- 系统设计：`docs/system_design.md`
- API 文档：`docs/api.md`
- 如有问题，先确认网络连通性和 Python 环境是否正确

---

*本文档由项目交接时自动生成，基于 2026-04-01 仓库状态。*
