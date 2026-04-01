# API Contract

Base URL: `http://127.0.0.1:8000`

All endpoints are served by FastAPI and return JSON.

Built-in HTML pages:

- `GET /` and `GET /workspace`: main analysis workspace
- `GET /review`: internal review and diff page

## Common Error Shape

Application errors raised from the trading layer use this format:

```json
{
  "detail": "human readable message",
  "timestamp": "2026-03-18T12:00:00+00:00"
}
```

Typical status codes:

- `400`: unsupported runtime configuration or application-layer validation
- `422`: request schema validation failure
- `404`: symbol not found or analysis record not found
- `502`: upstream exchange/network failure
- `500`: persistence or internal error

## `GET /health`

Returns the service health and version.

### Response

```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2026-03-18T12:00:00+00:00"
}
```

## `GET /symbols`

Lists active Binance perpetual USDT symbols.

### Query Parameters

- `exchange`: default `binance`
- `market_type`: default `perpetual`
- `limit`: default `200`, min `1`, max `500`

### Example

```bash
curl "http://127.0.0.1:8000/symbols?exchange=binance&market_type=perpetual&limit=20"
```

### Response

```json
{
  "exchange": "binance",
  "market_type": "perpetual",
  "count": 20,
  "symbols": ["ADA/USDT:USDT", "BCH/USDT:USDT"]
}
```

## `POST /analyze`

Runs the full analysis pipeline and persists the result.

### Request Body

`AnalyzeRequest` fields:

- `symbol`: unified ccxt symbol, for example `ETH/USDT:USDT`
- `market_type`: currently `perpetual`
- `exchange`: currently `binance`
- `timeframes`: default mainline profile needs `["1d", "4h", "1h"]`; legacy profiles may require extra frames such as `15m`
- `strategy_profile`: default `swing_trend_long_regime_gate_v1`
- `lookback`: candle count, default from env, minimum `220`

### Example

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ETH/USDT:USDT",
    "market_type": "perpetual",
    "exchange": "binance",
    "timeframes": ["1d", "4h", "1h"],
    "strategy_profile": "swing_trend_long_regime_gate_v1",
    "lookback": 300
  }'
```

### Response Contract

The response is `AnalysisResult`.

Top-level fields:

- `analysis_id`
- `timestamp`
- `symbol`
- `exchange`
- `market_type`
- `strategy_profile`
- `timeframes`
- `market_regime`
- `decision`
- `reasoning`
- `diagnostics`
- `raw_metrics`

### Timeframe Analysis

`timeframes` is a fixed object with keys:

- `1d`
- `4h`
- `1h`
- `15m`

Each timeframe entry contains:

- `timeframe`
- `latest_timestamp`
- `close`
- `ema21`
- `ema55`
- `ema100`
- `ema200`
- `atr14`
- `atr_pct`
- `price_vs_ema21_pct`
- `price_vs_ema55_pct`
- `price_vs_ema100_pct`
- `price_vs_ema200_pct`
- `ema_alignment`
- `trend_bias`
- `trend_score`
- `structure_state`
- `swing_high`
- `swing_low`
- `is_pullback_to_value_area`
- `is_extended`
- `trigger_state`
- `notes`

### Market Regime

- `higher_timeframe_bias`
- `trend_strength`
- `volatility_state`
- `is_trend_friendly`

### Decision

- `action`: `LONG`, `SHORT`, `WAIT`
- `bias`: `bullish`, `bearish`, `neutral`
- `confidence`: 0-100
- `recommended_timing`: `now`, `wait_pullback`, `wait_confirmation`, `skip`
- `entry_zone`
- `stop_loss`
- `invalidation`
- `take_profit_hint`

### Reasoning

- `reasons_for`
- `reasons_against`
- `risk_notes`
- `summary`

### Diagnostics

- `strategy_config_snapshot`
- `score_breakdown`
- `vetoes`
- `conflict_signals`
- `uncertainty_notes`
- `setup_quality`
- `trigger_maturity`

### Raw Metrics

`raw_metrics` is a debugging payload and may evolve. Current keys include:

- `scorecard`
- `mid_timeframe_assessment`
- `trigger_assessment`
- `timeframe_debug`

## `GET /analysis/{analysis_id}`

Returns a single stored analysis result by id.

### Example

```bash
curl "http://127.0.0.1:8000/analysis/03f6c949f79846d09aae78b1012eda9f"
```

### Response

Same schema as `POST /analyze`.

## `GET /workspace/backtest-trades/btc-best`

返回当前 BTC 主线最佳版本的离线回测交易簿，供工作台直接展示每一笔历史交易。

### Response

`WorkspaceBacktestTradeBookResponse`，包含：

- 数据集来源与生成时间
- 当前展示的 `strategy_profile`
- 汇总指标：`total_trades / profit_factor / expectancy_r / cumulative_r / max_drawdown_r`
- 多空拆分汇总
- `trades`: 每笔交易的 `entry_time / side / entry_price / exit_price / stop_price / bars_held / exit_reason / tp1_hit / tp2_hit / pnl_r / cumulative_r_after_trade`

## `GET /analysis/{analysis_id}/diff/{comparison_analysis_id}`

Returns a structured comparison between two stored analyses.

### Response

`AnalysisDiffResponse` fields:

- `left`
- `right`
- `same_symbol`
- `same_exchange`
- `same_market_type`
- `same_strategy_profile`
- `compared_at`
- `decision`
- `market_regime`
- `diagnostics`
- `timeframes`
- `changed_sections`
- `total_change_count`
- `summary`

Each section diff contains:

- `changed`
- `change_count`
- `changed_fields`
- `highlights`

Each timeframe diff additionally includes:

- `timeframe`
- `signal_shift`

## `GET /analyses`

Returns stored analyses ordered by newest first.

### Query Parameters

- `limit`: default `50`, min `1`, max `200`
- `offset`: default `0`
- `symbol`: optional exact symbol filter
- `action`: optional `LONG` / `SHORT` / `WAIT`
- `bias`: optional `bullish` / `bearish` / `neutral`
- `strategy_profile`: optional exact strategy profile filter
- `from_time`: optional ISO8601 lower time bound
- `to_time`: optional ISO8601 upper time bound

### Response

```json
{
  "items": [
    {
      "analysis_id": "03f6c949f79846d09aae78b1012eda9f",
      "timestamp": "2026-03-18T12:00:00+00:00",
      "recorded_at": "2026-03-18T12:00:01+00:00",
      "symbol": "ETH/USDT:USDT",
      "exchange": "binance",
      "market_type": "perpetual",
      "strategy_profile": "trend_pullback_v1",
      "requested_timeframes": ["1d", "4h", "1h", "15m"],
      "action": "WAIT",
      "bias": "neutral",
      "confidence": 15,
      "recommended_timing": "skip",
      "higher_timeframe_bias": "neutral",
      "trend_strength": 15,
      "volatility_state": "normal",
      "is_trend_friendly": false,
      "summary": "Higher and mid timeframes are not aligned cleanly enough, so the setup is better treated as noise than as a tradable signal."
    }
  ],
  "total": 1,
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 1,
    "returned": 1,
    "has_more": false,
    "next_offset": null,
    "previous_offset": null
  }
}
```

## `GET /review`

Returns a built-in internal review page implemented with static HTML and browser-side JavaScript.

This page is intentionally minimal and consumes the same JSON APIs described above:

- `GET /analyses`
- `GET /analysis/{analysis_id}/diff/{comparison_analysis_id}`

## `GET /` and `GET /workspace`

Return the built-in analysis workspace.

This page is the main front-end entry for the MVP and uses:

- `GET /symbols`
- `POST /analyze`
- `GET /analyses`
- `GET /analysis/{analysis_id}`

## Notes

- The API is analysis-only.
- No order placement endpoints exist.
- No trading API keys are required.
- `WAIT` responses intentionally return `null` for `entry_zone`, `stop_loss`, and `take_profit_hint` to avoid implying an executable plan.
