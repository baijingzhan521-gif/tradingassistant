# Trading Analysis Assistant MVP

一个基于真实 OHLCV 行情数据、规则引擎和多时间周期分析的“半自动交易分析助手”后端 MVP。

它的目标不是自动交易，而是输出可解释、可复盘、可调试的交易结论：

- 当前结论：`LONG` / `SHORT` / `WAIT`
- 多时间周期趋势判断
- 做出判断的依据与反对理由
- 风险说明与不确定性
- 建议动作：立即观察、等待回踩、等待确认、放弃交易
- 入场区、止损位、失效位、止盈建议
- 置信度（0-100）

## 为什么这不是自动交易系统

这个项目明确不做以下事情：

- 不下单
- 不接入真实资金执行
- 不需要交易 API Key
- 不把 LLM 当成最终拍板者

当前 MVP 只负责“分析与决策建议”，并把输出保存到本地数据库，便于后续复盘和扩展。

## 当前开发记忆

如果要在新的对话窗口继续开发，不要只看 README 的概念说明，优先先看这份项目交接文档：

- 当前项目记忆与开发交接: [`docs/project_memory.md`](./docs/project_memory.md)
- 回测交接文档（给接手同事的完整指南）: [`HANDOFF.md`](./HANDOFF.md)

## 当前主线状态

这里先纠正一个常见误解：仓库里“可运行的 profile”很多，但不代表它们都还是当前主线。

- 当前主线候选：`swing_trend_long_regime_gate_v1`
- 当前主线范围：`BTC only + swing only`
- 当前主线默认周期：`1d / 4h / 1h`
- 当前推荐 exit：`LONG 1R -> 3R scaled`，`SHORT 1.5R fixed`
- `trend_pullback_v1`、`swing_trend_v1`、`intraday_mtf_v1` 仍保留，但不再代表当前最优主线
- divergence / free-space / ablation / 更复杂的 regime 变体都属于研究分支，不应再当默认策略理解

这里还要再补一个现实约束：

- 这条主线在 `2024-03-19 -> 2026-03-19` 的近两年 confirmed-swing / OOS 下成立
- 但拉长到 `2020-03-19 -> 2026-03-19` 后，只能算“薄 edge”，不能再把它当成长期 always-on 强主线
- 把这条 BTC 主线直接复制到 `ETH / SOL / AAVE / HYPE / BNB` 的组合后，公平归一化结果并没有优于 `BTC-only`

所以，如果你是新开窗口继续开发，先看 [`docs/project_memory.md`](./docs/project_memory.md) 里的最新交接结论，不要只凭 README 里的主线简介继续推进。

## 核心设计

### 分层

- `app/api/`: FastAPI 路由层，负责 HTTP 输入输出
- `app/services/`: 用例编排，串起行情、策略、持久化
- `app/data/`: 交易所客户端与 OHLCV 拉取
- `app/indicators/`: EMA、ATR、swing、结构判断
- `app/strategies/`: 规则引擎与评分卡
- `app/models/`: SQLAlchemy 持久化模型
- `app/schemas/`: 请求/响应 schema
- `app/vision/`: 未来图像分析接口预留
- `app/notifications/`: 未来推送接口预留

### 决策逻辑

当前默认主线策略是 `swing_trend_long_regime_gate_v1`：

1. `1D + 4H` 决定高周期方向环境
2. `1H` 判断是否回到 `EMA21 / EMA55` 执行区
3. `1H` 使用较简洁但仍有效的 trigger 核心：`regained_fast + held_slow + reversal candle`
4. 多头环境比空头更严格
5. exit 采用多空非对称，而不是默认 `fixed 2R`

旧 profile 仍保留，目的是兼容历史记录和研究复盘，不是因为它们仍然是当前默认实现。

### 可解释性

所有分析输出都包含：

- `reasons_for`
- `reasons_against`
- `risk_notes`
- `raw_metrics`
- `scorecard`

因此每次输出都可以回看“为什么给出这个结论”。

## 技术栈

- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- SQLite
- ccxt
- pandas / numpy
- pytest

## 快速开始

### 1. 准备 Python 3.11

不要依赖系统自带的 `/usr/bin/python3`。这个项目要求的稳定开发基线是 Python 3.11+。

如果本机已经有 `python3.11`：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果本机没有 `python3.11`，更稳妥的做法是用 `uv` 安装解释器：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

项目根目录提供了 [`.python-version`](./.python-version)，用于明确本地开发的 Python 主版本。

### 2. 配置环境变量

```bash
cp .env.example .env
```

可配置项：

- `DEFAULT_EXCHANGE`: 默认交易所，当前 MVP 建议 `binance`
- `DEFAULT_MARKET_TYPE`: 默认市场类型，当前只支持 `perpetual`
- `DEFAULT_LOOKBACK`: 默认拉取 K 线根数
- `DATABASE_URL`: 数据库连接串，默认 SQLite
- `STRATEGY_CONFIG_DIR`: 策略 YAML 配置目录
- `LOG_LEVEL`: 日志级别
- `CCXT_TIMEOUT_MS`: ccxt 请求超时
- `CCXT_MAX_RETRIES`: 网络/交易所错误时的重试次数
- `CCXT_RETRY_DELAY_MS`: 重试间隔
- `CCXT_TRUST_ENV`: 是否允许 ccxt 继承环境中的证书/代理相关配置
- `CCXT_HTTP_PROXY` / `CCXT_HTTPS_PROXY` / `CCXT_SOCKS_PROXY`: 交易所访问代理
- `BINANCE_USE_SANDBOX`: 是否改连 Binance futures testnet，仅用于开发联调
- `BINANCE_HOSTNAME`: 可选 hostname 覆盖，只在你明确知道自己需要特殊域名时再用

### 3. 启动服务

```bash
uvicorn app.main:app --reload
```

启动后访问：

- 主工作台: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- OpenAPI JSON: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)
- 内建复盘页: [http://127.0.0.1:8000/review](http://127.0.0.1:8000/review)
- 独立系统设计文档: [`docs/system_design.md`](./docs/system_design.md)
- 独立 API 文档: [`docs/api.md`](./docs/api.md)
- 当前项目记忆与开发交接: [`docs/project_memory.md`](./docs/project_memory.md)

## 主前端入口

当前已经内建一个可视化工作台，不需要额外起前端项目。

- `/` 或 `/workspace`：主分析工作台，包含离线 BTC 回测交易簿面板
- `/review`：历史复盘与 diff 对比页

建议的使用路径：

1. 先打开 `/`
2. 在左侧表单输入交易对并发起分析
3. 在主区域查看 `decision / market_regime / reasoning / diagnostics`
4. 在左侧最近分析中回看已落库的结果
5. 需要做两次分析对比时，再进入 `/review`

## Binance 联通性说明

这里有一个常见误区：`ccxt.binance(..., options.defaultType=future)` 不等价于“已经正确建模 Binance USDT 永续市场”。对这个项目，更合理的默认实现是直接使用 `ccxt.binanceusdm`，因为它只加载 USDT 线性合约市场，避免在 `load_markets()` 时混入不必要的现货元数据请求。

当前代码已经切换到 `binanceusdm`。但如果你的网络环境会重置 `api.binance.com` / `fapi.binance.com` 的 TLS 握手，那么根因是网络可达性，而不是策略代码。

诊断命令：

```bash
python scripts/check_exchange_connectivity.py
```

这个脚本会检查：

- Python 版本
- 直连 Binance 是否可达
- 环境变量代理是否生效
- 本地常见代理端口是否在监听
- `ccxt.binanceusdm` 的 `load_markets()` 是否能跑通

如果你本地有代理进程但 shell 没有设置代理变量，诊断脚本会优先提示这个问题。

### 关于 sandbox

`BINANCE_USE_SANDBOX=true` 只适合开发联调，不适合生成真实市场结论。testnet 数据不能替代主网真实 OHLCV。

### 关于代理

如果你所在网络环境无法访问 Binance futures 主网，又必须保留 Binance perpetual 作为默认数据源，现实可行方案通常只有两种：

- 给 ccxt 配置可用代理
- 把服务部署到能稳定访问 Binance 主网的网络环境

“只改一点代码就让被网络阻断的主网 magically 可用”不是成立的前提。

我在这台机器上确认到一个具体例子：本地 `MonoProxy` 已经监听 `127.0.0.1:8118`，但 shell 里没有任何 `HTTP(S)_PROXY` 变量，所以应用默认仍然是裸连 Binance。像这种情况，问题在环境接线，不在策略逻辑。

## API 概览

### `GET /health`

健康检查。

### `GET /symbols`

列出当前支持的 Binance 永续 USDT 交易对。

示例：

```bash
curl "http://127.0.0.1:8000/symbols?exchange=binance&market_type=perpetual&limit=20"
```

### `POST /analyze`

`POST /analyze` 现在默认走主线候选 `swing_trend_long_regime_gate_v1`。它至少需要 `["1d", "4h", "1h"]`。  
`trend_pullback_v1` 这类旧 profile 仍可调用，但属于兼容/研究口径，不应再被当默认策略理解。

请求体：

```json
{
  "symbol": "ETH/USDT:USDT",
  "market_type": "perpetual",
  "exchange": "binance",
  "timeframes": ["1d", "4h", "1h", "15m"],
  "strategy_profile": "trend_pullback_v1",
  "lookback": 300
}
```

示例：

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ETH/USDT:USDT",
    "market_type": "perpetual",
    "exchange": "binance",
    "timeframes": ["1d", "4h", "1h", "15m"],
    "strategy_profile": "trend_pullback_v1",
    "lookback": 300
  }'
```

响应摘要示例：

```json
{
  "analysis_id": "bff62e33c8d74844ad8f305da23e2e20",
  "timestamp": "2026-03-18T12:00:00Z",
  "symbol": "ETH/USDT:USDT",
  "exchange": "binance",
  "market_type": "perpetual",
  "strategy_profile": "trend_pullback_v1",
  "timeframes": {
    "1d": {"timeframe": "1d"},
    "4h": {"timeframe": "4h"},
    "1h": {"timeframe": "1h"},
    "15m": {"timeframe": "15m"}
  },
  "market_regime": {
    "higher_timeframe_bias": "bullish",
    "trend_strength": 78,
    "volatility_state": "normal",
    "is_trend_friendly": true
  },
  "decision": {
    "action": "LONG",
    "bias": "bullish",
    "confidence": 74,
    "recommended_timing": "now",
    "entry_zone": {
      "low": 3520.5,
      "high": 3568.2,
      "basis": "1H EMA21/EMA55 pullback zone in bullish context"
    },
    "stop_loss": {
      "price": 3478.1,
      "basis": "Below recent 1H swing low with ATR buffer"
    },
    "invalidation": "Bullish thesis is invalid if 1H closes below the recent swing low and loses the 1H value area decisively",
    "take_profit_hint": {
      "tp1": 3625.7,
      "tp2": 3710.4,
      "basis": "1R/2R projection and prior 4H swing high"
    }
  },
  "diagnostics": {
    "strategy_config_snapshot": {
      "atr_period": 14
    },
    "score_breakdown": {
      "base": 50,
      "total": 74,
      "contributions": [
        {
          "label": "1d_4h_bias",
          "points": 15,
          "note": "1D and 4H both show bullish trend conditions"
        }
      ]
    },
    "vetoes": [],
    "conflict_signals": [],
    "uncertainty_notes": [
      "15m trigger is only partially formed, so confirmation is still immature"
    ],
    "setup_quality": {
      "higher_timeframe_bias": "bullish",
      "trend_friendly": true,
      "mid_timeframe_aligned": true,
      "mid_timeframe_pullback_ready": true,
      "mid_timeframe_extended": false,
      "one_hour_distance_to_value_atr": 0.22
    },
    "trigger_maturity": {
      "timeframe": "15m",
      "state": "bullish_confirmed",
      "score": 10,
      "supporting_signals": [
        "1H regained EMA21 with improving short-term structure"
      ],
      "blocking_signals": []
    }
  },
  "reasoning": {
    "reasons_for": [
      "1D and 4H both hold bullish conditions above EMA200 with constructive EMA alignment",
      "1H price is back near the EMA21/EMA55 value area instead of being far extended",
      "1H regained EMA21 with improving short-term structure"
    ],
    "reasons_against": [
      "15m local structure is still messy"
    ],
    "risk_notes": [
      "15m trigger is only partially formed, so confirmation is still immature"
    ],
    "summary": "Higher timeframes remain bullish, 1H is in a constructive pullback zone, and 15m has confirmed enough strength for a conditional long."
  }
}
```

### `GET /analysis/{id}`

根据 `analysis_id` 查询单次完整分析。

### `GET /analysis/{id}/diff/{comparison_id}`

对比两次历史分析，返回结构化 diff，重点包括：

- `decision`
- `market_regime`
- `diagnostics`
- 各时间周期的 `timeframes`
- `changed_sections`
- `total_change_count`
- `summary`

这条接口是正式对比契约。内部 `/review` 页面也直接消费它，不再维护第二套 ad hoc compare schema。

### `GET /analyses`

分页查看历史分析记录，并支持常用复盘筛选：

- `symbol`
- `action`
- `bias`
- `strategy_profile`
- `from_time`
- `to_time`

返回中包含分页元数据、请求时的时间周期、`recommended_timing` 和市场环境摘要，方便做复盘列表页。

### `GET /review`

内建的极简复盘页面，基于原生 HTML/JS，适合本地开发和内部审查。它支持：

- 拉取 `/analyses` 做历史筛选
- 选择两个 `analysis_id`
- 调用正式 diff 接口展示结构化差异

它不是正式前端产品，也没有鉴权，不应直接作为公网控制台使用。

错误响应示例：

```json
{
  "detail": "Strategy 'trend_pullback_v1' requires timeframes: 1d",
  "timestamp": "2026-03-18T12:00:00+00:00"
}
```

## 数据库

默认数据库是 SQLite，表 `analysis_records` 会保存：

- `analysis_id`
- 请求参数快照
- 完整分析 JSON
- 行动结论、方向、置信度、摘要

后续迁移到 PostgreSQL 时，服务层不需要大改。

## 策略配置覆盖

默认配置在：

- [`config/strategies/trend_pullback_v1.yaml`](./config/strategies/trend_pullback_v1.yaml)

系统会先加载内置默认值，再用 YAML 覆盖。这样后续可以把知识库规则逐步外置，而不是把逻辑写死在代码里。

## 测试

```bash
.venv/bin/python -m pytest
```

测试目标：

- 健康检查接口
- 指标计算
- 规则策略输出
- fixture 驱动的策略回归基线
- API 分析接口（通过 mock/override，避免依赖真实网络）
- `/review` 内建复盘页的最小浏览器级 smoke

如果你要跑浏览器 smoke，还需要安装 Playwright 浏览器二进制：

```bash
.venv/bin/python -m playwright install chromium
```

浏览器 smoke 是单独的 `browser` marker，用来做 `/review` 页面的最小端到端检查。它会在 Chromium 不可用时自动跳过，不影响主测试集。

原始回归样本现在直接保存在 [`tests/fixtures/trend_pullback_v1_baselines.json`](./tests/fixtures/trend_pullback_v1_baselines.json) 中，包含 `1d/4h/1h/15m` 的 raw OHLCV dump，而不是只保留参数化生成规则。

## Demo 脚本

```bash
python scripts/demo_analyze.py --symbol ETH/USDT:USDT --base-url http://127.0.0.1:8000
```

脚本支持 `--base-url`、`--symbol`、`--exchange`、`--market-type`、`--strategy-profile`、`--lookback` 和 `--timeout` 参数；当服务未启动或返回错误状态时，会给出更明确的错误信息。

## 后续扩展方向

- 接入图像截图分析：`app/vision/`
- 接入 Telegram / Discord / 企业微信推送：`app/notifications/`
- 新增 YAML/JSON 策略知识库
- 加回测模块与信号复盘模块
- 增加前端仪表盘或 Streamlit 页面

## 已知边界

- 当前只支持 Binance perpetual
- 当前策略只有 `trend_pullback_v1`
- 当前没有回测，不应把输出当成收益承诺
- 当前没有订单执行功能
