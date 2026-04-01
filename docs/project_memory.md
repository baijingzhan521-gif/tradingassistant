# 项目记忆与开发交接

更新时间：2026-03-28

这份文档不是产品宣传页，而是给后续开发对话使用的“项目真实状态快照”。如果下一轮开发要继续推进，应优先以这里的内容为准，而不是依赖聊天上下文残留记忆。

## 1. 当前项目是什么

这是一个基于真实 OHLCV 数据的交易分析与回测系统，当前范围仍然是：

- 不自动下单
- 不接真实资金执行
- 不需要交易 API key
- 核心拍板依赖规则引擎，不依赖黑箱 LLM

系统已经具备三层能力：

- 在线分析：FastAPI + 内建工作台，能对单个标的给出结构化交易建议
- 历史复盘：保存分析记录，可查看历史、diff 对比、review 页面
- 离线回测：对规则策略跑两年级别的历史回放，并输出 JSON / CSV 报告

## 2. 当前主入口

服务启动后：

- 主工作台：`/` 或 `/workspace`
- 复盘页面：`/review`
- Swagger：`/docs`
- 健康检查：`/health`

工作台当前会同时展示两套窗口结果，不需要手动切换模式：

- 日内参考窗口：`intraday_mtf_v1`
- 波段主线窗口：`swing_trend_long_regime_gate_v1`

这里要特别澄清：`intraday_mtf_v1` 仍保留在工作台里，更多是参考视角，不代表它已经被证明有稳定 edge。

## 3. 代码结构里最重要的文件

### API / 前端

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/main.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/static/workspace.html`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/static/review.html`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/api/routes_workspace.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/api/routes_workspace_analysis.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/api/routes_history.py`

### 策略 / 规则引擎

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/windowed_mtf.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_long_regime_gate_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/intraday_mtf_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/intraday_mtf_v2.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/services/strategy_service.py`

### 回测 / 诊断

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/backtesting/service.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/backtesting/diagnostics.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_backtest.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/diagnose_backtest_phases.py`

## 4. 当前策略口径

### 4.1 均线与周期

当前有效策略统一使用：

- `EMA21`
- `EMA55`
- `EMA100`
- `EMA200`

图表支持：

- `1D`
- `4H`
- `1H`
- `15m`
- `3m`

其中：

- 日内策略主决策使用 `1H / 15m / 3m`
- 波段策略主决策使用 `1D / 4H / 1H`
- `3m` 已进入日内策略逻辑，不再只是纯图表补充

### 4.2 当前策略集合

#### `swing_trend_long_regime_gate_v1`

- 当前主线候选
- 高周期：`1D + 4H`
- setup / trigger：`1H`
- entry 核心：`EMA21/55` 回踩 + `regained_fast + held_slow + reversal candle`
- 推荐 exit：`LONG 1R -> 3R scaled`，`SHORT 1.5R fixed`
- 当前是 no-lookahead confirmed-swing 和 walk-forward 下证据最强的主线

#### 研究分支 / 不再当主线的复杂 profile

- `swing_trend_divergence_v1`
- `swing_trend_long_divergence_gate_v1`
- `swing_trend_simple_candidate_v2`
- `swing_trend_axis_band_diagnostic_v1`
- `swing_trend_confluence_setup_v1`
- `swing_trend_structure_gate_hard_v1`
- `swing_trend_confluence_structure_gate_hard_v1`
- `swing_trend_level_aware_confirmation_v1`
- `swing_trend_axis_band_state_note_v1`
- `swing_trend_long_regime_short_relaxed_trigger_v1`
- `swing_trend_long_regime_short90_free_space_v1`
- `swing_trend_ablation_*`
- `swing_trend_simple_candidate_v1`
- `intraday_mtf_v2`

这些 profile 保留的原因是研究可追溯，不是因为它们仍然推荐继续当默认策略使用。

其中还要特别说明：

- `swing_trend_axis_band_state_note_v1` 只是解释层 profile，不是策略增强版
- 旧名字 `swing_trend_axis_band_risk_overlay_v1` 仍保留，但只作为兼容别名；语义上已经降级成 `state_note`

#### `swing_trend_v1`

- 高周期：`1D + 4H`
- 执行/触发：`1H`
- 目标：波段/趋势交易
- 现在更适合当历史基线，不再是当前主线候选

#### `intraday_mtf_v1`

- 高周期：`1H`
- setup：`15m`
- trigger：`3m`
- 目标：日内短持仓
- 当前仍无稳定 edge，不建议当主线

#### `intraday_mtf_v2`

- 在 `intraday_mtf_v1` 基础上做更严格过滤
- 加了更严的微观确认、更窄执行区、更强结构要求和 cooldown
- 回测证明“更严”没有换来更好的质量
- 当前不建议继续在这个版本上做小修小补

#### `trend_pullback_v1`

- 保留在代码中，主要是早期兼容与基线策略
- 已不应视为当前主工作台的核心策略

## 5. 前端当前真实状态

### 已实现

- 中文化工作台
- 同时展示日内和波段两栏
- 内嵌 BTC 主线回测交易簿面板
- 多周期 K 线图
- EMA21/55/100/200 四线
- 图表悬停 tooltip
- 1H 图上执行位覆盖层
- 横向缩放
- `执行位聚焦` 纵向视图
- 历史分析加载与 diff 复盘

### 仍然不够好的地方

- 图表仍是 SVG 自绘，功能够用，但不是专业级图表组件
- 方向结论和执行结论仍耦合在同一个 `action` 视图里，用户容易把“偏多但等确认”理解成“没有结论”
- 某些旧历史分析记录不带最新图表快照字段，页面会做兼容提示

## 6. 回测系统当前真实状态

### 已支持

- 拉取并缓存区间历史 OHLCV
- 多策略批量回测
- 默认两年窗口
- 输出 JSON / CSV
- 两种止盈模型：
  - 分批 `1R / 2R`
  - 固定 `1.5R` 全平
  - 固定 `2R` 全平

### 关键假设

- 信号在当前 bar 收盘形成
- 下一根 trigger bar 开盘进场
- 单边手续费默认 `5 bps`
- 单边滑点默认 `2 bps`
- 同一标的同一策略同一时刻只持有一笔
- 同一根 K 线触发止损和止盈时按保守处理，优先算止损

### 当前要点

- 回测是“可审计基线”，不是撮合级仿真
- 没有 funding、订单簿、真实滑点建模
- 不能把回测数字直接当成实盘承诺

## 7. 关键回测结论

### 7.0 当前主线候选的最新状态

当前主线已经收敛到 `swing_trend_long_regime_gate_v1`，而且这个结论只在 **confirmed swing / no-lookahead** 口径下成立。

当前主线的关键规则：

- 标的：`BTC/USDT:USDT`
- 周期：`1D + 4H + 1H`
- 高周期方向过滤：`1D + 4H`
- 执行区：`1H EMA21/55`
- trigger 核心：`regained_fast + held_slow + reversal candle`
- 非对称环境阈值：`LONG` 比 `SHORT` 更严格
- 当前推荐 exit：`LONG 1R -> 3R scaled`，`SHORT 1.5R fixed`

要特别澄清：

- `simple_candidate_v2` 不再是主线
- `swing_trend_v1` 只保留为旧基线
- 现在所有“主线成立”的结论，都应该默认理解为 **confirmed swing**，不是早期 centered swing

### 7.1 旧两年基线为什么被放弃

旧基线 `swing_trend_v1 + fixed 2R` 在两年三币上并不稳，主要作用是提供历史对照，不是当前主线证据。

两年三币旧结果：

- 时间窗口：`2024-03-18 -> 2026-03-18`
- `swing_trend_v1`：`270` 笔，`PF 0.9932`，`-1.2405R`
- `intraday_mtf_v1`：`3743` 笔，`PF 0.7962`，`-496.3054R`
- `intraday_mtf_v2`：`1044` 笔，`PF 0.5638`，`-374.0199R`

这一步真正带来的判断是：

- 日内线没有 edge，不应继续当主优化方向
- 旧 BTC 波段 fixed `2R` 里，`SHORT` 明显强于 `LONG`
- 统一对称规则不成立

### 7.2 当前主线在两年 confirmed-swing 下的结果

当前最强的 in-sample 主线证据，不是 fixed `2R`，而是：

- profile：`swing_trend_long_regime_gate_v1`
- exit：`LONG 1R -> 3R scaled`，`SHORT 1.5R fixed`
- swing 模式：`confirmed`
- 时间窗口：`2024-03-19 -> 2026-03-19`

结果：

- `77` 笔
- `PF 1.6872`
- `expectancy 0.2772R`
- `cumulative +21.3461R`
- `max drawdown 5.4578R`
- `LONG +14.6742R`
- `SHORT +6.6719R`

这一步证明的是：在近两年样本里，这条主线是当前证据最强的版本。

### 7.3 当前主线在两年 walk-forward / OOS 下的结果

两年 OOS 设定：

- `365d train / 90d test / 90d step`
- `4/4` fold
- 只比较主线相关候选

结果：

- `4/4` 个 fold 全部选中 `swing_trend_long_regime_gate_v1@long_scaled1_3_short_fixed1_5`
- OOS 总体：`48` 笔，`PF 1.4839`，`expectancy 0.2172R`，`+10.4275R`

这一步证明的是：在近两年视角下，当前主线不是纯 in-sample 幻觉。

### 7.4 把 BTC 主线直接复制到多标的组合的结果

这里有一个用户很容易误判的点：`raw 累计R` 大，不代表组合更好。必须看等权归一化后的结果。

固定同一条 BTC 主线，直接复制到：

- `BTC`
- `ETH`
- `SOL`
- `AAVE`
- `HYPE`
- `BNB`

然后做 multi-asset walk-forward，结果是：

- `BTC-only`：等权 `+10.43R`
- `BTC+ETH+SOL`：等权 `+1.25R`
- `alts5`：等权 `-0.03R`
- `full6`：等权 `+1.72R`

单标的 OOS 里：

- `BTC`：`+10.43R`
- `ETH`：`-5.08R`
- `SOL`：`-1.60R`
- `AAVE`：`+0.18R`
- `HYPE`：`+3.98R`
- `BNB`：`+2.40R`

结论非常明确：

- **把 BTC 主线直接复制到多标的组合后，组合并没有更好**
- 当前不应该把“多标的复制”升格成主线

### 7.5 把 BTC 主线样本拉长到 `2020-03-19 -> 2026-03-19` 之后的结果

这一步很关键，因为它直接改变了对主线的信心判断。

#### 全窗口 always-on 回测

- `271` 笔
- `PF 1.024`
- `expectancy 0.0126R`
- `cumulative +3.4107R`
- `max drawdown 25.9814R`

这不是“失效”，但只能叫 **勉强为正**，而不是强 edge。

#### 同口径 walk-forward / OOS

设定：

- `365d train / 90d test / 90d step`
- 总共 `20` 个 fold

结果：

- 只有 `9/20` 个 fold 训练窗为正并允许开策略
- `11/20` 个 fold 直接跳过
- 被选中的 `9` 个 OOS fold 合计：
  - `82` 笔
  - `PF 1.3419`
  - `expectancy 0.1477R`
  - `cumulative +12.1136R`
  - `max drawdown 6.7178R`

更重要的是：

- `2021` 早期两个有效 OOS fold 仍然亏损：`-4.80R`、`-2.05R`
- 真正连续像样的阶段，明显是从 `2024` 附近开始

这一步带来的工程判断不是“主线彻底无效”，而是：

- **这条 BTC 主线更像 regime-dependent 策略，不像 2020 以来一直稳定的 always-on 通用主线**

### 7.6 当前工程判断

截至 `2026-03-20`，当前最可靠的判断是：

- 暂停把日内策略当作主要优化对象
- 暂停把多标的复制当作当前主方向
- 保留 BTC 主线，但不要再把它理解为“长期 always-on 强策略”
- 当前项目已经从“继续调 entry/filter”转成“重新决定研究路线”的阶段

### 7.7 `2020-01-01 -> 2026-03-19` 铁律下的三候选验证

从这轮开始，要再补一个工程铁律：

- **后续主候选验证默认从 `2020-01-01` 起跑，到最新可用日期结束**
- 近两年窗口仍可保留，但只作为 secondary regime diagnostic，不再用来单独拍板长期主线

本轮只保留三条 entry 候选继续验证：

1. `swing_trend_long_regime_gate_v1` = `R1 RF1 HS1 AUX1`
2. `swing_trend_ablation_no_auxiliary_v1` = `R1 RF1 HS1 AUX0`
3. `swing_trend_simple_candidate_v2` = `R0 RF1 HS1 AUX0`

#### 长样本全窗口结果

窗口：`2020-01-01 -> 2026-03-19`

- `swing_trend_long_regime_gate_v1`：`272` 笔，`PF 1.0341`，`expectancy 0.0178R`，`+4.8353R`，`DD 25.9814R`
- `swing_trend_ablation_no_auxiliary_v1`：`270` 笔，`PF 1.0304`，`expectancy 0.0160R`，`+4.3111R`，`DD 23.6457R`
- `swing_trend_simple_candidate_v2`：`288` 笔，`PF 1.2305`，`expectancy 0.1100R`，`+31.6906R`，`DD 12.1964R`

这里要特别防一个误判：

- `simple_candidate_v2` 在长样本 **全窗口** 上明显更好
- 但这一步还只是 in-sample，不能据此直接升格成主线

#### 三候选池 walk-forward / OOS

窗口：`2020-01-01 -> 2026-03-19`

- 有效 OOS fold：`18 / 21`
- OOS：`203` 笔，`PF 1.1541`，`expectancy 0.0748R`，`+15.1847R`，`DD 13.6155R`

选择频次：

- `swing_trend_simple_candidate_v2`：`11` 次
- `swing_trend_long_regime_gate_v1`：`7` 次
- `swing_trend_ablation_no_auxiliary_v1`：`0` 次

这一步说明的是：

- `simple_candidate_v2` 更像覆盖 `2020 -> 2024H1` 的候选
- `gate_v1` 更像覆盖 `2024H2 -> 2026Q1` 的候选
- `no_auxiliary` 虽然和 `gate_v1` 很接近，但在候选池选优里没有单独胜出

#### 单候选独立 walk-forward / OOS

同样窗口：`2020-01-01 -> 2026-03-19`

- `swing_trend_long_regime_gate_v1`
  - 有效 OOS fold：`11 / 21`
  - OOS：`101` 笔，`PF 1.2744`，`expectancy 0.1298R`，`+13.1103R`，`DD 9.8498R`
- `swing_trend_ablation_no_auxiliary_v1`
  - 有效 OOS fold：`11 / 21`
  - OOS：`101` 笔，`PF 1.2770`，`expectancy 0.1309R`，`+13.2258R`，`DD 9.8498R`
- `swing_trend_simple_candidate_v2`
  - 有效 OOS fold：`15 / 21`
  - OOS：`168` 笔，`PF 0.9751`，`expectancy -0.0127R`，`-2.1392R`，`DD 13.6155R`

这一步带来的判断比全窗口更重要：

- **`simple_candidate_v2` 虽然长样本全窗口很好看，但单候选 OOS 为负，不能直接升格成长期主线**
- **`no_auxiliary` 相比 `gate_v1` 有极轻微优势，但差距非常小，还不够大到支持直接替换**
- 当前最稳妥的单线判断仍然是：`gate_v1 ~ no_auxiliary`
- 如果后续要继续推进“`simple_candidate_v2` 早期 + `gate_v1` 后期”的思路，那已经不是单策略验证，而是 **部署规则 / regime switch** 问题

### 7.8 `2026-03-23 -> 2026-03-24` 原框架整合尝试的结论

这一轮不是重写主线，而是尝试把“环境 + 位置 + K 线确认”的新框架，轻量接回当前主线里最薄弱的位置层。

结论先说：

- **`EMA55 + pivot + band` 的 confluence 路线目前不成立**
- **`level-aware confirmation` 近两年好看，但长样本不成立**
- **`axis_distance_z / band_position` 目前更像状态注释，不像风险管理规则**

#### `EMA55 + pivot + band` confluence 路线

做过三层验证：

1. 只替换 setup 解释层
2. `require_structure_ready=True` 的硬 A/B
3. 把 `pivot / band / EMA55` 拆成连续因子做单调性研究

结果是：

- setup 语义虽然变了，但在默认主线里没有改变最终交易决策
- 一旦做成硬 gate，样本会急剧塌缩；`confluence_structure_gate_hard_v1` 两年窗口只剩 `3` 笔，不能当作稳健增强
- 连续因子里只有 `pivot_anchor_gap` 有一点残余信息，但样本不足；`ema55_anchor_gap / band_anchor_gap / confluence_spread` 当前证据不支持“更紧更好”

工程判断：

- **不要继续调 confluence 的 proximity / min_hits / spread 阈值**
- 这条路现在应视为 **已停止分支**

#### `level-aware confirmation`

这条分支的定义是：

- 不改环境层
- 不改 exit
- 只把 `reversal_candle` 从“泛化 reversal”改成“必须发生在 `EMA55 / pivot / band` 附近的 reclaim/rejection”

结果：

- 近两年 `2024-03-19 -> 2026-03-19`：
  - 原主线：`90` 笔，`PF 1.39`，`+15.28R`
  - `level_aware_confirmation_v1`：`79` 笔，`PF 1.60`，`+19.98R`
- 长样本 `2020-03-19 -> 2026-03-19`：
  - 原主线：`299` 笔，`PF 1.0331`，`+5.2118R`，`DD 23.8605R`
  - `level_aware_confirmation_v1`：`276` 笔，`PF 1.0325`，`+4.8055R`，`DD 29.1790R`

这一步真正说明的是：

- 它不是“提纯原交易集”，而是在明显**改写交易集合**
- `2022-2023` 的变差，主要来自“跳过更早、更好的交易，转而放出更晚、更差的替代交易”
- bull-regime split 也没有支持它成为 bull 变体；那轮分层在当前交易集上几乎退化成 long/short 切分，信息量不足

工程判断：

- **`level-aware confirmation` 可以保留为研究结论，但不能升格成主线**
- 当前最合理的定位是：`two-year friendly / long-sample unproven`
- 这条分支也应视为 **冻结分支**

#### `axis_distance_z / band_position` 风险标签

原始想法是把：

- `pullback_risk`
- `rebound_risk`

当作主线上的轻量风险提示，然后再看能不能演化成减仓或管理层。

但轻量标签研究结果不支持它成为“风险规则”：

- `full_2020 / LONG` 下，`pullback_risk` active 交易反而 **expectancy 更高、MAE 更低**
- `full_2020 / SHORT` 下，`rebound_risk` active 交易也 **expectancy 更高、MAE 更低**
- `two_year` 里只有少数路径指标像风险提示，但和最终交易表现并不稳定同向

这说明：

- 这两个标签当前更像“扩张 / 状态强弱”说明，不像可靠的短线风险警示
- 不应该再把它们往减仓、止盈、禁追或 hard gate 上推

因此这条线已经正式降级为：

- `swing_trend_axis_band_state_note_v1`

它的正确用途是：

- 解释层 `state_note`
- 诊断层 `uncertainty_note`

而不是：

- `risk_overlay`
- 管理规则
- 主线增强器

### 7.9 derivatives / carry 路线的最新状态

这一轮还做了独立于主线的衍生品状态层和 carry/basis 路线验证。这里也要明确记忆，不然下一轮很容易重复走老路。

#### derivatives state 层

基于 Bybit 小时面板做过：

- `funding z-score`
- `OI change`
- `mark-index premium`
- `basis`

对未来 `1h / 4h / 24h` 收益和波动的严格状态研究。

结论：

- 这些特征**更像状态层和波动层，不像方向 alpha**
- `OI change` 对未来波动最有解释力
- `funding / premium / basis` 有一些条件性，但不够强，也不够单调
- 不应再强行把它们硬塞进 directional strategy

#### carry / basis

后续做了：

- 最小 `BTC spot-perp carry / basis` 原型
- 更真实的 execution-sensitive carry 原型
- 更诚实的 refined 成本 / 资金口径核验

结论最终收敛为：

- gross edge 不是完全没有
- 但一旦把显式 slippage、机会成本、资金占用加进去，年化 ROC 会迅速掉到接近 `0`，甚至直接转负
- **当前只能把 carry/basis 视为 research-only，不应包装成可部署 alpha**

工程判断：

- derivatives state 层可以继续当研究基础设施
- carry / basis 这条线当前应视为 **冻结的 research-only 分支**

## 8. 诊断工具与产物位置

### 当前主线的近两年核心产物

- confirmed-swing 主线删条件矩阵：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/btc_confirmed_swing_ablation/confirmed_swing_ablation_20260319T132510Z.md`
- 近两年主线 walk-forward / exit pool：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/btc_walk_forward_exit_pool/walk_forward_report_20260319T124823Z.md`

### 当前主线的长样本核心产物

- `2020-03-19 -> 2026-03-19` 全窗口正式回测：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/btc_mainline_2020_window/backtest_btc_mainline_2020_confirmed_long_scaled1_3_short_fixed1_5_20260320T034836Z.json`
- `2020-03-19 -> 2026-03-19` walk-forward / OOS：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/btc_walk_forward_2020_mainline/walk_forward_report_20260320T034729Z.md`

### 原框架整合尝试产物

- `level-aware confirmation` 两年对照：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/level_aware_confirmation_compare_two_year/backtest_long_scaled1_3_short_fixed1_5_20260323T132150Z.json`
- `level-aware confirmation` 长样本对照：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/level_aware_confirmation_compare_full_2020/backtest_long_scaled1_3_short_fixed1_5_20260323T135105Z.json`
- `2022-2023` 熊市 diff：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/level_aware_confirmation_bear_diff_2022_2023/report.md`
- `axis/band` 轻量标签研究：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/axis_band_risk_label_study/report.md`

### derivatives / carry 路线产物

- derivatives 严格状态研究：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/derivatives/bybit_btcusdt_strict_state_study/report.md`
- carry/basis refined execution 研究：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/derivatives/bybit_btcusdt_carry_basis_execution_refined/report.md`

### 多标的复制路线产物

- multi-asset 组合 walk-forward：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/multi_asset_portfolio_walk_forward/multi_asset_walk_forward_report_20260320T033200Z.md`
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/multi_asset_portfolio_walk_forward/multi_asset_walk_forward_20260320T033200Z.json`

## 9. 已知问题与技术债

### 9.1 研究层问题

- 当前 BTC 主线在近两年样本里成立，但拉长到 `2020 -> 2026` 后只剩很薄的 always-on edge
- 多标的复制路线已经首轮证伪，不能默认“多加几个币，组合就更优”
- `confidence` 仍更像解释层，不像真正强分层的前置过滤器
- 现在继续在 entry/filter 上加条件，过拟合风险高于增益预期
- `EMA55 + pivot + band` 这条 confluence 分支当前不成立，不应继续靠阈值微调强救
- `level-aware confirmation` 当前只算研究分支，不应再当主线升级候选
- `axis_distance_z / band_position` 当前只能当 `state_note`，不应再往风险规则或管理动作上推
- derivatives 特征当前更像 state layer，不像独立方向 alpha
- carry/basis 当前只应视为 research-only，不应误写成 deployment candidate

### 9.2 数据与方法边界

- `HYPE` 历史不完整，只从 `2025-05-30` 开始有数据；不能把它和 BTC 的长样本强行等价
- multi-asset 组合研究里的 `scaled_*` 已经比 `raw_*` 更公平，但仍不是完整逐时盯市组合模型
- 回测环境里持续出现 `fatal: bad revision 'HEAD'` 噪音；当前不影响结果产出，但来源未定位

### 9.3 产品层问题

- 前端仍把方向与执行耦合展示，容易让“有方向但不该开仓”看起来像“没有判断”
- 工作台目前没有把“策略是否适合部署”单独显示成状态层

## 10. 下个对话建议从哪里接着做

如果下一轮开发要继续，建议先明确选一条路线，而不是默认沿着旧对话继续：

1. `冻结 BTC 主线，重开一个真正不同的策略原型`
2. `把 BTC 主线做成显式部署规则（何时启用 / 何时停用）`
3. `重做多标的路线，但先做 symbol admission，不再直接复制 BTC 规则`

不建议默认继续的方向：

- 不要继续在 `intraday_mtf_v2` 上做小阈值微调
- 不要继续把多标的复制当作已被证明的正确路线
- 不要再把 `2024 -> 2026` 的两年结果误读成“长期 always-on 已经成立”
- 不要再只拿近两年窗口给长期主线拍板；默认先看 `2020-01-01 -> 最新`
- 不要再默认继续 `confluence gate / level-aware confirmation / state_note` 这条原框架整合支线
- 不要再把 derivatives state 特征硬塞回 directional strategy
- 不要把 carry/basis 的 research proxy 误当成可部署执行结果

## 11. 常用命令

启动服务：

```bash
cd /Users/memelilihuahua/量化agent构建/trading-assistant
.venv311/bin/python -m uvicorn app.main:app --reload
```

主回归测试：

```bash
cd /Users/memelilihuahua/量化agent构建/trading-assistant
.venv311/bin/pytest -q -k 'not smoke'
```

跑近两年主线 walk-forward：

```bash
cd /Users/memelilihuahua/量化agent构建/trading-assistant
.venv311/bin/python scripts/run_walk_forward.py \
  --start 2024-03-19 \
  --end 2026-03-19 \
  --candidates swing_trend_long_regime_gate_v1@long_scaled1_3_short_fixed1_5 \
  --output-dir artifacts/backtests/btc_walk_forward_exit_pool
```

跑 `2020 -> 2026` BTC 主线正式回测：

```bash
cd /Users/memelilihuahua/量化agent构建/trading-assistant
.venv311/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
from app.backtesting.service import BacktestService, BacktestAssumptions
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService

service = BacktestService(
    ohlcv_service=OhlcvService(get_exchange_client_factory()),
    strategy_service=StrategyService(),
    assumptions=BacktestAssumptions(
        exit_profile='btc_mainline_2020_confirmed_long_scaled1_3_short_fixed1_5',
        take_profit_mode='scaled',
        scaled_tp1_r=1.0,
        scaled_tp2_r=3.0,
        long_exit={'take_profit_mode': 'scaled', 'scaled_tp1_r': 1.0, 'scaled_tp2_r': 3.0},
        short_exit={'take_profit_mode': 'fixed_r', 'fixed_take_profit_r': 1.5},
        swing_detection_mode='confirmed',
    ),
)
report = service.run(
    exchange='binance',
    market_type='perpetual',
    symbols=['BTC/USDT:USDT'],
    strategy_profiles=['swing_trend_long_regime_gate_v1'],
    start=datetime(2020, 3, 19, tzinfo=timezone.utc),
    end=datetime(2026, 3, 19, tzinfo=timezone.utc),
)
service.save_report(report, Path('artifacts/backtests/btc_mainline_2020_window'))
PY
```

跑 `2020-01-01 -> 2026-03-19` 三候选长样本 walk-forward：

```bash
cd /Users/memelilihuahua/量化agent构建/trading-assistant
.venv311/bin/python scripts/run_walk_forward.py \
  --start 2020-01-01 \
  --end 2026-03-19 \
  --candidates \
  swing_trend_long_regime_gate_v1@long_scaled1_3_short_fixed1_5,\
  swing_trend_ablation_no_auxiliary_v1@long_scaled1_3_short_fixed1_5,\
  swing_trend_simple_candidate_v2@long_scaled1_3_short_fixed1_5 \
  --output-dir artifacts/backtests/btc_entry_candidates_2020_walk_forward
```

跑 multi-asset 组合 walk-forward：

```bash
cd /Users/memelilihuahua/量化agent构建/trading-assistant
.venv311/bin/python scripts/run_multi_asset_portfolio_walk_forward.py
```

## 12. 交接判断

到 `2026-03-24` 为止，这个项目最重要的现实判断是：

- 当前 BTC 主线在近两年里可用，但拉长到 `2020 -> 2026` 后只能算 **薄 edge**
- 它不像长期 always-on 主线，更像 **regime-dependent 策略**
- 把 BTC 主线直接复制到多标的组合，**没有**把组合做得更好
- 原框架整合尝试里，`confluence gate` 与 `level-aware confirmation` 都没有证明自己是主线升级
- `axis_distance_z / band_position` 当前只能作为 `state_note` 留在解释层
- derivatives state 路线值得保留为研究基础设施，但当前不支持直接做方向 alpha
- carry / basis 当前只应冻结为 `research-only`

### 7.10 Side-Conditional Permission Model v1 的当前状态

这一轮不是把外部模型接进主策略，而是先把 **研究型 side permission gate** 的基础设施搭好。

已经完成的部分：

- 新增研究服务：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/app/services/side_permission_research_service.py`
- 新增研究脚本：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_mainline_side_permission_backtest.py`
- 新增定向测试：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_side_permission_research_service.py`

当前固定的研究契约：

- 输入必须是 **小时级 CSV**
- 最小字段：
  - `timestamp`
  - `permission_label`
  - `allow_long`
  - `allow_short`
  - `model_version`
- 可选字段：
  - `long_score`
  - `short_score`
  - `meta_regime`
- 第一版只接受三态 side permission：
  - `allow_long`
  - `allow_short`
  - `allow_none`
- **不接受** “只有 bull/bear 标签、没有 side permission” 的输入

当前研究脚本已固定 4 个变体：

- `baseline_no_permission`
- `permission_long_only`
- `permission_short_only`
- `permission_full_side_control`

当前 gate 语义也已经固定：

- join 点是 `signal_time`
- join 方式是 backward join：取最近一个 `timestamp <= signal_time` 的 permission row
- missing model coverage 一律 **pass-through**
- 不改 `windowed_mtf.py`
- 不改默认 profile
- 不改 entry / exit / sizing / confirmation
- veto 只发生在 `signal.recommended_timing == NOW` 之后、挂单之前

已完成的验证：

- 定向测试：`5 passed`
- 非 smoke 全量回归：`113 passed`
- 用合成 permission CSV 做过端到端 smoke
  - 产物路径：
    - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/mainline_side_permission_backtest_smoke/two_year/`
  - 说明：这证明 **两年窗口** 的完整 sequence-aware gated backtest 路径已经能从 CSV 输入正常跑到报告落盘

当前还 **没有正式完成** 的部分：

- 还没有用你的 **真实 regime / side permission CSV** 跑正式研究
- 还没有拿到 `full_2020` 的正式 side permission 输出报告
- 还没有对 5 个研究问题做正式结论：
  1. `full_side_control` 是否改善长样本 `PF / expectancy / cum_r / maxDD`
  2. 增量主要来自 `LONG` 还是 `SHORT`
  3. `long_only` / `short_only` 哪一边贡献更大
  4. 连续亏损簇是否明显改善
  5. 被 veto 的交易到底是低质量，还是误伤有效交易

还不能下结论的原因不是实现没做好，而是：

- 当前还缺真实模型 CSV
- 我用合成 CSV 做 full_2020 smoke 时，为避免继续长时间占用资源，中途手动停止了进程；因此只保留了 two-year smoke 产物，不把那次 full_2020 未完成运行误写成正式结果

这条分支的当前工程判断是：

- **研究基础设施已就绪**
- **正式结论还没跑出来**
- 下一轮如果继续，优先动作应该是：
  - 提供真实 side permission CSV
  - 跑 `run_mainline_side_permission_backtest.py`
  - 先看 `full_2020` 的 `summary_all / cohort_summary / side_summary / loss_cluster_summary / vetoed_signals`

所以，下一窗口不要默认沿着旧的 `fixed 2R / swing_trend_v2_btc / 直接扩币 / confluence gate / carry 执行微调` 继续。

更合理的做法是：

- 先读这份 `project_memory.md`
- 再明确选择“部署规则路线 / 新策略原型路线 / 多标的准入路线”三者之一
- 不要把这三条路混在同一轮里同时做

### 7.11 SHORT 非对称 Entry + Post-TP1 Universal Overlay 的当前结论

`2026-03-29` 这一轮已经把 “SHORT 非对称 entry” 和 “post-TP1 universal overlay” 的主线研究链路跑到 promotion gate。

这轮先后得到的正式状态是：

- `SHORT` 非对称 entry 的 winner 是：
  - `swing_trend_long_regime_short_no_reversal_no_aux_v1`
- `post_tp1 extension` 的 universal overlay 是：
  - `be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98`
- 因此进入 promotion gate 的两条 managed candidate 固定为：
  - `champion_managed = swing_trend_long_regime_gate_v1 + be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98`
  - `challenger_managed = swing_trend_long_regime_short_no_reversal_no_aux_v1 + be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98`

本轮新增/确认过的脚本与基础设施：

- 抽出的 managed replay helper：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/post_tp1_managed_replay.py`
- dual-baseline post-TP1 研究脚本：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_post_tp1_extension_matrix.py`
- fixed managed comparison：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_post_tp1_managed_candidate_comparison.py`
- promotion gate：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_mainline_managed_candidate_promotion_gate.py`

对应正式产物：

- dual-baseline overlay 正式结果：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/post_tp1_extension_dual_baseline/`
- managed candidate fixed comparison：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/post_tp1_managed_candidate_comparison/`
- promotion gate 正式结果：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/mainline_managed_candidate_promotion_gate/`

promotion gate 的固定规则是：

- 主窗口：`2020-01-01 -> latest`
- 次级窗口：`2024-03-19 -> latest`
- `base` 口径：默认成本
- `stress_x2` 口径：
  - `taker_fee_bps = 10`
  - `slippage_bps = 4`
- challenger 只有在下面 6 条同时满足时才升格：
  1. `base` 主窗口 `cum_r > champion_managed`
  2. `base` 主窗口 `pf > champion_managed`
  3. `base` 主窗口 `max_dd_r` 不得比 champion 更差超过 `2R`
  4. `base` 次级窗口 `LONG cum_r` 不得比 champion 低超过 `2R`
  5. `stress_x2` 主窗口 `cum_r > champion_managed`
  6. `stress_x2` 主窗口 `pf >= champion_managed`

正式 promotion gate 结果：

- 状态：
  - `challenger_managed_promoted`
- base 主窗口：
  - champion managed：`11.9647R / PF 1.0883 / MaxDD 21.1941R`
  - challenger managed：`30.0499R / PF 1.2186 / MaxDD 13.3870R`
- stress_x2 主窗口：
  - champion managed：`-8.8904R / PF 0.9383 / MaxDD 32.6905R`
  - challenger managed：`8.3865R / PF 1.0574 / MaxDD 15.8578R`
- 次级窗口 `LONG` guardrail：
  - `delta = 0.0R`
- base 主窗口增量来源：
  - `LONG delta = 0.0R`
  - `SHORT delta = 18.0852R`

所以，这一轮之后的正确主线研究判断是：

- `challenger_managed` 已经不是“可疑 challenger”，而是 **后续主线增强研究的默认 baseline**
- 但它仍然带有 `regime-specialist tendency persists`
  - 证据来自季度集中度：`top3 positive quarter share = 96.26%`
- 这意味着：
  - 可以把它当研究 baseline
  - 但还**不能**直接把它写成“已经证明适合无条件生产替换”

当前冻结/不再继续扩的方向：

- side-permission
- 新的 `entry/filter` 组合搜索
- 新的 `post_tp1` overlay 参数扩矩阵
- derivatives 方向 alpha 化

当前固定的下一条路线：

- 先用 `challenger_managed` 作为默认主线研究 baseline
- 下一条 alpha 不再改主线本体，而是转去 **互补 alpha**
- 顺序固定：
  1. `range-failure`
  2. `exhaustion divergence`

这一步要特别避免的误读：

- `managed comparison` 只能说明 overlay 生效后 challenger 仍更强
- `promotion gate` 才是“能不能升格成默认主线研究 baseline”的正式判定
- 当前应引用 promotion gate 结果，而不是继续引用更早那份 fixed comparison 作为最高等级证据

### 7.12 互补 Alpha 复核：Range-Failure 出局，Exhaustion 进入 Watchlist

`2026-03-29` 这一轮已经按新主线 baseline 口径，把两条互补 alpha 候选重新复核过一遍。

这轮固定的 baseline 不再是旧 raw mainline，而是：

- `challenger_managed = swing_trend_long_regime_short_no_reversal_no_aux_v1 + be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98`

新增脚本：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_range_failure_vs_challenger_managed.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_exhaustion_divergence_vs_challenger_managed.py`

新增定向测试：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_range_failure_vs_challenger_managed.py`

这轮固定判定规则：

- 先看 standalone floor：
  - `base full_2020 PF >= 0.80`
  - `base full_2020 cum_r >= -10R`
  - `base full_2020 max_dd_r <= 20R`
  - `stress_x2 full_2020 PF >= 0.75`
  - `stress_x2 full_2020 cum_r >= -15R`
- 只有 floor 过线，才看 complementarity gate：
  - `full_2020 monthly_corr <= 0.20`
  - `full_2020 baseline_negative_alt_positive_months >= 3`
  - `full_2020 offset_r_sum >= 3R`
- 最终只允许三种状态：
  - `rejected_floor`
  - `rejected_offset`
  - `complementary_watchlist`

#### 7.12.1 Range-Failure 复核结论

正式产物：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/range_failure_vs_challenger_managed/`

正式结果：

- 状态：
  - `rejected_floor`
- `base full_2020`：
  - `cum_r = -82.9048R`
  - `PF = 0.3007`
  - `MaxDD = 92.7765R`
- `stress_x2 full_2020`：
  - `cum_r = -111.1022R`
  - `PF = 0.1979`
- 虽然它仍有一定互补性迹象：
  - `monthly_corr = -0.1358`
  - `baseline_negative_alt_positive_months = 5`
  - `offset_r_sum = 5.3603R`
- 但 standalone 已经差到不值得继续，因此直接出局

当前正确解读：

- `range-failure` 不能因为“有 offset 月份”就继续保留
- 这条线的 standalone 负贡献太大，已经超过互补价值
- 后续不再继续深挖 `range-failure`

#### 7.12.2 Exhaustion Divergence 复核结论

正式产物：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/exhaustion_divergence_vs_challenger_managed/`

正式结果：

- 状态：
  - `complementary_watchlist`
- `base full_2020`：
  - `cum_r = 0.5237R`
  - `PF = 1.0827`
  - `MaxDD = 3.0905R`
- `stress_x2 full_2020`：
  - `cum_r = -0.4261R`
  - `PF = 0.9361`
- `full_2020` 互补性：
  - `monthly_corr = -0.0103`
  - `baseline_negative_alt_positive_months = 3`
  - `offset_r_sum = 3.2104R`
  - `opposite_sign_months = 7`

这里有一个很重要的样本提醒：

- smoke 窗口下，`exhaustion divergence` 只得到 `rejected_offset`
- 但正式窗口下，它过了两层门槛，升到了 `complementary_watchlist`

当前正确解读：

- `exhaustion divergence` 不是主线升级
- 但它已经是当前唯一一个通过新 baseline 互补 alpha 复核的候选
- 后续如果继续做“补亏损月份”研究，应该优先围绕它展开，而不是回去继续挖 `range-failure`

#### 7.12.3 当前主线研究路线更新

这一轮之后，默认路线应更新为：

- 主线 baseline 仍是：
  - `challenger_managed`
- 主线内部 tweak 暂停继续扩
- 互补 alpha 当前只有一条 active watchlist：
  - `swing_exhaustion_divergence_v1_btc`
- `range-failure` 已正式出局

所以，下一窗口如果继续做 alpha 研究，优先顺序应是：

1. 围绕 `exhaustion divergence` 做更窄的确认或组合诊断
2. 不要回头重开 `range-failure`
3. 也不要再退回旧 raw mainline 口径去复读老 takeaways

### 7.13 Exhaustion 独立收益晋级门（One-Rule Gate）

`2026-03-29` 这一轮把 `exhaustion` 从“互补逻辑”切到“独立收益候选”评估，并固定为单规则候选池：

- `swing_exhaustion_divergence_v1_btc`（baseline）
- `swing_exhaustion_divergence_short_only_v1_btc`
- `swing_exhaustion_divergence_min_level3_v1_btc`
- `swing_exhaustion_divergence_ct_block80_v1_btc`

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_exhaustion_divergence_min_level3_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_exhaustion_divergence_ct_block80_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_exhaustion_standalone_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_exhaustion_standalone_one_rule_gate.py`

正式产物目录：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/exhaustion_standalone_one_rule_gate/`

正式结论（`full_2020`）：

- `swing_exhaustion_divergence_ct_block80_v1_btc` 通过晋级门，状态：
  - `promoted_standalone_candidate`
- 关键指标：
  - `base geometric_return_pct = 13.2661%`（baseline `3.5926%`）
  - `base PF = 2.1428`
  - `base max_dd_r = 1.6954`
  - `stress geometric_return_pct = 11.7524%`
  - `stress PF = 1.8362`
  - `top3_trades_pnl_share_pct = 63.4941%`
  - `best_year_geometric_pct_share = 32.1511%`

未通过的候选：

- `short_only`：
  - 收益提升明显，但 `trades=8` 且 `top3_share=84.9541%`，集中度与样本数不达标
- `min_level3`：
  - `top3_share=66.9331%` 超过 `65%` 上限

路线更新：

- exhaustion 家族暂不冻结，保留并晋级：
  - `swing_exhaustion_divergence_ct_block80_v1_btc`
- 下一步应按既定口径与主线并列评估，且不做收益直接相加

### 7.14 并列评估：`ct_block80` vs `challenger_managed`（不做收益相加）

`2026-03-29` 已把 `swing_exhaustion_divergence_ct_block80_v1_btc` 放入独立候选池，并按同口径与主线 baseline 做并列对照。

新增脚本：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_exhaustion_candidate_vs_challenger_managed.py`

新增测试：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_exhaustion_candidate_vs_challenger_managed.py`

正式产物目录：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/exhaustion_candidate_vs_challenger_managed/`

正式结论：

- `pool_status`：
  - `independent_pool_member`
- `status`：
  - `mainline_still_preferred_candidate_kept_in_pool`

关键对照（`base/full_2020`）：

- `challenger_managed`：
  - `geometric_return_pct = 53.1026%`
  - `PF = 1.2186`
  - `cum_r = 30.0499R`
  - `max_dd_r = 13.3870R`
- `ct_block80`：
  - `geometric_return_pct = 13.2661%`
  - `PF = 2.1428`
  - `cum_r = 3.6580R`
  - `max_dd_r = 1.6954R`

补充观察：

- `ct_block80` 的风险控制和稳定性很好，但收益规模显著低于主线
- `stress_x2` 下 `ct_block80` 仍为正（`11.7524%`, `PF 1.8362`）
- `two_year` 的 `LONG` 侧几乎没有贡献，导致 `LONG guard` 相对主线不通过（`delta = -13.5129R`）

路线更新：

- 当前不替换主线，主线仍是 `challenger_managed`
- `ct_block80` 保留在独立候选池，后续按“并列评估”继续跟踪
- 继续保持“不做收益直接相加”的治理边界

### 7.15 Breakout One-Rule Gate（单规则晋级门）+ 并列评估分叉

`2026-03-30` 已按固定口径完成 breakout 单规则晋级门，实现了对称候选池、成本双场景、几何收益/CAGR/年度几何收益输出。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_breakout_setup_proximity_045_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_breakout_trigger_buffer_004_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_breakout_base_width_45_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_breakout_standalone_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_breakout_candidate_vs_challenger_managed.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_breakout_standalone_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_breakout_candidate_vs_challenger_managed.py`

正式产物目录：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/breakout_standalone_one_rule_gate/`

正式结论（`promotion_decision.csv`）：

- 三个 breakout 单规则候选全部为：
  - `rejected_fragile_or_unprofitable`
- 共同失败点：
  - `base PF < 1.10`
  - `base max_dd_r > 6.0`
  - `stress_x2 geometric_return_pct <= 0`
  - `stress_x2 PF < 1.00`
- 结果解读：
  - breakout 家族在当前主线口径下未通过晋级门，不进入并列挑战环节

路线分叉执行：

- 按预定 fallback，不继续深挖 breakout 参数
- 直接切到 `Neutral-Range` one-rule gate

### 7.16 Fallback 执行：Neutral-Range One-Rule Gate（同门槛）

由于 breakout 无晋级者，已执行 fallback，并新增 neutral-range 单规则候选池（同等强度门槛，不放宽）。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_neutral_range_reversion_edge_030_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_neutral_range_reversion_sweep_008_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_neutral_range_reversion_opp_r_100_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_neutral_range_standalone_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_neutral_range_standalone_one_rule_gate.py`

正式产物目录：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/neutral_range_standalone_one_rule_gate/`

正式结论（`promotion_decision.csv`）：

- 三个 neutral-range 单规则候选全部为：
  - `rejected_fragile_or_unprofitable`
- 关键失败原因：
  - `PF` 显著低于 1（多数在 `0.16 ~ 0.44`）
  - `stress_x2` 下几何收益继续为负
  - 部分候选 `trades < 10` 或 `top3_share / best_year_share` 超限

当前状态更新：

- 主线 baseline 仍为：
  - `challenger_managed`
- 独立候选池仍保留：
  - `swing_exhaustion_divergence_ct_block80_v1_btc`
- breakout 与 neutral-range 两个家族在本轮 one-rule gate 都未证明可晋级

### 7.17 固定日历 Regime-Switch 复核：`simple_candidate_v2 -> challenger_managed`

`2026-03-30` 已按固定切换日 `2024-03-19` 完成 `simple_candidate_v2 -> challenger_managed` 的管理层切换复核，新增专用脚本：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_simple_candidate_v2_regime_switch_fixed_calendar.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_simple_candidate_v2_regime_switch_fixed_calendar.py`

正式产物目录：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/simple_candidate_v2_regime_switch_fixed_calendar/`

正式结论（`comparison_decision.csv`）：

- 状态：
  - `promoted_calendar_switch_candidate`
- 主窗口 `full_2020`：
  - baseline `challenger_managed`：`53.9274%` 几何收益，`CAGR 7.1538%`，`PF 1.2196`
  - switch `simple_candidate_v2 -> challenger_managed`：`157.8458%` 几何收益，`CAGR 16.3855%`，`PF 1.3445`
- 次级窗口 `two_year`：
  - switch 的 `LONG` 侧 `cum_r` 与 baseline 一致，说明后段被 `challenger_managed` 稳住
- 风险口径：
  - `base` 与 `stress_x2` 下都优于 baseline 的几何收益
  - 但这条线仍然是管理层切换候选，不代表可以扩展成新的搜索自由度

补充说明：

- smoke 原计划窗口与固定切换日有冲突，所以 smoke 改为一个能同时覆盖切换前后两段的短窗，只用于验证脚本与产物结构
- 这条线当前只接受固定切换，不再搜索切换点

### 7.17.1 固定日历 Regime-Switch 稳健性复核

`2026-03-30` 已对 `simple_candidate_v2 -> challenger_managed` 做更严格的 probe 面板稳健性复核，新增脚本：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_simple_candidate_v2_regime_switch_robustness.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_simple_candidate_v2_regime_switch_robustness.py`

正式产物目录：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/simple_candidate_v2_regime_switch_robustness/`

结论：

- `robustness_decision.csv` 状态为：
  - `robust_under_probe_panel`
- 5 个固定 probe 切点全部通过：
  - `2024-01-19`
  - `2024-02-18`
  - `2024-03-19`
  - `2024-04-18`
  - `2024-05-18`
- 所有 probe 都同时满足：
  - `base` 几何收益高于 `challenger_managed`
  - `stress_x2` 几何收益高于 `challenger_managed`
  - `stress_x3` 仍为正，且仍高于 `challenger_managed`
- 收益集中度诊断：
  - `top3_trade_pnl_share_pct = 13.00%`
  - `best_year_pnl_share_pct = 29.43%`
  - `best_month_pnl_share_pct = 17.17%`
  - 没有出现明显的“少数交易/少数月份”驱动型过拟合迹象

补充判断：

- 这条切换线不像单点巧合，更像一个在固定日历下可复现的管理层规则
- 但它仍然是 regime switch，不是新的自由度入口；后续如果再扩，只能做更正交的 alpha 家族，不再继续搜切点

### 7.17.2 固定切换版升格候选 baseline：`switch` vs `ct_block80` 一次性并列对照

`2026-03-30` 已执行一次性 head-to-head，并保持“并列评估、不做收益直接相加”的治理边界。

新增脚本：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_switch_candidate_vs_ct_block80.py`

新增测试：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_switch_candidate_vs_ct_block80.py`

正式产物目录：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_candidate_vs_ct_block80/`

正式结论（`comparison_decision.csv`）：

- 状态：
  - `promoted_new_baseline_candidate`
- 对照对象：
  - `switch_candidate = swing_trend_simple_candidate_v2 -> swing_trend_long_regime_short_no_reversal_no_aux_v1 + be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98`
  - `ct_block80 = swing_exhaustion_divergence_ct_block80_v1_btc`

关键结果（`base/full_2020`）：

- `switch_candidate`：
  - `geometric_return_pct = 157.8458%`
  - `CAGR = 16.3855%`
  - `PF = 1.3445`
  - `cum_r = 45.6437R`
  - `max_dd_r = 12.1964R`
  - `trades = 286`
- `ct_block80`：
  - `geometric_return_pct = 13.2661%`
  - `CAGR = 2.0156%`
  - `PF = 2.1428`
  - `cum_r = 3.6580R`
  - `max_dd_r = 1.6954R`
  - `trades = 10`

压力成本与 guardrail：

- `stress_x2`：
  - `switch = 83.4452%`，`ct_block80 = 11.7524%`
- `stress_x3`：
  - `switch = 33.5957%`，`ct_block80 = 10.2566%`
- `two_year LONG delta`（switch - ct）：
  - `+13.5129R`

集中度诊断（switch）：

- `top3_trade_pnl_share_pct = 13.00%`
- `best_year_pnl_share_pct = 29.43%`
- `best_month_pnl_share_pct = 17.17%`
- 本轮配置下未见“少数交易/少数月份”撑起收益的集中风险

路线更新：

- 后续主线研究 baseline 升格为：
  - `switch_simple_candidate_v2_then_challenger_managed`
- `ct_block80` 继续保留在独立候选池做并列跟踪
- 不继续新增 regime-switch 切点或规则，维持固定日历规则不扩自由度

### 7.18 正交家族 One-Rule Gate：Range-Failure（固定切换基线之后）

`2026-03-30` 已按“先 one-rule gate，后并列评估”的固定流程执行 `range-failure` 正交家族复核。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_range_failure_edge_035_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_range_failure_sweep_008_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_range_failure_max_width_45_v1_btc.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_range_failure_standalone_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_range_failure_standalone_one_rule_gate.py`

同时更新：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/services/strategy_service.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/utils/timeframes.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_strategy.py`

正式产物目录：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/range_failure_standalone_one_rule_gate/`

正式结论（`promotion_decision.csv`）：

- 三个候选全部：
  - `rejected_fragile_or_unprofitable`
- 候选列表：
  - `swing_range_failure_edge_035_v1_btc`
  - `swing_range_failure_sweep_008_v1_btc`
  - `swing_range_failure_max_width_45_v1_btc`

关键结果（`base/full_2020`）：

- baseline `swing_range_failure_v1_btc`：
  - `geometric_return_pct = -46.5705%`
  - `CAGR = -9.5536%`
  - `PF = 0.3007`
  - `max_dd_r = 92.7765R`
- 最好候选 `edge_035`：
  - `geometric_return_pct = -26.4954%`
  - `CAGR = -4.8116%`
  - `PF = 0.3258`
  - `max_dd_r = 58.8488R`

拒绝原因（共同）：

- `base PF` 明显低于 `1.10`
- `base max_dd_r` 远高于 `6R`
- `stress_x2` 几何收益为负、`PF < 1.00`
- `best_year_geometric_pct_share = 100%`（收益分布质量不过门）

路线更新：

- `range-failure` 这一轮 one-rule 分支冻结，不进入与新主线 baseline 的并列挑战
- 当前可用研究基线仍为：
  - `switch_simple_candidate_v2_then_challenger_managed`
- 独立候选池维持：
  - `swing_exhaustion_divergence_ct_block80_v1_btc`（并列跟踪，不做收益相加）

### 7.19 固定 `switch` 基线：风险预算 One-Rule 晋级门（中等门槛 M1）

`2026-03-30` 已完成“固定 `switch` 基线 + 风险预算 one-rule”正式 gate。该轮不新增 signal profile、不改切换日、仅评估管理层仓位预算候选。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_switch_risk_budget_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_switch_risk_budget_one_rule_gate.py`

运行与验证：

- 单测：
  - `tests/test_switch_risk_budget_one_rule_gate.py`：`16 passed`
- 回归：
  - `tests/test_strategy.py` + `tests/test_backtesting.py`：`77 passed`
- smoke 产物：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_risk_budget_one_rule_gate_smoke/`
- 正式产物：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_risk_budget_one_rule_gate/`

候选池（固定 4 条）：

- `flat_1_0`（baseline）
- `long_1.05_short_0.93`
- `long_1.10_short_0.86`
- `long_1.15_short_0.79`

正式结论（`promotion_decision.csv`）：

- 状态：
  - `promoted_management_overlay_candidate`
- 入选候选：
  - `long_1.15_short_0.79`

关键指标（`base/full_2020`）：

- `flat_1_0`：
  - `geometric_return_pct = 157.8458%`
  - `CAGR = 16.3855%`
  - `PF = 1.3445`
  - `max_dd_r = 12.1964R`
- `long_1.15_short_0.79`：
  - `geometric_return_pct = 191.1993%`
  - `CAGR = 18.6758%`
  - `PF = 1.3561`
  - `max_dd_r = 11.6214R`
  - `avg_size = 1.0015`（满足预算近中性 `[0.98, 1.02]`）

压力成本与稳健性：

- `stress_x2`：`geometric_return_pct = 107.0028%`，`PF = 1.1832`
- `stress_x3`：`geometric_return_pct = 51.6976%`，`PF = 1.0515`
- `two_year LONG delta_r = +2.0269R`（未触发 LONG guard 退化）
- walk-forward OOS（`365/90/90`）：
  - `candidate_oos_geometric_return_pct = 74.6421%`
  - `delta_vs_flat = +10.7422%`

集中度诊断（`base/full_2020`）：

- `top3_trades_pnl_share_pct = 3.8731%`（<= `65%`）
- `best_year_geometric_pct_share = 42.8802%`（<= `80%`）

补充观察（诊断项）：

- OOS 聚合口径下，候选 `geometric` 高于 flat，但 `cum_r` 略低（`31.9285R` vs `32.4865R`）。当前 gate 以几何收益/CAGR 为主指标，这个差异已保留在产物供后续并列评估使用。

路线更新：

- 风险预算 one-rule 按“一轮后收口”执行完成，不开第二轮参数扩展。
- `long_1.15_short_0.79` 升格为新的“管理层候选 baseline”。
- 后续继续只做“正交家族 one-rule gate”并列评估，不回到 regime-switch/entry/filter 扩展。

### 7.19.1 BTC 基准对照（年度与全窗口）

`2026-03-30` 按当前正式产物对新候选 baseline 做了 BTC 年度基准复核，比较对象为：

- 策略：`switch + risk-budget(long_1.15_short_0.79)`（`base/full_2020`）
- 基准：`BTC/USDT:USDT` 永续 1d 买入持有（与当前回测市场类型一致）

产物：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_risk_budget_one_rule_gate/btc_benchmark_yearly_comparison.csv`

年度结果（策略是否跑赢 BTC）：

- 2020：未跑赢
- 2021：未跑赢
- 2022：跑赢
- 2023：未跑赢
- 2024：未跑赢
- 2025：未跑赢
- 2026（YTD）：跑赢

汇总：

- 年度胜率：`2 / 7 = 28.57%`
- 全窗口（`2020-01-01 -> 2026-03-30`）：
  - 策略：`geometric_return_pct = 191.1993%`，`CAGR = 18.6758%`
  - BTC 基准：`buy&hold return = 834.6752%`，`CAGR = 43.0531%`

结论：

- 该候选在“绝对收益提升（相对旧 flat）”上成立，但在“每年跑赢 BTC”目标上不成立。
- 若后续目标明确要求“年年跑赢 BTC”，需要切换研究目标与门槛（当前主线更偏风险控制 + 管理层稳健，而非高 beta 追涨）。

### 7.20 固定 `switch` 基线：Axis/Band 管理层 One-Rule Gate（M1 + BTC 诊断不退步）

`2026-03-30` 已按固定方案完成 Axis/Band 管理层 one-rule gate。本轮不新增 signal profile，不改 entry/filter，不扩 regime-switch，只在固定切换基线上做单规则管理层加权。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_switch_axis_band_management_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_switch_axis_band_management_one_rule_gate.py`

运行与验证：

- 新增单测：
  - `tests/test_switch_axis_band_management_one_rule_gate.py`：`18 passed`
- 回归：
  - `tests/test_strategy.py` + `tests/test_backtesting.py`：`77 passed`
- smoke 产物：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_axis_band_management_one_rule_gate_smoke/`
- 正式产物：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_axis_band_management_one_rule_gate/`

候选池（固定 4 条）：

- `flat_control`（当前基线控制组）
- `long_active_k105`
- `long_active_k110`
- `long_active_k115`

正式结论（`promotion_decision.csv`）：

- 状态：
  - `promoted_management_overlay_candidate`
- 入选候选：
  - `long_active_k115`
- 满意结果（M1 + BTC 不退步）：
  - `satisfied_result = True`

关键指标（`base/full_2020`）：

- `flat_control`：
  - `geometric_return_pct = 190.8090%`
  - `CAGR = 18.6503%`
  - `PF = 1.3561`
  - `max_dd_r = 11.6044R`
- `long_active_k115`：
  - `geometric_return_pct = 210.9473%`
  - `CAGR = 19.9298%`
  - `PF = 1.3675`
  - `max_dd_r = 11.4122R`
  - `avg_size = 1.0000`（预算中性约束满足）

压力成本与稳健性：

- `stress_x2`：`geometric_return_pct = 125.1650%`，`PF = 1.2027`
- `stress_x3`：`geometric_return_pct = 64.5709%`，`PF = 1.0679`
- `two_year LONG delta_r = +0.5065R`（LONG guard 保持）
- walk-forward OOS（`365/90/90`）：
  - `candidate_oos_geometric_return_pct = 82.4769%`
  - `delta_vs_flat = +7.9492%`

集中度诊断（`base/full_2020`）：

- `top3_trades_pnl_share_pct = 4.2388%`
- `best_year_geometric_pct_share = 43.2709%`

BTC 诊断（非硬拒绝）：

- 文件：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_axis_band_management_one_rule_gate/btc_yearly_comparison.csv`
- `flat_control` 与 `long_active_k115` 都是 `2/7` 年跑赢 BTC（`28.57%`）
- 因为候选 `>=` 基线，所以满足“BTC 年度不退步”诊断条件

路线更新：

- Axis/Band one-rule 这一轮已完成并收口，不做第二轮扩参。
- 在当前治理口径下，`long_active_k115` 可作为下一步管理层候选 baseline。
- 后续继续优先正交家族 one-rule gate，并保持“不把 BTC 年度胜率当硬卡”的纪律，避免目标函数诱导过拟合。

### 7.21 固定 `switch` 基线：Trend-Strength 管理层 One-Rule Gate（M1 + CAGR>=22 + BTC 不退步）

`2026-03-30` 已完成固定 `switch` 基线下的 Trend-Strength one-rule gate。该轮保持低自由度：不新增 signal profile、不改 entry/filter、不扩 regime-switch，只做管理层单规则。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_switch_trend_strength_management_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_switch_trend_strength_management_one_rule_gate.py`

实现口径（固定）：

- 基线：`long_1.15_short_0.79 + long_active_k115`（在脚本内作为 `flat_control` 固定层）
- one-rule：仅对 `LONG & trend_strength >= 95` 施加额外倍率 `k`
- 候选池（固定 4 条）：
  - `flat_control`
  - `long_trend95_k110`
  - `long_trend95_k115`
  - `long_trend95_k120`
- 预算中性归一：`base/full_2020 avg_size` 归一到 `1.0`，并要求 `[0.98,1.02]`
- 成本场景：`base / stress_x2 / stress_x3`

运行与验证：

- 新增单测：
  - `tests/test_switch_trend_strength_management_one_rule_gate.py`：`19 passed`
- 回归：
  - `tests/test_strategy.py` + `tests/test_backtesting.py`：`77 passed`
- smoke 产物：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_trend_strength_management_one_rule_gate_smoke/`
- 正式产物：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_trend_strength_management_one_rule_gate/`

正式结论（`promotion_decision.csv`）：

- 入选候选：
  - `long_trend95_k120`
- 状态：
  - `promoted_management_overlay_candidate`
- 满意结果（`CAGR>=22` + BTC 不退步）：
  - `satisfied_result = False`
  - 失败主因：`base_full_2020_candidate_cagr_pct = 21.0118%`，未达到 `22.0%`

关键指标（`base/full_2020`）：

- `flat_control`：
  - `geometric_return_pct = 210.9473%`
  - `CAGR = 19.9298%`
  - `PF = 1.3675`
  - `max_dd_r = 11.4122R`
- `long_trend95_k120`：
  - `geometric_return_pct = 228.8781%`
  - `CAGR = 21.0118%`
  - `PF = 1.3811`
  - `max_dd_r = 11.3002R`
  - `avg_size = 1.0000`（预算中性约束通过）

稳健性与诊断：

- M1 全部通过（`geo/cagr/pf/dd/stress/LONG guard/OOS/集中度` 均通过）
- `stress_x2`：`geometric_return_pct = 138.0757%`，`PF = 1.2150`
- `stress_x3`：`geometric_return_pct = 74.6439%`，`PF = 1.0801`
- `two_year LONG delta_r = +1.7173R`
- OOS（`365/90/90`）：
  - `candidate_oos_geometric_return_pct = 88.3202%`
  - `delta_vs_flat = +5.8433%`
- BTC 年度诊断：
  - `candidate_btc_outperform_years = 2`
  - `flat_btc_outperform_years = 2`
  - 满足“不退步”

路线更新：

- 本轮按纪律收口：不做第二轮 trend-strength 扩参。
- 由于未达 `CAGR>=22` 满意条件，当前结论是“可提升但未达到本轮收益目标”，应冻结该分支并切到下一条正交家族 one-rule gate。

### 7.22 固定 `switch` 基线：Post-TP1 路径持有 One-Rule Gate（M1 + CAGR>=22 + BTC 不退步）

`2026-03-30` 已完成固定 `switch` 基线下的 Post-TP1 路径持有 one-rule gate。该轮保持低自由度：不新增 signal profile、不改 entry/filter、不扩 regime-switch，只做路径管理单规则。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_switch_post_tp1_path_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_switch_post_tp1_path_one_rule_gate.py`

实现口径（固定）：

- 基线交易流：`switch_simple_candidate_v2_then_challenger_managed`
- 固定管理层 sizing：`long_1.15_short_0.79 + long_active_k115`
- 路径候选池（固定 4 条）：
  - `flat_control`
  - `path_hold_2bars_4h_bullish`
  - `path_hold_3bars_4h_bullish`
  - `path_hold_3bars_4h_bullish_then_be_on_4h_loss`
- 成本场景：`base / stress_x2 / stress_x3`
- 满意条件：`M1 全通过 + CAGR>=22 + BTC 年度不退步`

运行与验证：

- 新增单测：
  - `tests/test_switch_post_tp1_path_one_rule_gate.py`：`19 passed`
- 回归：
  - `tests/test_strategy.py` + `tests/test_backtesting.py`：`77 passed`
- smoke 产物：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_post_tp1_path_one_rule_gate_smoke/`
- 正式产物：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/switch_post_tp1_path_one_rule_gate/`

正式结论（`promotion_decision.csv`）：

- 入选候选：
  - `path_hold_3bars_4h_bullish`
- 状态：
  - `rejected_management_overlay`
- 满意结果：
  - `satisfied_result = False`

关键指标（`base/full_2020`）：

- `flat_control`：
  - `geometric_return_pct = 210.9473%`
  - `CAGR = 19.9298%`
  - `PF = 1.3675`
  - `max_dd_r = 11.4122R`
- `path_hold_3bars_4h_bullish`：
  - `geometric_return_pct = 330.9891%`
  - `CAGR = 26.3688%`
  - `PF = 1.4418`
  - `max_dd_r = 9.8140R`
  - `avg_size = 1.0000`

拒绝原因（硬门槛）：

- `two_year LONG guard` 未通过：
  - `two_year_long_delta_r = -5.3098R`（低于 `-2R` 守门）
- 其余约束（base/stress/OOS/集中度/CAGR22/BTC 不退步）均通过。

路线更新：

- 该分支按纪律收口：路径持有 one-rule 本轮已完成且被拒绝，不做第二轮扩参。
- 主线内管理层微调继续冻结，下一步应转向新的正交家族 one-rule gate。

### 7.23 转正交家族 one-rule：Trend-Divergence Standalone Gate（单轮收口）

`2026-03-30` 已执行“冻结 path 微调后”的正交家族 one-rule gate，本轮选用 `trend_divergence` 家族，固定候选池、固定门槛、单轮判定，不做二次扩参。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_divergence_min_level3_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_divergence_no_reversal_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_trend_divergence_standalone_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_trend_divergence_standalone_one_rule_gate.py`

接入改动：

- `StrategyService` 注册新增 profile：
  - `swing_trend_divergence_min_level3_v1`
  - `swing_trend_divergence_no_reversal_v1`
- `app/utils/timeframes.py` 同步新增上述 profile 的支持列表与 `1d/4h/1h` 映射。
- `tests/test_strategy.py` 新增：
  - 新 profile 可用性测试
  - divergence one-rule 单键改动一致性测试

候选池（固定）：

- baseline：`swing_trend_divergence_v1`
- candidates：
  - `swing_trend_divergence_no_reversal_v1`
  - `swing_trend_divergence_min_level3_v1`
  - `swing_trend_long_divergence_gate_v1`

验证结果：

- 单测：
  - `tests/test_trend_divergence_standalone_one_rule_gate.py`：`3 passed`
  - `tests/test_strategy.py + tests/test_backtesting.py`：`80 passed`
- smoke：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/trend_divergence_standalone_one_rule_gate_smoke/`
- formal：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/trend_divergence_standalone_one_rule_gate/`

formal 结论（`promotion_decision.csv`）：

- 三个 candidate 全部：
  - `rejected_fragile_or_unprofitable`
- 最好表现是 `swing_trend_divergence_no_reversal_v1`：
  - `base/full_2020 geometric_return_pct = +21.8659%`
  - 但 `PF = 1.067 < 1.10`，`max_dd_r = 17.6777 > 6.0`，
  - 且 `stress_x2 geometric_return_pct = -20.7449%`, `PF = 0.9453`，
  - 未通过独立收益 gate。

路线更新：

- path 微调分支继续冻结；
- trend-divergence 正交 one-rule 分支本轮也冻结（单轮未晋级）；
- 下一步应继续正交家族 one-rule，但保持单轮、固定候选池纪律，避免“试到满意”为目标函数。

### 7.24 继续正交家族 one-rule：Trend-Confluence Standalone Gate（单轮收口）

`2026-03-30` 已按相同纪律执行下一正交家族 `trend_confluence` 的 one-rule gate：固定候选池、固定门槛、单轮判定，不回主线内部微调。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_confluence_min_hits_3_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_confluence_max_spread_10_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_trend_confluence_standalone_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_trend_confluence_standalone_one_rule_gate.py`

接入与测试更新：

- `StrategyService` 注册新增 profile：
  - `swing_trend_confluence_min_hits_3_v1`
  - `swing_trend_confluence_max_spread_10_v1`
- `app/utils/timeframes.py` 同步新增支持与 `1d/4h/1h` 映射。
- `tests/test_strategy.py` 新增：
  - confluence one-rule profile 可用性测试
  - confluence one-rule 单键改动一致性测试

候选池（固定）：

- baseline：`swing_trend_confluence_setup_v1`
- candidates：
  - `swing_trend_confluence_min_hits_3_v1`
  - `swing_trend_confluence_max_spread_10_v1`
  - `swing_trend_confluence_structure_gate_hard_v1`

验证结果：

- 新单测：
  - `tests/test_trend_confluence_standalone_one_rule_gate.py`：`3 passed`
- 回归：
  - `tests/test_strategy.py + tests/test_backtesting.py`：`82 passed`
- smoke：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/trend_confluence_standalone_one_rule_gate_smoke/`
- formal：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/trend_confluence_standalone_one_rule_gate/`

formal 结论（`promotion_decision.csv`）：

- 三个 candidate 全部：
  - `rejected_fragile_or_unprofitable`
- 主要失败原因：
  - baseline/candidates 在 `base full_2020` 几何收益整体为负或接近零；
  - `stress_x2` 下均未达到生存门槛（几何收益为负，PF < 1）；
  - `structure_gate_hard` 交易数仅 `3`，未过 trades floor。

路线更新：

- `trend_confluence` 本轮 one-rule 分支冻结，不做二轮扩参。
- 继续保持研究纪律：下一步仍应选新的正交家族做单轮 gate，而不是回主线内部继续加条件。

### 7.25 继续正交家族 one-rule：Level-Aware Confirmation Standalone Gate（单轮收口）

`2026-03-30` 已继续按同样纪律执行下一正交家族 `level-aware confirmation` 的 one-rule gate：固定候选池、固定门槛、单轮判定，不回主线内部微调。

新增代码：

- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_level_aware_confirmation_min_hits_2_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_level_aware_confirmation_ema55_025_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/app/strategies/swing_trend_level_aware_confirmation_band_touch_035_v1.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/scripts/run_level_aware_confirmation_standalone_one_rule_gate.py`
- `/Users/memelilihuahua/量化agent构建/trading-assistant/tests/test_level_aware_confirmation_standalone_one_rule_gate.py`

接入与测试更新：

- `StrategyService` 注册新增 3 个 level-aware one-rule profile。
- `app/utils/timeframes.py` 同步新增支持与 `1d/4h/1h` 映射。
- `tests/test_strategy.py` 新增：
  - level-aware one-rule profile 可用性测试
  - level-aware one-rule 单键改动一致性测试

候选池（固定）：

- baseline：`swing_trend_level_aware_confirmation_v1`
- candidates：
  - `swing_trend_level_aware_confirmation_min_hits_2_v1`
  - `swing_trend_level_aware_confirmation_ema55_025_v1`
  - `swing_trend_level_aware_confirmation_band_touch_035_v1`

验证结果：

- 新单测：
  - `tests/test_level_aware_confirmation_standalone_one_rule_gate.py`：`3 passed`
- 回归：
  - `tests/test_strategy.py + tests/test_backtesting.py`：`84 passed`
- smoke：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/level_aware_confirmation_standalone_one_rule_gate_smoke/`
- formal：
  - `/Users/memelilihuahua/量化agent构建/trading-assistant/artifacts/backtests/level_aware_confirmation_standalone_one_rule_gate/`

formal 结论（`promotion_decision.csv`）：

- 三个 candidate 全部：
  - `rejected_fragile_or_unprofitable`
- 最好候选是 `min_hits_2`，但仍失败：
  - `base/full_2020 geometric_return_pct = -3.4909%`
  - `PF = 0.8222`
  - `stress_x2 geometric_return_pct = -10.8168%`
  - 未满足独立收益 gate 的盈利与压力生存门槛。

路线更新：

- `level-aware confirmation` 本轮 one-rule 分支冻结，不做二轮扩参。
- 继续保持纪律：下一步仍应选择新的正交家族做单轮 gate，而不是回到主线内部微调。
