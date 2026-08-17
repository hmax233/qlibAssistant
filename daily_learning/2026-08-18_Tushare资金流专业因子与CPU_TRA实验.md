# 2026-08-18 Tushare资金流、筹码与CPU TRA实验

## 1. 最终结论

本轮使用升级后的 Tushare 权限补充了 `daily_basic`、`moneyflow`、`cyq_perf`、
`stk_factor_pro`，并在同一批 300 只 CSI1000 主板历史成分股上完成 GBDT 因子消融、
CPU TRA 三随机种子实验和严格成交回测。

结论分为两层：

1. **研究指标有实质改善。** 日频资金流/筹码/专业因子是 A/B 开发折里最稳定的
   GBDT 特征组；CPU TRA 的三 seed 集成在冻结 Fold C test 上取得
   `Rank IC=0.0443`、`Rank ICIR=0.2511`，明显高于本轮日频 XGBoost。
2. **可交易结果仍不合格。** 在与旧 XGB240、Fixed Ensemble 共同的 99 个交易日、
   共同 300 只股票和同一原始收益标签上，TRA 严格 Top1/3/10 净累计分别为
   `-29.45%/-12.35%/-15.44%`。TRA 比旧基线少亏，但绝对收益仍为负，当前不得替换
   每日实盘流程。

这说明提高全截面排序质量并不等价于提高顶部少量股票的一日可成交收益。下一轮的
重点不应继续盲目扩大 TRA，而应转向绝对收益校准、可买概率和低换手持有策略。

## 2. 新增数据

数据范围为 2019-01-01 至 2026-08-13，股票样本与分钟实验一致：CSI1000 主板历史
成分的稳定哈希样本 300 只，不含科创板和创业板。

| 接口 | 行数 | 主要用途 |
|---|---:|---|
| `daily_basic` | 517,609 | 换手率、量比、市值、估值、流通盘结构 |
| `moneyflow` | 527,119 | 小/中/大/特大单成交与净流入结构 |
| `cyq_perf` | 506,681 | 获利盘比例、成本分位、筹码宽度和集中度 |
| `stk_factor_pro` | 527,348 | DMI、VR、BRAR、MFI、MASS、ATR 等专业技术字段 |

1,200 个“股票×接口”任务全部成功，总下载时间 682.55 秒。token 仍只从本机配置读取，
没有写入报告、日志或元数据。原始数据目录：

```text
.qlibAssistant/supplemental/tushare_daily/
```

## 3. 因子构建与防泄漏

`script/build_tushare_daily_factors.py` 生成 234 个同日可知的日频特征，共 527,348 个
股票日。因子包含原始变换、5/20 日后向滚动统计和同日横截面排名。

主要因子族：

- 估值、自由流通换手、市值和流通盘比例；
- 大/特大/中/小单成交量及成交额不平衡；
- 获利盘比例、加权成本偏离、70%/90%筹码成本带宽；
- DMI/ADX/ADXR、VR、BRAR、MFI、MASS、ATR相对价格；
- 上述变量的 5/20 日均值、波动率及同日截面排名。

核心字段缺失率总体较低。股息率约 28.57%、正盈利收益率约 18.49% 缺失，保留为
缺失信息，不用未来日期回填。因子文件：

```text
.qlibAssistant/supplemental/tushare_daily_factors/all_factors.parquet
```

## 4. 固定时间切分

开发阶段只能看 A/B；特征组、模型结构和 TRA 超参数冻结后，才运行 C：

| Fold | Train | Valid（早停） | Selection-valid（选配置） | Test |
|---|---|---|---|---|
| A | 2019-01-01~2022-12-31 | 2023H1 | 2023H2 | 2024H1 |
| B | 2019-01-01~2023-12-31 | 2024H1 | 2024H2 | 2025全年 |
| C | 2019-01-01~2024-12-31 | 2025H1 | 2025H2 | 2026-01-01~07-17 |

信号为 T 日收盘后，计划 T+1 收盘买入、T+2 收盘卖出。严格回测模拟涨跌停无法成交、
停牌/缺价、无法买入留现金、无法卖出继续持仓和换手成本。`event_guard` 使用完整
T+1 日线，只能当理想化诊断，不能作为主结果。

## 5. GBDT因子消融

为了公平，所有 Alpha158、分钟、日频补充因子组合都限制在相同 300 只股票和相同
股票日交集上。A/B selection 结果按 Rank ICIR 排名：

| 模型与特征 | 平均Rank IC | 平均Rank ICIR | 最差折Rank ICIR |
|---|---:|---:|---:|
| XGBoost + daily | 0.0259 | 0.2578 | 0.2393 |
| LightGBM + Alpha158+分钟 | 0.0233 | 0.2152 | — |
| LightGBM + 全部 | 0.0218 | 0.2138 | — |
| XGBoost + 全部 | 0.0217 | 0.2136 | — |
| XGBoost + 分钟+daily | 0.0214 | 0.2084 | — |
| XGBoost + Alpha158 | 0.0209 | 0.1776 | — |

`daily` 单独使用优于简单拼接，说明补充因子提供了有效信息，但 Alpha158/分钟与它
直接拼接会增加噪声和共线性。重要因子集中在自由流通换手、量比波动、获利盘比例、
筹码成本宽度、ADX/ADXR、资金流不平衡和 ATR 相对价格。

历史 C 对比中，daily XGBoost 的 test Rank ICIR 为 0.0482，高于 Alpha158 的
0.0087，但严格收益仍为负；2026-07-18~08-11 的 16 日短影子样本中优势也没有保持。
因此不能仅凭开发折指标上线。

## 6. CPU TRA配置和结果

TRA 使用本项目安装的 Qlib 官方 `TRAModel` 实现，设备明确设为 CPU：

- 输入：从 GBDT A/B 重要性中选出的 40 个稳定 daily 因子；
- 序列长度：60 个交易日；
- 主干：1 层 LSTM，hidden size 64；
- TRA：3 个潜在状态，hidden size 16；
- epoch 30，early stop 5，每 epoch 最多 50 step；
- seed：0/1/2；
- 标签：Qlib 官方 CSRankNorm 形式；
- 评价：重新按“每日股票截面”计算外部 Rank IC，**不采用 TRA 日志里约 0.8 的
  样本级内部 IC**。

A/B 六次运行耗时 1,375 秒，C 三次运行耗时 678 秒。

### 6.1 三seed集成

| Fold与数据段 | Rank IC | Rank ICIR | Top1简化净累计 | Top3简化净累计 |
|---|---:|---:|---:|---:|
| A selection | 0.0688 | 0.5152 | +32.71% | -0.10% |
| A test | 0.0573 | 0.3351 | -35.34% | -19.08% |
| B selection | 0.0422 | 0.2169 | +218.74% | +63.26% |
| B test | 0.0589 | 0.3034 | +24.37% | +1.45% |
| C selection | 0.0609 | 0.3719 | +30.05% | +18.16% |
| C test | 0.0443 | 0.2511 | +5.90% | -22.72% |

三 seed 的截面分数 Spearman 相关在 C selection/test 分别为 0.836/0.811，说明结果
不是单个随机种子偶然产生。但简化 Top-K 收益在 fold 间高度不稳定，必须执行严格
回测。

## 7. 与XGB240、Fixed Ensemble的共同样本严格比较

旧基线 Fold3 有预测的区间是 2026-02-24~2026-07-17。比较时取：

- 三种模型共同的 99 个交易日；
- 三种模型共同的 13,402 个股票日（同一 300 样本的有效交集）；
- 同一份 TRA 原始绝对收益标签；
- 主口径 `baseline`，不使用带有同Bar理想化信息的 `event_guard`。

| 模型 | Top1净累计 | Top3净累计 | Top5净累计 | Top10净累计 |
|---|---:|---:|---:|---:|
| CPU TRA（三seed） | -29.45% | **-12.35%** | **-16.10%** | **-15.44%** |
| XGBoost-240 | -30.27% | -14.61% | -24.34% | -33.34% |
| Fixed Ensemble | -34.45% | -35.25% | -33.56% | -35.11% |
| CSI1000同期 | -13.84% | -13.84% | -13.84% | -13.84% |
| CSI300同期 | +0.07% | +0.07% | +0.07% | +0.07% |

TRA Top3 比 CSI1000 多约 1.73 个百分点，但仍亏 12.35%；Top1 胜率只有 40.4%。
因此它只是研究候选，不能解释为“已经获得更好的实盘策略”。旧 Fixed 曾报告的高收益
使用了不同股票池/区间，不能与本表直接混用；本表专门消除了共同样本差异。

可读结果与图：

```text
.qlibAssistant/analysis/tra_xgb240_fixed_strict_common_260818_0050/strict_summary.csv
.qlibAssistant/analysis/tra_xgb240_fixed_strict_common_260818_0050/strict_daily.csv
.qlibAssistant/analysis/tra_xgb240_fixed_strict_common_260818_0050/strict_baseline_comparison.png
```

## 8. 为什么Rank IC提高但Top-K仍亏损

Rank IC 衡量全截面数百只股票的平均排序。它可以因为中间大量股票的次序更合理而
提高，但个人交易只买 Top1/Top3，收益由最顶部极少数样本决定。本轮还存在：

- 一日标签噪声大，顶部预测容易被短期反转吞噬；
- 截面归一化标签只保证相对排序，不校准绝对收益是否能覆盖成本；
- Top1/Top3 对单日极端股票敏感，样本有效量远小于 IC 使用的全截面样本；
- 日频高换手使微弱预测优势被费用和错误成交时点快速消耗；
- `event_guard` 不是信号日可知数据，不能用它美化主结果。

## 9. 复现命令

```bash
PY=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python

$PY script/download_tushare_daily_supplement.py
$PY script/build_tushare_daily_factors.py

$PY script/run_intraday_factor_experiment.py \
  --stage dev \
  --feature-sets alpha158 combined daily alpha_daily intraday_daily all \
  --models XGBoost LightGBM \
  --run-tag multisource_raw_dev_fair_CUSTOM

$PY script/run_cpu_tra_supplement.py \
  --folds A B --seeds 0 1 2 --top-features 40 \
  --epochs 30 --early-stop 5 --max-steps 50 \
  --run-tag cpu_tra_daily_dev_CUSTOM

$PY script/summarize_cpu_tra_supplement.py \
  .qlibAssistant/experiments/cpu_tra_daily_dev_CUSTOM
```

只有 A/B 结果冻结后才允许运行 C：

```bash
$PY script/run_cpu_tra_supplement.py \
  --folds C --seeds 0 1 2 --top-features 40 \
  --epochs 30 --early-stop 5 --max-steps 50 \
  --run-tag cpu_tra_daily_finalC_CUSTOM
```

## 10. 下一轮建议

1. 训练独立的“未来绝对收益”与“T+1 可买概率”模型，使用
   `期望收益 × 可买概率 - 成本 - 安全边际` 决定是否交易。
2. 在新的开发切分上校准空仓阈值，禁止根据本轮 C test 继续调阈值。
3. 测试 3/5 日标签与分批持有，降低一日噪声和换手；对应标签、训练与回测必须同时改。
4. 用主板全量样本复核 300 样本结论，新增一段真正未来的 shadow 数据后再决定上线。
5. 分钟数据优先用于 T 日 14:30 前可知的买入可行性/尾盘状态模型，不再直接无筛选地
   拼入 Alpha158。

