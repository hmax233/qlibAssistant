# 2026-07-21 前向评估、科创板归因与 DoubleEnsemble

## 今日改动

- `evaluate_batch.py` 自动输出 `board_variant_summary.csv` 和 `board_variant_daily.csv`，同时评估完整股票池与剔除科创板后的Top1/3/5/10毛收益、净收益、换手、回撤、Sharpe及收益集中度。
- 新增 `evaluate_forward_recorder.py`，解决原Test结束后只能截取旧 `pred.pkl`、不能对最新日期重新推理的问题。
- 新增 `evaluate_consensus_strategies.py`，有限测试XGB240与DoubleEnsemble-240 Fold3的截面一致投票和两日持续性规则。
- 修正 `summary.test_trading_days`：使用真正具备预测和完整标签的日数，而不是尚未剔除空标签的日期数。

## 关键结果

XGB240 Fold3 Top3：全池扣成本 +44.68%，剔除科创板后 +11.88%；科创板对历史亮眼收益贡献显著。

2026-07-15～17三个最新可验证信号日：XGB240全池Top3净累计 +1.44%，剔除科创板后 -3.74%。样本只有3天，不能据此判断模型失效。

DoubleEnsemble-240 Top3扣成本：Fold1全池/剔科创 +61.16%/+53.23%，Fold2 +68.14%/+76.44%，Fold3 -5.93%/-12.87%。最新Fold明显变差，不适合作为当前单独主模型。

XGB240+DE240的同Test探索里，两个模型同时进入Top10的规则表现很高，但两日持续性规则明显亏损。所有这些投票结果已经使用Fold3 Test观察，只能作为下一轮Walk-forward或前向验证假设，不能作为无偏结论。

## 报告目录

- XGB前向三日：`.qlibAssistant/analysis/forward_evaluation_20260721_22_55_01/`
- DoubleEnsemble Fold1：`.qlibAssistant/analysis/evaluation_20260721_22_50_05/`
- DoubleEnsemble Fold2：`.qlibAssistant/analysis/evaluation_20260721_22_50_19/`
- DoubleEnsemble Fold3：`.qlibAssistant/analysis/evaluation_20260721_22_50_33/`
- 投票探索：`.qlibAssistant/analysis/consensus_strategy_20260721_22_54_39/`
