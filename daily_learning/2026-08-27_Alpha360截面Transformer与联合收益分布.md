# 2026-08-27 Alpha360 截面 Transformer 与联合收益分布

## 用户确认的方案

优先实现本模型，TRA 后排。Mac 用于开发和小测试，台式机 RTX 3060 12GB
用于正式训练。不同时跑两个 GPU 训练任务。

输入为历史 CSI1000 成分股的 Alpha360：每只股票 60 个交易日、6 个量价通道。
不能把股票列表的顺序当成位置；时间位置编码只加在 60 天维度上。
用户提出并采纳：建立股票代码词典，每只股票对应随机初始化后冻结的 16 维身份向量。
身份向量不是行业信息，冻结也不等于不会过拟合；后续可用 `--stock-embedding-width 0` 对照。

## 网络与损失

1. 每个时间点的 6 维特征线性映射至 64 维，加可学习的时间位置编码。
2. 共享时序 Transformer：2 层、4 heads、FFN=256、dropout=0.1。
3. Attention pooling 得到每只股票的 64 维历史表示。
4. 拼接冻结的 16 维股票身份向量，再经 80→64 映射。
5. 截面 Transformer：2 层、4 heads，让同一信号日的全部股票相互注意。
6. 输出 9 个数：3 个均值和下三角 Cholesky 矩阵的 6 个元素。

可训练参数约 21 万。模型虽然小，60 天时序和约 1000 只股票的注意力仍需要计算。
首轮每次处理一个完整日期，累积 4 个日期的梯度后更新一次；不拆散截面。

信号在 T 日收盘后产生；第一版“早盘”=开盘价，“尾盘”=收盘价，
**不是 10:30 和 14:55 的实际可成交价**。

| 三段目标 | 定义 |
|---|---|
| a | ln(C[T+1]/O[T+1]) |
| b | ln(O[T+2]/C[T+1]) |
| c | ln(C[T+2]/O[T+2]) |

模型输出 `(mu_a,mu_b,mu_c)` 和 `L`，并用 `Sigma=L L^T` 构造合法协方差。
L 对角线为 softplus(raw)+0.001；其他三个元素可正可负。
三段联合高斯 NLL 用 Cholesky 求解，不直接求逆，不对秩不足的四目标协方差求逆。

四个输出分别是 `a+b+c`、`b`、`a+b`、`b+c`，对应：

| 文件字段前缀 | 买入→卖出 |
|---|---|
| open1_close2 | 明日开盘→后日收盘 |
| close1_open2 | 明日收盘→后日开盘 |
| open1_open2 | 明日开盘→后日开盘 |
| close1_close2 | 明日收盘→后日收盘，与 Fixed 持仓区间一致 |

四个对数收益满足 y1+y2=y3+y4。普通收益不能直接线性相加。
预测四区间的均值、方差和相关性均由三段联合分布推导。
普通收益期望是 exp(log_mean+log_variance/2)-1，不是 exp(log_mean)-1。

标签不做截面标准化；训练时仅乘固定倍率 100，输出时还原。
编码器 CUDA 使用支持时的 BF16；协方差与 NLL 保持 float32。
AdamW lr=0.0003、weight_decay=0.0001、梯度范数上限 1。
最多 50 epoch；Valid NLL 连续 10 轮不改善早停。
预测方差不是“保证正确的置信度”，需要看概率校准和区间覆盖率。

## 日期、数据与防泄漏

保持 Fixed Fold3 的非训练段，并采用 120 个月名义训练窗：

| 分段 | 开始 | 结束 | 用途 |
|---|---|---|---|
| Train | 2015-04-17 | 2025-04-16 | 梯度训练与输入标准化拟合 |
| Valid | 2025-04-17 | 2025-09-16 | 早停 |
| Selection-valid | 2025-09-17 | 2026-02-16 | 完成后评估，后续选模/集成 |
| Test | 2026-02-17 | 2026-07-17 | 完成后一次评估，不用于逐轮早停 |

Train、Valid、Selection-valid 各最后两个交易日的标签清空，避免 T+2 标签跨段。
训练使用信号日当时的成分股，历史窗口可以包含它进入 CSI1000 之前的已知行情。
不能用当前成分股列表代替历史列表。

**新发现的数据缺口**：本机 csi1000.txt 在 2015 年 4 月至 5 月初只含 3 只股票。
所以导出器要求至少 800 只成分股：不完整的 Train 日期显式排除、写入 manifest；
Valid/Selection-valid/Test 不完整则报错，不悄悄筛掉。名义边界不变，有效训练起点
可能后移，最终以 manifest 的 `excluded_train_dates` 和 `parts` 为准。
800 是数据完整性检查阈值，不是选股数量，也不是要求所有股票都有可用未来标签。

Alpha360 的 360 列按 CLOSE59..0、OPEN59..0、HIGH、LOW、VWAP、VOLUME 排列。
数据导出先读取六种基础行情，再向量化构造相同的 60 天比例特征；
第一块数据会与安装的 Qlib Alpha360DL 全部 360 列及未来价格逐项比对，失败即停止。
这样可以减少大量重复的表达式读取，不能改变因子含义。

输入每列的均值/标准差只在 Train 拟合；非有限值在标准化后填 0。
没有对预测标签做 CSZScoreNorm，也没有用真实标签决定 Attention。

两台电脑的原行情文件哈希不同，所以未直接使用台式机原有行情，
也未覆盖它。已将本机行情打包，放到新实验独立的 provider 目录。
归档 SHA256：`1d5833bbfe31e1e6263c80dfe688336ac8acf5b34342f8f5de2565b602d0dff8`。

## 代码、目录和命令

- 模型：`roll/alpha360_cross_stock.py`
- 数据/训练/评估入口：`script/train_alpha360_cross_stock.py`
- 单元测试：`tests/test_alpha360_cross_stock.py`
- Windows 串行执行器：`script/run_alpha360_cross_stock_windows.ps1`
- Windows 一次性任务启动器：`script/launch_alpha360_cross_stock_windows.ps1`
- Git 分支：`codex/alpha360-cross-stock-nll-260827`

本机实验目录：
`/Users/hmax/qlibAssistant/.qlibAssistant/remote_runs/alpha360_cross_stock_fold3_120m_260827/`

台式机目录：
`E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827\`

Python：台式机 `E:\Miniconda\envs\qlibass\python.exe`；
本机 `/Users/hmax/miniconda3/envs/qlibAssistant/bin/python`。

在项目根目录，手动执行方式：

```bash
python script/train_alpha360_cross_stock.py export --data /path/to/data --provider /path/to/cn_data
python script/train_alpha360_cross_stock.py train --data /path/to/data --output /path/to/benchmark --benchmark-only
python script/train_alpha360_cross_stock.py train --data /path/to/data --output /path/to/run
```

`--device cpu` 用于本地小测试，默认 `cuda`，不可用时直接报错，不偷偷改成 CPU。
`--segments-json` 接收日期配置文件，不是内联 JSON 字符串。
`script/alpha360_cross_stock_smoke_segments.json` 仅用于代码烟雾测试，不是正式报告。

### 日志与断点

Windows 的 `train.log` 是实时日志，`data/export_status.json` 是导出进度，
`run/status.json` 是训练/评估状态。训练过程中每 25 个日期更新一次状态。

```powershell
Get-Content E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827\train.log -Tail 30 -Wait
```

运行目录创建 `STOP_AFTER_EPOCH` 文件，会在完成当前 epoch、保存检查点后暂停。
恢复前移走这个标记，再使用相同参数加 `--resume`；不能修改输入、结构或代码后
冒充同一个可恢复实验。每轮检查点包含模型、优化器、随机状态和历史指标。

一次性任务没有每天重跑的触发器；台式机需保持开机和用户登录，不要注销。
Mac 合盖不会终止独立的 Windows 训练任务，但 Mac 不再能实时显示新日志。

### 产物含义

| 产物 | 含义 |
|---|---|
| data/manifest.json | 精确数据段、排除日期、特征定义、样本数、文件校验 |
| benchmark/benchmark.json | 真实截面的训练耗时与估算，不是最终效果 |
| run/configuration.json | 网络、超参、设备、代码与数据哈希 |
| run/stock_ids.json | 冻结身份向量对应的股票代码词典 |
| run/normalizer.npz | 仅 Train 拟合的输入标准化参数 |
| run/best_model.pt | Valid NLL 最优权重及配置 |
| run/last_checkpoint.pt | 可恢复的最近完整 epoch |
| run/epoch_metrics.csv | 每轮 loss、Valid 指标与耗时 |
| run/summary.csv | Valid、Selection-valid、Test 的分布和预测指标 |
| run/*_predictions.csv | 每只股票四种区间的预测均值、方差、概率及真实收益 |
| run/*_daily_metrics.csv | 每日 Rank IC、MAE、Brier、区间覆盖率 |

这是独立实验产物，**暂未注册为 MLflow recorder**。
概率和收益指标还不是严格交易回测。涨跌停、停牌、买入替补、手续费、
event_guard 和主板过滤需要在后续统一执行回测中比较，不能把原始目标收益当成可成交收益。
当前只训练第一组主模型，尚未声称完成随机种子集成或各架构消融。

## TRA 的处理

按用户新优先级，已停止原 TRA 进程 PID 5396，保留原有日志、最佳模型、最近检查点，
另备份 `run/priority_pause_checkpoint_260827.pt`；停止前最后确认完成预训练 epoch 6。
原任务目录为 `tra_fixed_fold3_120m_260826`。
其旧入口尚无自动恢复功能，恢复 TRA 时必须先补齐恢复逻辑，不能直接从头重跑冒充续训。

## 验证与进度记录

- 单元测试已覆盖字段顺序、三段标签、边界 purge、NLL 与 PyTorch 分布一致性、
  股票排列等变性、冻结身份向量、padding mask、反向传播和未来价格不进入特征。
- 已完成 3 股票的极小端到端功能测试；其 IC 不具有经济意义，不作效果汇报。
- 完整截面 CUDA 短测和正式训练状态，见本节后续运行记录。
