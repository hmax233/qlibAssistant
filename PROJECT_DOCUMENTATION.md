# QlibAssistant 量化选股系统 - 技术文档

## 1. 项目概述

QlibAssistant 是基于 **Microsoft Qlib** 框架构建的全自动量化选股系统，专注于 **CSI300 指数成分股**的次日收益预测。系统采用多模型集成策略，通过滚动训练持续更新模型，输出每日选股信号。

### 1.1 核心特性

- **多模型集成**：支持 Linear、XGBoost、LightGBM、CatBoost、DoubleEnsemble 等多种模型
- **滚动训练机制**：支持 expanding（扩展窗口）、sliding（滑动窗口）、custom（自定义）三种滚动方式
- **全自动流程**：数据更新 → 模型训练 → 预测选股 → 结果输出
- **MLflow 模型管理**：统一管理模型版本、指标和预测结果
- **Alpha158 因子**：基于 158 个技术分析因子构建特征

### 1.2 技术栈

| 组件 | 技术 | 版本要求 |
|------|------|----------|
| 框架 | Microsoft Qlib | 0.2.x+ |
| 语言 | Python | 3.10+ |
| 数据库 | Qlib Data | - |
| 模型管理 | MLflow | - |
| 数据获取 | AkShare | - |

---

## 2. 项目结构

```
qlibAssistant/
├── .qlibAssistant/           # 运行时数据（可配置路径）
│   ├── mlruns/              # MLflow 实验数据（模型、指标、配置）
│   └── analysis/            # 预测结果输出目录
├── roll/                    # 核心代码目录
│   ├── roll.py              # 主入口（命令行调度器）
│   ├── config.yaml          # 运行配置文件
│   ├── myconfig.py          # 模型和数据集配置
│   ├── datacli.py           # 数据管理子模块
│   ├── traincli.py          # 训练引擎子模块
│   ├── modelcli.py          # 模型仓库子模块
│   ├── model_backup.py      # 模型备份/恢复
│   ├── model_review.py      # 模型复盘/回测
│   └── utils.py             # 工具函数集
├── script/                  # 辅助脚本
├── page/                    # 前端页面（可选）
├── model_pkl/               # 模型备份文件存放
└── .github/workflows/       # CI/CD 工作流
```

---

## 3. 核心代码解析

### 3.1 主入口：`roll.py`

**职责**：命令行入口，配置加载，子命令分发

```python
class RollingTrader:
    def __init__(self, config_path: str = "./config.yaml", **kwargs):
        # 1. 加载配置文件 + 命令行参数合并
        self._load_and_merge_config(kwargs)
        
        # 2. 自动补全预测日期（默认取数据集最新日期）
        self._ensure_predict_dates()
        
        # 3. 延迟初始化子模块（避免提前加载 Qlib）
        self.data = DataCLI(**self.params)  # 立即初始化
        self._train = None                  # 延迟初始化
        self._model = None                  # 延迟初始化
        
        # 4. 修复 MLflow 路径（解决迁移问题）
        fix_mlflow_paths(self.params.get("uri_folder"))
```

**关键设计**：
- **延迟初始化**：`train` 和 `model` 属性使用 `@property` 装饰器，首次访问时才创建实例，避免纯数据操作时加载 Qlib
- **配置优先级**：命令行参数 > 配置文件 > 默认值

### 3.2 配置管理：`config.yaml`

```yaml
# 文件夹路径配置
uri_folder: "../.qlibAssistant/mlruns/"      # MLflow 实验存储路径
analysis_folder: "../.qlibAssistant/analysis/" # 预测结果输出路径
provider_uri: "~/.qlib/qlib_data/cn_data/"   # Qlib 数据源路径

# 模型与数据集配置
model_name: Linear           # 模型名称（Linear/XGBoost/LightGBM/CatBoost/DoubleEnsemble）
dataset_name: Alpha158       # 数据集名称（Alpha158/Alpha360）
stock_pool: csi300           # 股票池（csi300/csi100）
step: 60                     # 滚动步长（天数）

# 滚动类型
rolling_type: expanding      # expanding/sliding/custom

# 过滤器配置
model_filter:                # 模型名称过滤（支持正则）
  - .*
rec_filter:                  # 模型质量过滤（IC/ICIR/Rank IC/Rank ICIR 阈值）
  - ic: 0.001
  - icir: 0.001
  - rankic: 0.001
  - rankicir: 0.001
```

### 3.3 模型配置：`myconfig.py`

**支持的模型**：

| 模型名称 | 类名 | 说明 |
|----------|------|------|
| Linear | LinearModel | 岭回归，简单高效，基准模型 |
| XGBoost | XGBModel | XGBoost 梯度提升树 |
| LightGBM | LGBModel | LightGBM 梯度提升树 |
| CatBoost | CatBoostModel | CatBoost 梯度提升树 |
| DoubleEnsemble | DEnsembleModel | 双重集成模型，效果最佳但最慢 |
| KRNN | KRNN | 基于 CNN+RNN 的深度学习模型 |
| Sandwich | Sandwich | 三明治神经网络模型 |

**数据集配置**：

```python
def get_dataset_config(
    dataset_class=DATASET_ALPHA158_CLASS,
    train=("2015-01-01", "2016-12-31"),
    valid=("2017-01-01", "2017-02-28"),
    test=("2017-03-01", "2026-12-31"),
    handler_kwargs=None,
):
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": dataset_class,  # Alpha158 或 Alpha360
                "module_path": "qlib.contrib.data.handler",
                "kwargs": get_data_handler_config(**kwargs),
            },
            "segments": {"train": train, "valid": valid, "test": test},
        },
    }
```

### 3.4 数据管理：`datacli.py`

**职责**：市场数据下载、更新、状态检查

```python
class DataCLI:
    def need_update(self) -> bool:
        # 对比远程最新交易日和本地数据最新日期
        latest_data = get_latest_trade_date_ak()  # 从 AkShare 获取
        local_data = get_local_data_date(self.kwargs["provider_uri"])
        return str(latest_data) != str(local_data)
    
    def update(self, proxy="A"):
        # 通过 GitHub 代理下载 Qlib 数据压缩包
        url = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
        # 支持多个代理（A/B/C/D）应对网络问题
```

**数据来源**：由 `chenditc/investment_data` 项目维护的 Qlib 格式 CN 市场数据

### 3.5 训练引擎：`traincli.py`

**职责**：滚动训练调度、断点续训、多进程训练

```python
class TrainCLI:
    def __init__(self, step=40, region=REG_CN, **kwargs):
        # 1. 初始化 Qlib
        qlib.init(provider_uri=provider_uri, region=region, exp_manager=exp_manager)
        
        # 2. 获取任务配置
        self.task_config = get_my_config(model_name, dataset_name, stock_pool)
        
        # 3. 创建滚动生成器
        self.rolling_gen = RollingGen(step=step, rtype=rolling_type, ...)
    
    def start(self):
        # 1. 生成滚动任务列表
        tasks = self.gen()
        
        # 2. 逐个执行训练（支持断点续训）
        self.task_training(tasks)
```

**滚动策略**：

| 滚动类型 | 说明 |
|----------|------|
| expanding | 扩展窗口：训练集从起点开始，逐步扩展 |
| sliding | 滑动窗口：固定窗口大小，向后滑动 |
| custom | 自定义：按 12/24/36/48/60 个月分别训练 |

**断点续训机制**：
```python
# 检查已训练的时间段，跳过重复训练
for rid in exp.list_recorders():
    rec = exp.get_recorder(recorder_id=rid)
    task = rec.load_object("task")
    train_time_seg = task["dataset"]["kwargs"]["segments"]["train"]
    exp_train_time_segs_list.append(train_time_seg)

# 跳过已存在的训练段
if train_time_seg in exp_train_time_segs_list:
    continue
```

**多进程训练**：
使用 `multiprocessing.Process` 创建子进程训练，避免主进程内存泄漏：
```python
def run_train_blocking(task, exp_name, region, **kwargs):
    p = multiprocessing.Process(target=_train_worker, args=(task, exp_name, region), kwargs=kwargs)
    p.start()
    p.join()
    return p.exitcode == 0
```

### 3.6 模型仓库：`modelcli.py`

**职责**：模型查询、预测、结果收集、复盘

#### 3.6.1 模型筛选逻辑

```python
def get_model_list(self):
    # 1. 遍历 MLflow 实验
    exps = R.list_experiments()
    
    for name in exps:
        # 过滤不匹配的实验名称
        if not check_match_in_list(name, model_filter):
            continue
        
        # 检查每个 recorder 是否有效（有必要的 artifacts）
        for rid in exp.list_recorders():
            recorder = exp.get_recorder(recorder_id=rid)
            if self._is_valid_recorder(recorder):
                mc.rid.append(rid)
    
    # 2. 根据 validation Rank ICIR 计算模型权重
    total_rank_icir = sum(self.rid_rank_icir[rid] for mc in ret for rid in mc.rid)
    for mc in ret:
        for rid in mc.rid:
            self.rid_weight[rid] = self.rid_rank_icir[rid] / total_rank_icir
```

#### 3.6.2 预测流程

```python
def analysis(self):
    ret = []
    model_list = self.get_model_list()
    
    for mc in model_list:
        for rid in mc.rid:
            # 1. 加载模型和任务配置
            rec = exp.get_recorder(recorder_id=rid)
            task = rec.load_object("task")
            model = rec.load_object("params.pkl")
            
            # 2. 修改 test 时间段为预测日期
            dataset_config['kwargs']['segments']['test'] = (predict_date1, predict_date2)
            
            # 3. 执行预测
            dataset = init_instance_by_config(dataset_config)
            pred_score = model.predict(dataset, segment="test")
            
            ret.append([mc.exp_name, rid, pred_score])
    return ret
```

#### 3.6.3 结果收集与集成

```python
def collect(self, results):
    # 1. 合并所有模型预测结果
    df_final = pd.concat(processed_list, axis=0, ignore_index=True)
    
    # 2. 加载真实标签（次日收益率）
    real_df = self.get_real_label()
    
    # 3. 计算加权平均分数（基于模型权重）
    ret_df = (
        group_df.groupby('instrument')
        .apply(lambda g: pd.Series({
            "avg_score": (g['score'] * g['weight']).sum() / g['weight'].sum(),
            "pos_ratio": (g['score'] > 0).mean(),
        }))
    )
    
    # 4. 稳健性过滤
    ret_filter_df = self.filter_ret_df(ret_df)
    
    # 5. 保存结果
    self._save_results(df_final, func_name, latest_stock_list)
```

#### 3.6.4 稳健性过滤

```python
def filter_ret_df(self, df):
    # 波动率过滤：短期波动率不能过高
    df = df[(df['STD5'] < 0.10) & (df['STD20'] < 0.10) & (df['STD60'] < 0.10)]
    
    # 波动率一致性：短期波动不能远超长期波动
    df = df[df['STD5'] < (df['STD60'] * 2)]
    
    # 趋势过滤：确保处于上升趋势
    df = df[(df['ROC10'] > 0.80) & (df['ROC20'] > 0.80) & (df['ROC60'] > 0.80)]
    
    # 避免过热：涨幅不能过大
    return df[df['ROC20'] < 1.30]
```

#### 3.6.5 真实标签计算

```python
def get_real_label(self, dates=None, instruments='csi300'):
    # Ref($close, -2)/Ref($close, -1) - 1
    # Ref($close, -1) = 下一交易日收盘价
    # Ref($close, -2) = 再下一交易日收盘价
    # 所以计算的是次日收益率
    df = D.features(D.instruments(instruments), 
                   ['Ref($close, -2)/Ref($close, -1) - 1'],
                   start_time=dates['start'], end_time=dates['end'], freq='day')
    df.columns = ['real_label']
    return df
```

### 3.7 工具函数：`utils.py`

**核心工具函数**：

| 函数 | 用途 |
|------|------|
| `get_latest_trade_date_ak()` | 获取最近一个已收盘的交易日 |
| `get_local_data_date()` | 获取本地数据最新日期 |
| `fix_mlflow_paths()` | 修复 MLflow 配置中的路径前缀 |
| `generate_qlib_segments()` | 按 9:2:1 比例生成 train/valid/test 时间段 |
| `get_normalized_stock_list()` | 获取标准化 A 股股票列表（多源 fallback） |
| `check_match_in_list()` | 正则匹配列表 |

---

## 4. 数据流与处理流程

### 4.1 完整流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         数据更新阶段                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  AkShare 获取最新交易日 ──→ 对比本地数据 ──→ 需要更新? ──→ 下载 Qlib 数据   │
│                                                        ↓                │
│                                                   解压到 provider_uri    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         模型训练阶段                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  配置加载 ──→ 任务生成(RollingGen) ──→ 滚动训练 ──→ MLflow 保存           │
│                  ↓                                                      │
│           expanding/sliding/custom                                      │
│                  ↓                                                      │
│           每个滚动窗口训练一个模型切片                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         预测选股阶段                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  加载模型列表 ──→ 遍历预测 ──→ 加权集成 ──→ 稳健性过滤 ──→ 输出结果        │
│       ↓                 ↓           ↓              ↓                    │
│  MLflow查询       模型.predict()   Rank ICIR加权   STD/ROC过滤            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 日期语义

| 日期字段 | 含义 |
|----------|------|
| `predict_dates` | 预测基准日（使用该日及之前数据预测） |
| 预测结果 | 次日收益率（基准日+1 → 基准日+2 的涨跌幅） |

**示例**：
- 若 `predict_dates = {"start": "2026-07-03", "end": "2026-07-03"}`
- 使用 7 月 3 日及之前的特征数据
- 预测 7 月 4 日的涨跌情况（7月4日收盘价 / 7月5日收盘价 - 1）

### 4.3 模型集成策略

系统采用 **validation Rank ICIR 加权平均**策略。validation 用于选模和定权，test 只用于最终泛化评估，避免利用测试集信息挑模型：

1. **收集所有有效模型**：根据 validation 的 IC/ICIR/Rank IC/Rank ICIR 阈值过滤
2. **计算模型权重**：`weight = valid_rank_icir / sum(all_valid_rank_icir)`
3. **加权集成预测**：`avg_score = sum(score_i * weight_i) / sum(weight_i)`
4. **看多比例**：`pos_ratio = (score > 0).mean()`

---

## 5. 输出文件说明

### 5.1 输出目录结构

```
.qlibAssistant/analysis/selection_YYYYMMDD_HH_MM_SS/
├── 2026-07-03_ret.csv          # 全量预测结果（所有 CSI300 股票）
├── 2026-07-03_filter_ret.csv   # 过滤后预测结果（经过稳健性筛选）
├── total.csv                   # 完整特征数据（包含 Alpha158 全部因子）
└── total.md                    # 模型信息汇总（IC/ICIR/Rank IC/Rank ICIR）
```

### 5.2 文件内容详解

#### `2026-07-03_ret.csv` 和 `2026-07-03_filter_ret.csv`

| 字段 | 类型 | 含义 |
|------|------|------|
| `instrument` | str | 股票代码（如 SH600000） |
| `avg_score` | float | 加权平均预测分数（越高越看好） |
| `pos_ratio` | float | 看多模型比例（0~1） |
| `rank` | int | 按 `avg_score` 从高到低的当日排名 |
| `model_count` | int | 参与该股票集成的模型记录数 |
| `valid_model_count` | int | 实际给出非空预测分数的模型数 |
| `positive_model_count` | int | 给出正预测分数的模型数 |
| `real_label` | float | 真实次日收益率（未收盘时为 NaN） |
| `error` | float | 预测误差（score - real_label） |
| `abs_error` | float | 绝对误差 |
| `code` | str | 股票代码 |
| `name` | str | 股票名称 |
| `datetime` | datetime | 预测基准日 |
| 其他字段 | float | Alpha158 因子特征值 |

CSV 使用 `index=False` 保存，不再生成无业务含义的 `Unnamed: 0` 列。
如果 `valid_model_count < model_count`，说明部分模型对该股票产生了 NaN，需检查模型数值稳定性。

#### `total.md`

记录参与预测的所有模型及其评估指标：

```markdown
Experiment: EXP_LinearModel_Alpha158_csi300_expanding_step60_s_20260530_14
Recorder: 439eca0070524e7cabe406a6ff30c6c1
Model: {
    'model': 'LinearModel',
    'ic_info': {'IC': 0.008, 'ICIR': 0.064, 'Rank IC': 0.035, 'Rank ICIR': 0.304},
    'data_train_vec': ['2021-05-28', '2025-02-27'],
    'weight': '0.110'
}
```

**评估指标解读**：

| 指标 | 含义 | 取值范围 | 解读 |
|------|------|----------|------|
| IC | 预测值与实际收益的线性相关系数 | [-1, 1] | 越高越好，>0 表示正向预测能力 |
| ICIR | IC 的信息比率（均值/标准差） | - | 越高越稳定 |
| Rank IC | 预测排名与实际收益排名的相关系数 | [-1, 1] | 越高越好 |
| Rank ICIR | Rank IC 的信息比率 | - | 越高越稳定 |

---

## 6. 配置与自定义

### 6.1 修改股票池

在 `config.yaml` 中修改：
```yaml
stock_pool: csi100    # 改为 csi100 或自定义池
```

### 6.2 切换模型

在 `config.yaml` 中修改：
```yaml
model_name: XGBoost   # Linear/XGBoost/LightGBM/CatBoost/DoubleEnsemble
```

### 6.3 调整滚动步长

```yaml
step: 60              # 每 60 天滚动一次
```

### 6.4 自定义预测日期

```yaml
predict_dates:
  - start: 2026-07-03
    end: 2026-07-03
```

### 6.5 修改模型过滤条件

```yaml
rec_filter:
  - ic: 0.001        # IC 阈值
  - icir: 0.001      # ICIR 阈值
  - rankic: 0.001    # Rank IC 阈值
  - rankicir: 0.001  # Rank ICIR 阈值
```

---

## 7. 命令行使用

### 7.1 数据更新

```bash
cd ./roll && python ./roll.py data update
cd ./roll && python ./roll.py data status    # 检查数据状态
cd ./roll && python ./roll.py data need_update  # 检查是否需要更新
```

### 7.2 模型训练

```bash
cd ./roll && python ./roll.py train start          # 启动滚动训练
cd ./roll && python ./roll.py train start_custom   # 自定义滚动训练
cd ./roll && python ./roll.py train need_train     # 检查是否需要训练
```

### 7.3 模型管理

```bash
cd ./roll && python ./roll.py model ls             # 列出所有模型
cd ./roll && python ./roll.py model ls --all       # 列出所有模型及详情
cd ./roll && python ./roll.py model clean          # 清理无效模型
cd ./roll && python ./roll.py model compress_mlruns   # 压缩模型备份
cd ./roll && python ./roll.py model decompress_mlruns # 解压模型备份
```

### 7.4 预测选股

```bash
# 使用默认日期（数据集最新日期）
cd ./roll && python ./roll.py model selection

# 指定预测日期
cd ./roll && python ./roll.py model selection --predict_dates '[{"start": "2026-07-03", "end": "2026-07-03"}]'
```

### 7.5 模型复盘

```bash
cd ./roll && python ./roll.py model review     # 模型复盘分析
cd ./roll && python ./roll.py model backtest   # 回测验证
```

---

## 8. Alpha158 因子体系

Alpha158 是 Qlib 内置的因子集，包含 158 个技术分析因子，分为以下几类：

| 因子类型 | 示例 | 数量 |
|----------|------|------|
| K线特征 | KMID, KLEN, KUP, KLOW | ~10 |
| 价格特征 | OPEN0, HIGH0, LOW0, VWAP0 | ~5 |
| 动量特征 | ROC5, ROC10, ROC20, ROC30, ROC60 | ~5 |
| 均线特征 | MA5, MA10, MA20, MA30, MA60 | ~5 |
| 波动特征 | STD5, STD10, STD20, STD30, STD60 | ~5 |
| Beta特征 | BETA5, BETA10, BETA20, BETA30, BETA60 | ~5 |
| 相关特征 | CORR5, CORR10, CORR20, CORR30, CORR60 | ~5 |
| 量价特征 | VMA5, VSTD5, WVMA5, VSUMP5 | ~10 |
| 其他特征 | 剩余约 100+ 个衍生因子 | ~100+ |

---

## 9. 高级主题

### 9.1 MLflow 实验结构

```
mlruns/
├── experiment_id/
│   ├── meta.yaml              # 实验元数据
│   └── recorder_id/
│       ├── artifacts/         # 模型文件和产物
│       │   ├── dataset/       # 数据集配置
│       │   ├── params.pkl     # 模型参数
│       │   ├── pred.pkl       # 预测结果
│       │   ├── task           # 任务配置
│       │   └── sig_analysis/  # 信号分析（IC.pkl, ric.pkl）
│       ├── metrics/           # 评估指标
│       │   ├── IC
│       │   ├── ICIR
│       │   ├── Rank IC
│       │   └── Rank ICIR
│       ├── params/            # 超参数
│       ├── tags/              # 标签
│       └── meta.yaml          # 记录器元数据
```

### 9.2 自定义模型

在 `myconfig.py` 中添加新模型配置：

```python
MY_CUSTOM_MODEL = {
    "class": "MyModelClass",
    "module_path": "my_module.path",
    "kwargs": {
        "param1": value1,
        "param2": value2,
    }
}

def get_model_config(model_name: str):
    match model_name:
        case "MyCustom":
            return MY_CUSTOM_MODEL
        # ... 其他模型
```

### 9.3 CI/CD 流程

项目包含 GitHub Actions 工作流：

- `analysis.yml`：定时运行模型预测
- `train.yml`：定时训练模型
- `build.yml`：构建测试
- `deploy_page.yml`：部署前端页面
- `review.yml`：代码审查

---

## 10. 学习路径建议

### 入门阶段
1. 理解 Qlib 框架基础概念（数据集、模型、任务）
2. 运行 `data update` 和 `model selection` 体验完整流程
3. 分析输出文件，理解预测结果含义

### 进阶阶段
1. 学习 `myconfig.py` 中的模型配置
2. 理解 `traincli.py` 中的滚动训练机制
3. 调整 `config.yaml` 参数，观察效果变化

### 高级阶段
1. 深入理解 Alpha158 因子体系
2. 添加自定义模型或因子
3. 优化稳健性过滤逻辑
4. 扩展回测和风险控制功能

---

## 11. 训练 / 预测 / 回测 实操教程

> 本节是动手教程，覆盖日常完整流程：更新数据 → 训练模型 → 预测选股 → 回测验证。
> 前面章节讲代码原理，本节讲"怎么跑"。环境为 macOS / Apple Silicon。

### 11.0 前置：环境与约定

- **两个 conda 环境**（用绝对路径调用，不必 activate）：
  - `qlibAssistant`：`/Users/hmax/miniconda3/envs/qlibAssistant/bin/python` —— 跑训练、预测、数据增量更新（含 tushare/pymysql/qlib/mlflow/各模型）
  - `qlib_env`：`/Users/hmax/miniconda3/envs/qlib_env/bin/python` —— 跑数据格式转换 `dump_qlib_bin.sh`（qlib 版本与仓库匹配）
- **所有 mlflow 命令必须加** `MLFLOW_ALLOW_FILE_STORE=true`（mlflow 3.14 不再默认支持文件存储后端）
- **两个仓库必须在 `local/hmax-fixes` 分支**：`/Users/hmax/qlib`（format_data 修复）和 `/Users/hmax/investment_data`（dump 脚本修复）的改动提交在此分支，切回 main 会丢、dump 会重现"丢最后一天"bug
- **一键脚本**：`/Users/hmax/qlibAssistant/update-predict.py` 封装"更新数据+预测"，日常首选；tushare token 存 `~/.config/tushare_token`

### 11.1 更新数据

数据流：Tushare(第三方代理 stockai888.top) → Dolt → CSV → Qlib 二进制 → `~/.qlib/qlib_data/cn_data/`

**日常一键**（推荐）：
```bash
/Users/hmax/miniconda3/envs/qlibAssistant/bin/python /Users/hmax/qlibAssistant/update-predict.py
```
脚本自动：查 dolt 最新日期 → 从交易日历算待补日期 → curl 探测 tushare 可用性 → 增量更新 → dump → 替换数据 + 重生成 csi300 成分股 → 跑预测 → 打印 Top10。

> - 数据通常收盘后 1-2 小时才出，当天没出就补到上一交易日。
> - 第三方代理对 Python SDK 的 TLS 连接偶有异常，脚本统一用 curl + HTTP/1.1 + 重试调用；这不是 VPN 前提。
> - 只更新数据不预测：手动分步见 memory（investment-data-dump-pipeline）。

#### Tushare 与 AkShare 的职责说明

- **价格、成交量、复权因子等模型输入数据**：始终来自 Tushare 代理，经 Dolt 和 Qlib 转换后供训练/预测使用。
- **股票代码与中文名称**：只用于结果 CSV 展示，不参与模型特征或分数计算。当前也优先从 Tushare `stock_basic` 获取，并缓存到 `.qlibAssistant/cache/stock_basic.csv`。
- 名称缓存 7 天内直接读取；过期后尝试刷新。代理暂时不可用时继续使用旧缓存，因此不会再回退到多个 AkShare 网站等待数分钟。
- `DataCLI` 仍保留原项目的 AkShare 状态检查/远程发布包下载逻辑，但本地一键更新脚本不依赖它更新行情。

### 11.2 训练模型

#### 何时训练
模型会"过期"（市场风格变了）。`need_train` 判断"模型数据日期 < 本地数据日期"就返回 True。**建议每周/每月训一次**，不必每天（DoubleEnsemble 单次约 37 分钟）。

#### 检查是否需要训练
```bash
cd /Users/hmax/qlibAssistant/roll
MLFLOW_ALLOW_FILE_STORE=true /Users/hmax/miniconda3/envs/qlibAssistant/bin/python ./roll.py train need_train
```

#### 训练命令
单模型（先试，Linear 最快 ~1 分钟）：
```bash
MLFLOW_ALLOW_FILE_STORE=true /Users/hmax/miniconda3/envs/qlibAssistant/bin/python ./roll.py \
  --pfx_name="EXP" --model_name="Linear" --dataset_name="Alpha158" \
  --stock_pool="csi300" --rolling_type="custom" train start_custom
```
默认 4 模型（XGBoost/Linear/LightGBM/CatBoost，不含耗时较长的 DoubleEnsemble）：
```bash
MLFLOW_ALLOW_FILE_STORE=true PATH="/Users/hmax/miniconda3/envs/qlibAssistant/bin:$PATH" \
  /Users/hmax/miniconda3/envs/qlibAssistant/bin/python ../script/run.py
```

#### 滚动训练的 5 个窗口（custom 模式）
每次 `train start_custom` 生成 5 个不同总长（12/24/36/48/60 月）的窗口，**都截止今天**，按 **9:2:1** 切 train/valid/test。因此：
- train 长度 = 9/18/27/36/45 月（不同）
- train 截止 = 今天 − 3/6/9/12/15 月（不同，因为更长的窗口按比例留更多 valid+test）
- 最近的数据留给 test，**不参与训练**（无前视偏差）
- 5 个窗口训 5 个模型，预测时一起加权投票

#### 断点续训
同一实验里已存在的 train 区间会跳过，只训新窗口。所以增量训练不重训旧的（Linear 重跑会秒过）。

#### 训练产物与查看
存入 `.qlibAssistant/mlruns/`，每模型一个实验、5 个窗口 = 5 个 recorder：
```bash
MLFLOW_ALLOW_FILE_STORE=true /Users/hmax/miniconda3/envs/qlibAssistant/bin/python ./roll.py model ls
```
`Recorders: X/Y` = 通过质量筛选 X 个 / 共 Y 个。IC 低于阈值（rec_filter）的会被排除，不进集成。

#### 加速
- Mac M5 Pro 15 核，**纯 CPU 训练**（GBDT 不支持 Apple GPU）
- `myconfig.py` 里 CatBoost/LightGBM/DoubleEnsemble 的 `thread_count`/`num_threads` 已设 14（匹配核数，留 1 核给系统）
- DoubleEnsemble 是瓶颈（~37 分钟），其余各 ~1 分钟
- 想大幅加速需 Linux + NVIDIA GPU（XGB/LGB/CatBoost 支持 CUDA）

### 11.3 预测选股

```bash
# 一键（数据已是最新时自动跳过更新，直接预测）
/Users/hmax/miniconda3/envs/qlibAssistant/bin/python /Users/hmax/qlibAssistant/update-predict.py --predict-only

# 或手动
cd /Users/hmax/qlibAssistant/roll
MLFLOW_ALLOW_FILE_STORE=true /Users/hmax/miniconda3/envs/qlibAssistant/bin/python ./roll.py model selection
```

`predict_dates` 自动取数据最新日期。流程：加载 mlruns 所有通过筛选的模型 → 各自预测 → 按 Rank ICIR 加权集成 → 稳健性过滤（STD/ROC）→ 输出。

当前 validation 筛选得到 26 个 recorder 时，预测约需数分钟。主要耗时是每个 recorder 都重新构建一次 Alpha158 数据集，不是股票名称网络请求；名称表使用本地缓存。

**输出**：`.qlibAssistant/analysis/selection_<时间戳>/`

### 11.4 Validation 选模与 recorder

`recorder` 是一次完整训练实例，而不是一种算法。它绑定了“模型算法 + train/valid/test 时间窗口 + 超参数 + 拟合参数 + 预测和评价产物”。因此，同一个 LightGBM 在 5 个时间窗口、2 次训练批次中会产生 10 个 recorder。

每个 recorder 的核心产物包括：

- `task`：模型、数据处理器和 train/valid/test 划分。
- `params.pkl`：训练完成的模型参数。
- `sig_analysis/`：原有 test 预测及 IC 指标，只用于最终评估。
- `valid_sig_analysis/`：validation 的预测、标签、IC、Rank IC 和汇总指标，用于筛选及加权。

历史 recorder 可补算 validation 指标：

```bash
conda run -n qlibAssistant python script/backfill_validation_metrics.py
```

脚本默认跳过已经生成的结果，可以安全地中断后续跑。新训练完成后会自动生成 `valid_sig_analysis`。

截至 2026-07-18，本地共有 50 个 recorder（5 种算法 × 5 个窗口 × 2 个训练批次）。旧 test 选模得到 28 个（5 月批次 17、7 月批次 11）；validation 选模得到 26 个（5 月批次 13、7 月批次 13）。最新预测目录为 `selection_20260718_22_35_06`，包含 26 × 300 = 7800 行逐 recorder 预测和 300 行集成结果。

批量训练脚本支持明确标记股票池和训练批次：

```bash
/Users/hmax/miniconda3/envs/qlibAssistant/bin/python script/run.py \
  --pool csi1000 --run-tag retrain260718
```

不要依赖 MLflow 自动生成的数字 `experiment_id` 识别业务批次；应使用包含股票池和 `run-tag` 的 `experiment_name`。统一测试报告使用：

```bash
/Users/hmax/miniconda3/envs/qlibAssistant/bin/python script/evaluate_batch.py \
  --experiment-pattern csi1000_custom_step0_retrain260718
```
- `_<日期>_ret.csv`：全量 csi300 预测（300 只，含 avg_score / pos_ratio）
- `_<日期>_filter_ret.csv`：过滤后推荐（几十只）
- `total.md`：参与模型及 IC/ICIR/权重
- `real_label` 为 NaN 正常（次日收益需等下个交易日数据）

### 11.4 回测与复盘

回测/复盘**基于历史预测**（`.qlibAssistant/analysis/selection_*` 目录），所以要先有连续多日的预测才能回测，一两天没意义。

#### 复盘（马后炮）：看历史预测准不准
```bash
cd /Users/hmax/qlibAssistant/roll
MLFLOW_ALLOW_FILE_STORE=true /Users/hmax/miniconda3/envs/qlibAssistant/bin/python ./roll.py model review
```
- 对每个历史预测日，取 Top{10,20,30,50,80,100} 股票，算实际收益
- 输出：`/tmp/review_result.md`、`../review_csv/`（即 `/Users/hmax/qlibAssistant/review_csv/`）
- 需要那些预测日的 real_label 已有（预测日 +1 的数据已更新）

#### 回测：TopK 组合净值曲线
```bash
MLFLOW_ALLOW_FILE_STORE=true /Users/hmax/miniconda3/envs/qlibAssistant/bin/python ./roll.py model backtest
```
- 用历史预测构建 TopK 组合（每天选 top N 只），算日收益、净值曲线、最大回撤
- benchmark = csi300；TopK = [10,20,30,50,80,100] 各跑一遍
- 输出：`/Users/hmax/qlibAssistant/backtest_csv/<top>_ret.csv` 与 `<top>_filter_ret.csv`（净值/回撤序列）
- 日期范围 = 最早到最新的预测日

> 意义：积累一段时间（如一个月）每日预测后，回测能看出"每天选 top30 的净值涨了多少、最大回撤多大、是否跑赢 csi300"。

### 11.5 日常节奏

| 频率 | 操作 | 命令 |
|------|------|------|
| 每天（收盘后 1-2h） | 更新数据 + 预测 | `update-predict.py` |
| 每周/每月 | 重训模型 | `train need_train` → `script/run.py` |
| 积累一段时间后 | 复盘 + 回测 | `model review` / `model backtest` |

### 11.6 常见问题
- **预测/训练报 mlflow 错** → 加 `MLFLOW_ALLOW_FILE_STORE=true`
- **数据更新 tushare SSL 失败** → VPN 拦截，脚本已用 curl（不用管）
- **dump 后最新日期缺失** → `/Users/hmax/qlib` 不在 `local/hmax-fixes` 分支（format_data bug）
- **预测 real_label 全 NaN** → 正常，次日数据没出
- **训练太慢** → 主要是 DoubleEnsemble；可先跳过它，或减 `n_estimators`（降质量换速度）

---

## 12. 自助研究流程：训练、滚动 Fold、统一评估与画图

本节用于自己完成一轮可复现的量化实验。所有命令都从项目根目录执行：

```bash
cd /Users/hmax/qlibAssistant
export MLFLOW_ALLOW_FILE_STORE=true
export XDG_CACHE_HOME=/tmp/qlibAssistant-cache
export MPLCONFIGDIR=/tmp/qlibAssistant-mpl
PY=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python
```

其中 `MLFLOW_ALLOW_FILE_STORE=true` 允许当前 MLflow 版本读取本地文件型实验仓库；另外两个变量把 Matplotlib 缓存放到可写目录，避免字体缓存警告。

### 12.1 给实验取一个可识别的名称

每次训练都应设置唯一 `run-tag`，建议格式为：

```text
研究目的_YYMMDD
```

例如：

- `retrain260718`：2026-07-18 常规重训。
- `splitval260719`：拆分选模验证集实验。
- `fold1_260719`：第一个滚动时间 Fold。

最终实验名类似：

```text
EXP_LinearModel_Alpha158_csi1000_custom_step0_splitval260719_20260719_16
```

定位实验时使用 `experiment_name/run-tag`，不要依赖无业务含义的数字 `experiment_id`。

### 12.2 训练一个或多个模型

只训练 Linear，生成默认 12/24/36/48/60 月五个 recorder：

```bash
$PY script/run.py \
  --pool csi1000 \
  --run-tag my_linear_260719 \
  --models Linear
```

训练 Linear 和 LightGBM：

```bash
$PY script/run.py \
  --pool csi1000 \
  --run-tag linear_lgb_260719 \
  --models Linear LightGBM
```

显式训练五种模型（包含耗时较长、默认已排除的 DoubleEnsemble）：

```bash
$PY script/run.py \
  --pool csi1000 \
  --run-tag full_260719 \
  --models XGBoost Linear DoubleEnsemble LightGBM CatBoost
```

只训练60个月窗口，并拆分训练监控验证集和选模验证集：

```bash
$PY script/run.py \
  --pool csi1000 \
  --run-tag linear60_split_260719 \
  --models Linear \
  --window-months 60 \
  --split-selection-valid
```

先查看将执行什么而不训练：

```bash
$PY script/run.py \
  --pool csi1000 \
  --run-tag dryrun \
  --models Linear \
  --window-months 60 \
  --split-selection-valid \
  --dry-run
```

主要参数：

| 参数 | 含义 |
|---|---|
| `--pool` | 股票池，例如 `csi300`、`csi1000`、`all` |
| `--run-tag` | 本批次业务标识，用于检索实验 |
| `--models` | 一个或多个模型名称 |
| `--window-months` | 指定总窗口月数；不传时默认五个窗口 |
| `--end-date` | 模拟当时只能看到该日期之前数据，用于历史 Fold |
| `--split-selection-valid` | 使用 `9:1:1:1` 的独立选模验证集 |
| `--dry-run` | 只打印命令，不训练 |

未传 `--models` 时默认训练 `XGBoost Linear LightGBM CatBoost`。DoubleEnsemble 的配置和历史 recorder 都会保留；需要专项实验时显式写入 `--models DoubleEnsemble` 即可。

### 12.3 理解 Validation 拆分

旧任务使用：

```text
train : valid : test = 9 : 2 : 1
```

其中 `valid` 既可能参与模型训练监控，又用于筛选 recorder。新实验推荐：

```text
train : valid : selection_valid : test = 9 : 1 : 1 : 1
```

- `train`：拟合模型参数。
- `valid`：模型训练监控和 early stopping。
- `selection_valid`：比较、筛选和加权 recorder。
- `test`：只用于最终报告，不能反过来选模型或调参数。

旧 recorder 没有 `selection_valid` 时，代码自动回退到 `valid`，所以历史实验仍可读取。

### 12.4 运行三个时间滚动 Folds

滚动 Fold 是多次执行“只用过去训练、用随后未来测试”，不能像普通机器学习交叉验证一样随机打乱股票时序。

当前现成脚本：

```bash
$PY script/run_rolling_folds.py 2>&1 | \
  tee .qlibAssistant/logs/rolling_folds_manual.log
```

另开终端观察：

```bash
tail -f /Users/hmax/qlibAssistant/.qlibAssistant/logs/rolling_folds_manual.log
```

脚本当前固定使用 CSI1000、Linear、60个月窗口和三个严格不重叠的 Test 时段。要修改模型、股票池或日期，编辑 `script/run_rolling_folds.py` 顶部的 `FOLDS` 和训练命令参数。

也可以手动运行任意历史截止日：

```bash
$PY script/run.py \
  --pool csi1000 \
  --run-tag my_fold_260719 \
  --models Linear \
  --window-months 60 \
  --end-date 2026-02-16 \
  --split-selection-valid
```

### 12.5 创建可读的实验文件夹和 CSV 索引

MLflow 原始目录以数字 experiment ID 命名，不便浏览。运行：

```bash
$PY script/build_experiment_index.py
```

生成：

- `.qlibAssistant/experiment_index.csv`：实验名、ID、模型、股票池、run-tag、recorder 数量和原始路径。
- `.qlibAssistant/mlruns_by_name/`：以完整实验名称命名的软链接；不会复制或删除模型文件，不额外占用一份模型空间。

训练批次结束后 `script/run.py` 会自动刷新索引，一般不需要手动执行。

#### MLflow 两级目录分别是什么

本项目的原始目录是：

```text
.qlibAssistant/mlruns/
└── <experiment_id>/                 # 第一级：实验/训练批次
    └── <recorder_id>/               # 第二级：该实验中的一次具体训练运行
        ├── artifacts/
        ├── metrics/
        ├── params/
        └── meta.yaml
```

- `experiment_id` 是 MLflow 自动生成的数字 ID。同一个“模型算法 + 股票池 + run-tag + 启动小时”对应一个 experiment；它只是数据库主键，本身没有业务含义。
- `recorder_id` 是一次具体训练运行的随机十六进制 ID，绑定一个模型、一个时间窗口、拟合参数、预测结果和指标。custom 五窗口训练通常在一个 experiment 下面产生5个 recorder；指定 `--window-months 60` 时通常只有1个。
- 真正适合人识别的是 experiment 的 `name`，例如 `EXP_LinearModel_Alpha158_csi1000_custom_step0_linear_raw_fold3_260719_20260719_23`。它依次表达：模型、因子集、股票池、滚动方式、run-tag、启动日期与小时。
- recorder 的时间窗口要查看其 `artifacts/task`，或者优先看自动导出的 `summary.md`、`metrics.csv` 和 `.qlibAssistant/experiment_index.csv`，不能从 recorder ID 本身推断。

例如：

```text
781864565834297631/efb3e64a5db74ed2ad7d40040624cbea
```

前者是 raw-label Fold3 的 experiment ID，后者是该 Fold 中 Linear 60个月窗口的 recorder ID。为了避免记数字，日常应从 `.qlibAssistant/mlruns_by_name/<完整实验名>/` 进入。

### 12.6 把 pkl 指标导出成可读文件

PKL 不是 MLflow 强制要求的格式，而是 Qlib 用来无损保存 DataFrame、Series、模型和配置对象的默认序列化格式。现有代码会通过 `load_object("...pkl")` 读取这些文件，因此不能把 PKL 直接删除或仅改成 CSV。

从 2026-07-19 起，新 recorder 会对适合表格化的数据对象双写：

```text
artifacts/
├── pred.pkl / pred.csv                 # Test 预测
├── label.pkl / label.csv               # Test 真实标签
├── sig_analysis/
│   ├── ic.pkl、ric.pkl / daily_ic.csv
│   └── metrics.csv
└── valid_sig_analysis/
    ├── pred.pkl / pred.csv
    ├── label.pkl / label.csv
    ├── ic.pkl、ric.pkl / daily_ic.csv
    ├── metrics.pkl / metrics.csv
    └── segment.pkl / segment.csv
```

PKL 供程序继续读取，CSV 供人工查看。`params.pkl` 是已拟合模型，无法等价表示成 CSV；`task` 和 `dataset` 是嵌套配置对象，也不属于二维表格，继续保留原格式并通过 `summary.md`/索引查看摘要。已有 recorder 可原地补充 CSV：

```bash
$PY script/export_validation_csv_inplace.py
```

需要重新覆盖已有 CSV 时添加 `--overwrite`。2026-07-19 已为101个 recorder 回填 validation CSV，并为102个 recorder 回填顶层 Test 与 `sig_analysis` CSV。

按 run-tag 导出：

```bash
$PY script/export_readable_artifacts.py \
  --experiment-pattern 'splitval260719' \
  --topn 20
```

如果需要导出 Test 全量股票预测：

```bash
$PY script/export_readable_artifacts.py \
  --experiment-pattern 'splitval260719' \
  --topn 20 \
  --full-predictions
```

输出目录：

```text
.qlibAssistant/readable_artifacts/
└── <experiment_name>/
    └── <recorder_id>/
        ├── summary.md
        ├── metrics.csv
        ├── validation_daily_ic.csv
        ├── test_daily_ic.csv
        ├── test_top20_predictions.csv
        └── test_all_predictions.csv  # 仅 --full-predictions
```

根目录的 `export_index.csv` 可用于一次查看所有导出 recorder 的 validation 指标。

### 12.7 生成统一 Test 报告

按实验名称的一部分筛选批次：

```bash
$PY script/evaluate_batch.py \
  --experiment-pattern csi1000_custom_step0_retrain260718 \
  --topk 10
```

只集成 Selection-validation Rank ICIR 最高的6个：

```bash
$PY script/evaluate_batch.py \
  --experiment-pattern csi1000_custom_step0_retrain260718 \
  --top-models 6 \
  --topk 10
```

限制相同 Test 区间，以便不同方案公平比较：

```bash
$PY script/evaluate_batch.py \
  --experiment-pattern csi1000_custom_step0_retrain260718 \
  --top-models 6 \
  --test-start 2026-06-22 \
  --test-end 2026-07-17 \
  --topk 10
```

报告生成到 `.qlibAssistant/analysis/evaluation_<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `summary.csv` | Rank IC、Rank ICIR、胜率和累计收益等汇总 |
| `daily_metrics.csv` | 每个 Test 交易日的 IC、Rank IC、TopK 收益和累计曲线 |
| `recorders.csv` | 候选 recorder、各段日期、Validation/Test 指标和集成权重 |
| `ensemble_test_predictions.csv` | 每个股票日的集成分数与真实标签 |
| `test_report.png` | 累计收益与逐日 Rank IC 图 |

报告中的收益为未扣手续费、滑点和冲击成本的研究指标，不能直接等同于实盘收益。

对于3日或5日标签，必须传入持有周期，使用H组错峰组合避免重叠收益被每天重复复利：

```bash
$PY script/evaluate_batch.py \
  --experiment-pattern linear_h3_fold1_260719 \
  --threshold -999 \
  --holding-period 3 \
  --topk 10
```

`summary.csv` 中以 `executable_*` 开头的字段是错峰组合结果；旧的 `gross_*` 字段是逐日标签直接复利，只适合1日标签，不应用于跨持有周期比较。

### 绝对收益标签与截面标准化标签

Alpha158 默认对训练标签执行 `CSZScoreNorm`：每天在股票池截面内减均值、除标准差。因此默认模型的 `score` 主要用于当天股票间排序，不应直接解释为“预计上涨百分比”。

实验时可取消标签截面标准化：

```bash
MLFLOW_ALLOW_FILE_STORE=true /Users/hmax/miniconda3/envs/qlibAssistant/bin/python script/run.py \
  --pool csi1000 --run-tag linear_raw_label_YYMMDD --models Linear \
  --split-selection-valid --window-months 60 --end-date YYYY-MM-DD \
  --label-horizon 1 --raw-label
```

`--raw-label` 不修改收益公式、特征处理或 Ridge 正则化，只把 label 的学习处理器改为 `DropnaLabel`。此时 score 具有收益率单位，但仍必须检查校准，不能仅凭单位就把它当成准确收益预报。

统一评估会额外生成：

- `summary.csv`：`absolute_MAE`、`absolute_RMSE`、预测偏差、预测/实际均值与标准差；
- `absolute_return_calibration.csv`：按预测 score 十分位统计预测均值、实际均值和上涨胜率；
- `ensemble_test_predictions.csv`：逐股票、逐交易日的 score 与实际收益。

2026-07-19 的 CSI1000 Linear 三折实验中，raw label 的 Rank IC 为 `0.0044、-0.0018、-0.0066`，明显低于默认标准化标签的 `0.0391、0.0340、0.0263`；因此当前生产默认值仍保留截面标准化。绝对收益模式只作为研究开关。

### 12.8 画所有 recorder 的 IC/ICIR 性能图

```bash
$PY script/plot_model_performance.py \
  --experiment-pattern 'csi1000.*retrain260718' \
  --output-prefix csi1000_retrain260718_performance
```

输出：

- `.qlibAssistant/analysis/csi1000_retrain260718_performance.csv`
- `.qlibAssistant/analysis/csi1000_retrain260718_performance.png`

图片分为 Validation 和 Test 两个面板：

- 横轴：Rank IC，越向右越好。
- 纵轴：Rank ICIR，越向上越稳定。
- 颜色：模型类型。
- 点旁数字：训练总窗口月数。
- 右上象限：该数据段内排序方向正确且相对稳定。

不能根据 Test 面板反复挑选模型，否则会产生 Test 泄漏。正式选择只能看 Validation/Selection-validation，Test 面板用于一次性验收。

### 12.9 用 Top-N recorder 生成最新股票排名

例如使用 validation Rank ICIR 前6名预测 CSI1000：

```bash
cd /Users/hmax/qlibAssistant/roll
$PY roll.py \
  --stock_pool=csi1000 \
  --model_filter='[".*csi1000.*retrain260718.*"]' \
  --top_models=6 \
  model selection
```

输出位于 `.qlibAssistant/analysis/selection_<时间戳>/`。股票级 CSV 开头字段依次为：

```text
instrument, code, name, datetime, rank, avg_score, pos_ratio, ...
```

- `<date>_ret.csv`：全量模型排序。
- `<date>_filter_ret.csv`：经过现有 STD/ROC 规则过滤后的子集。
- `total.csv`：逐 recorder 原始预测，行数约为股票数乘参与模型数。
- `total.md`：参与模型、Validation/Test 指标和权重。

### 12.10 推荐的一轮研究顺序

```text
1. 设定明确 run-tag
2. 先用 Linear/LightGBM + 60个月窗口做低成本实验
3. 用独立 selection_valid 选模
4. 至少运行3个不重叠时间 Folds
5. 比较 Rank IC、Rank ICIR、正比例、Top1/Top3/Top10 与超额收益
6. 只有跨 Fold 稳定后，才扩展模型、窗口或股票池
7. 最终锁定配置后再运行最新预测
```

不要一开始就混合大量模型。更多 recorder 只有在误差具有互补性时才可能改善结果；弱模型、重复模型和高度相关模型会稀释有效信号。

### 12.10 选择性交易（允许空仓）实验

`script/evaluate_selective_trading.py` 使用已有 Linear + LightGBM recorder，不重新训练模型。阈值只根据 `selection_valid` 的分布确定，然后原样应用到 Test，避免用 Test 收益挑阈值。

当前提供三类简单条件：

- `strength`：当日 Top3 集成标准分的平均强度；
- `agreement`：Linear 与 LightGBM 对 Top3 的预测一致程度；
- `persistence`：当日 Top3 与前一交易日 Top3 的重合比例。

示例（Fold1）：

```bash
$PY script/evaluate_selective_trading.py \
  --experiment-ids 666244200422019204 124235235455944455 \
  --fold-name fold1 --topk 3 --cost-rate 0.0015
```

`cost-rate=0.0015` 表示每个交易日按完整换仓扣除0.15%，属于偏保守的简化成本；当前没有按照实际持仓重合比例精算换手率。

2026-07-20 三折初步结果：

| 规则 | 平均交易覆盖率 | 三折平均净累计 | 最差 Fold | 平均最大回撤 |
|---|---:|---:|---:|---:|
| 始终交易 | 100.0% | 12.5% | -18.3% | -20.7% |
| strength Q70 | 31.3% | 4.9% | -4.0% | -15.1% |
| persistence Q70 | 53.8% | 3.9% | -6.7% | -12.2% |
| agreement Q70 | 34.1% | -7.4% | -18.3% | -15.6% |

- 强市场 Fold1/2 中，空仓规则通常牺牲总收益；弱市场 Fold3 中，`persistence Q70` 将净累计从约 -18.3% 改善到 +13.7%，最大回撤从 -31.3% 降到 -8.2%。
- `strength Q70` 的跨 Fold 最差结果较小，适合继续作为简易“没有足够强信号则空仓”的候选。
- 单独使用模型一致性没有稳定改善；多个条件同时叠加会造成交易样本过少。
- 目前只有三个时间 Fold，且交易成本和换手率仍是简化模拟，不应直接视为实盘规则。

#### 指标含义和继续实验

- `Q70`：在 selection-validation 中取某个信号指标的第70百分位作为门槛。例如 Top3 persistence 的门槛为1/3时，表示今天Top3至少有1只也在昨天Top3才允许交易；不是在 Test 中挑收益最好的70%。
- `净累计收益`：交易日先用 TopK 真实收益减去0.15%简化成本，空仓日收益记0，再按时间执行复利；不是逐日收益的简单相加。
- `三折平均净累计`：三个 Fold 各自终值收益的算术平均，用来快速比较规则；它不是年化收益，也不是把三个时段串接后的真实总收益。
- `最差 Fold`：同一规则在三个独立历史时期中最小的终值收益，衡量市场环境变化时可能出现的最坏阶段；它不同于 Fold 内最大回撤。

Fold3 Top3 Persistence Q70 的具体计算：97个测试日中触发52日，触发日毛收益算术和约22.0%，成本为 `52 × 0.15% = 7.8%`，按每日净收益复利后为13.7255%；交易胜率53.85%，最大回撤-8.16%。

继续比较 Top1/Top3/Top5 后，Top1结果高度不稳定；Top3的 `strength AND persistence Q70` 三折分别约为 -1.1%、+15.3%、+16.3%，交易日为23、9、26日。三个 Fold 串接复利约32.6%，接近始终交易约33.3%，但样本很少，尤其 Fold2 只有9次触发。

多日标签可添加：

```bash
$PY script/evaluate_selective_trading.py \
  --experiment-ids <3日或5日实验ID> --fold-name my_h3_fold \
  --topk 3 --holding-period 3 --cost-rate 0.0015
```

代码使用H组错峰资金组合，避免3/5日收益标签重叠时被逐日重复复利。初步结果中，3日规则近期仍为负；5日 Persistence Q70 三折约为+0.5%、+8.1%、+3.6%，5日 Strength AND Persistence Q70 为-3.3%、+2.0%、+6.1%。当前仍以1日 Top3 规则作为主要研究方向，5日 persistence 作为持仓更久的备选。

#### 5日 Top3 Persistence Q70 的完整持仓语义

当前报告模拟的是5组等资金错峰组合，而不是每天用全部资金重新买一次：

1. 每天收盘后，5日收益模型用截至当天的数据为 CSI1000 排名，取当日 Top3。
2. 比较当日 Top3 与前一交易日 Top3。三个 Fold 的 selection-validation Q70 门槛均为 `2/3`，即至少2只股票连续留在Top3才触发。
3. 总资金理论上分成5份；每天只有一份资金到期并重新决策。
4. 触发时，该份资金等权买入当日Top3；未触发时，该份资金未来5个交易日保持现金。
5. 触发的子组合从下一交易日收盘价开始计收益，持有5个交易日，在第6个交易日收盘退出，对应标签 `Ref($close,-6)/Ref($close,-1)-1`。
6. 到期后该份资金重新进入相同判断；五份资金交错运行，因此最多同时存在5个不同开仓日的持仓批次。
7. 每个触发批次扣除0.15%简化往返成本；当前尚未按股票重合精确计算换手率，也未模拟涨跌停和成交失败。

以10万元直接照搬时，每份约2万元、再分3只约每只6667元，可能受到100股整数手和高价股限制。因此这套5组错峰规则目前是研究口径；个人实盘可进一步比较“每5天只运行一个完整组合”或减少到Top1/Top2，但需要重新回测，不能直接套用本报告收益。

报告目录：`selective_fold1_20260720_00_06_06`、`selective_fold2_20260720_00_06_07`、`selective_fold3_20260720_00_06_08`。

### 12.11 LightGBM 超参数在哪里调整

`lambda_l1`、`lambda_l2`、`num_leaves`、`max_depth`、`min_data_in_leaf` 都是 **LightGBM** 参数，不是 Linear 的参数：

- `lambda_l1/lambda_l2`：叶子权重的 L1/L2 正则，越大越保守；
- `num_leaves/max_depth`：树容量，越大越容易拟合复杂关系，也越容易过拟合；
- `min_data_in_leaf`：每个叶子的最少样本数，增大可抑制小样本叶子；
- `learning_rate`：每棵树的更新步长；
- `early_stopping_rounds`：Valid 指标连续多少轮不改善后停止。

原始默认参数仍在 `roll/myconfig.py` 的 `GBDT_MODEL`。日常实验建议编辑更安全的预设文件：

```text
roll/model_params.yaml
```

调用示例：

```bash
$PY script/run.py \
  --pool csi1000 --run-tag my_lgb_test_YYMMDD --models LightGBM \
  --split-selection-valid --window-months 60 --end-date YYYY-MM-DD \
  --model-preset low_regularization_v1
```

只要在 `model_params.yaml` 的 `LightGBM:` 下复制一个预设、改名并修改数字，就可以通过 `--model-preset 新名称` 运行。务必修改 `run-tag`，避免与旧实验混淆。

2026-07-20 对照：默认 `lambda_l1/lambda_l2=205.7/581.0`；`low_regularization_v1=0.5/5.0`，并使用 `max_depth=6、num_leaves=63、min_data_in_leaf=200`。

| Fold | 默认 Rank IC | Low-reg Rank IC | 默认 Top10累计 | Low-reg Top10累计 |
|---|---:|---:|---:|---:|
| 1 | 0.0438 | 0.0438 | 50.4% | 63.7% |
| 2 | 0.0372 | 0.0378 | 51.0% | 25.3% |
| 3 | 0.0103 | 0.0081 | -23.1% | -0.1% |

Low-reg 改善了 Fold1 和最差的 Fold3，但明显损害 Fold2 收益，Rank IC 没有全面提高。因此它是值得继续验证的候选，而不是新的默认参数。优化时应同时看多 Fold Rank IC、TopK 净收益和最差回撤，不能只追求 Train loss 降得更多。

### 12.12 固定三折模型比较教程（推荐入口）

正式实验使用四段数据，且所有模型与训练窗口共用完全相同的 Valid、Selection-valid 和 Test。`--train-months` 只改变 Train 起点，不移动后三段：

```bash
cd /Users/hmax/qlibAssistant
PY=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python
export MLFLOW_ALLOW_FILE_STORE=true

# 先打印三折日期与命令，不训练
$PY script/run_fixed_folds.py \
  --models Linear LightGBM XGBoost CatBoost \
  --train-months 45 --pool csi1000 \
  --tag-prefix my_model_test --date-tag 260720 --dry-run

# 确认后正式训练；会依次训练三个 Fold
$PY script/run_fixed_folds.py \
  --models Linear LightGBM XGBoost CatBoost \
  --train-months 45 --pool csi1000 \
  --tag-prefix my_model_test --date-tag 260720
```

当前快速模型候选为 Linear、LightGBM、XGBoost、CatBoost；默认批处理已排除 DoubleEnsemble。单个模型单个 Fold 在本机本轮约2～3分钟，其中大量时间用于加载 Alpha158 数据和生成 recorder，并非纯模型拟合时间。

评估某个精确实验名：

```bash
$PY script/evaluate_batch.py \
  --experiment-pattern '<完整实验名>' --exact-experiment-name \
  --weighting equal --threshold 0.001 --topk 10
```

报告目录位于 `.qlibAssistant/analysis/evaluation_时间/`：

- `summary.csv`：Rank IC/ICIR、TopK收益、官方中证1000与沪深300收益及超额；
- `daily_metrics.csv`：逐日指标和净值；
- `ensemble_test_predictions.csv`：逐股票、逐日预测与标签；
- `recorders.csv`：参与评估的 recorder、数据区间及验证指标；
- `test_report.png`：净值和每日 Rank IC 图。

评估脚本会同时写出Top1、Top3、Top5、Top10的胜率、累计收益、同期中证1000与沪深300累计收益，并提供两种相对指标：`return_diff` 是策略收益减指数收益的百分点差值；`excess` 是 `(1+策略累计)/(1+指数累计)-1` 的净值比率超额。

每行汇总还包含 `model`、`train_months`、`train_start`、`train_end`、`test_start`、`test_end`，用于脱离实验文件夹名称识别模型和数据窗口。中证1000/沪深300累计收益不是数据源预先给出的区间统计：程序从Qlib中的SH000852/SH000300每日收盘价出发，按模型标签相同的交易时点计算每日收益，再用 `(1+r).cumprod()-1` 复利得到测试区间累计收益。

多个报告可用 `script/compare_evaluation_reports.py` 汇总。每个 `--report` 的格式是 `模型名,Fold名,报告目录`，脚本会输出逐 Fold 表、模型均值、最差 Fold 和比较图。第一轮示例结果在：

```text
.qlibAssistant/analysis/model_selection_round1_20260720/
```

注意：当前 Top10 收益是无手续费、无滑点、允许碎股的等资金权重信号实验。它适合比较模型，不是10万元账户可直接执行的回测。A股实际等资金配置应按每只股票目标资金除以价格，再向下取整到100股；余款留作现金，因此实际权重只能近似相等。

#### 12.12.1 “只改Train，其他区间不变”是如何实现的

正式的模型/时间窗口对照实验应调用 `script/run_fixed_folds.py`，而不是直接使用 `script/run.py --window-months`。实际调用链是：

```text
run_fixed_folds.py --train-months N
    ↓ 计算四段精确日期
run.py --segments-json '{train,valid,selection_valid,test}'
    ↓ 转为 roll.py 参数
roll.py --fixed_segments=...
    ↓
TrainCLI 覆盖 Dataset segments 和 handler 起止日期
```

`script/run_fixed_folds.py` 的 `FOLDS` 常量明确保存三个Fold的后三段：

| Fold | Valid | Selection-valid | Test |
|---|---|---|---|
| Fold1 | 2024-06-15～2024-11-14 | 2024-11-15～2025-04-14 | 2025-04-15～2025-09-15 |
| Fold2 | 2024-11-16～2025-04-15 | 2025-04-16～2025-09-15 | 2025-09-16～2026-02-16 |
| Fold3 | 2025-04-17～2025-09-16 | 2025-09-17～2026-02-16 | 2026-02-17～2026-07-17 |

对于每个Fold，脚本只做两件事：

```python
train_start = valid_start - relativedelta(months=train_months)
train_end = valid_start - timedelta(days=1)
```

因此将 `--train-months 60` 改为 `120` 或 `240` 时，同一Fold的Valid、Selection-valid、Test完全不动，只将Train起点向历史方向扩展。“Test不变”指的是 **不同模型和不同Train月份在同一Fold内使用相同Test**；Fold1、Fold2、Fold3本身是三个不同时期，并不共用一个Test日期。

先用 `--dry-run` 查看实际四段日期，不进行训练：

```bash
$PY script/run_fixed_folds.py \
  --models XGBoost \
  --train-months 240 \
  --pool csi1000 \
  --tag-prefix windowcmp \
  --date-tag YYMMDD \
  --dry-run
```

终端会为每个Fold打印：

```text
[fixed-fold] foldN segments={'train': (...), 'valid': (...), 'selection_valid': (...), 'test': (...)}
```

确认日期后删除 `--dry-run` 正式训练。

`script/run.py --window-months` 属于普通custom窗口入口，会由滚动生成逻辑分配时间段；它适合一般训练，不应用来声称“只改Train的严格公平对照”。如果必须直接调用 `script/run.py`，应显式传入包含全部四段的 `--segments-json`。

### 12.13 第一轮四模型结论（2026-07-20）

固定45个月 Train、固定三折后三段、Alpha158、CSI1000、1日标签、Top10毛收益：

| 模型 | 三折平均 Rank IC | 平均 Rank ICIR | 平均Top10累计 | 平均超额/中证1000 | 最差Fold超额/中证1000 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.0321 | 0.2822 | 38.3% | 20.8% | +3.3% |
| Linear | 0.0332 | 0.2863 | 13.7% | -0.5% | -16.5% |
| LightGBM | 0.0305 | 0.2785 | 26.1% | 9.9% | -21.9% |
| CatBoost | 0.0338 | 0.2625 | 19.5% | 3.1% | -29.5% |

第一轮推荐保留 **XGBoost + Linear + LightGBM** 进入下一步：XGBoost近期最稳且收益表现最好；Linear是稳定、快速、低复杂度基线；LightGBM便于做超参数搜索。CatBoost并非永久淘汰，但它在Fold3退化最大，先降为备选。下一步先对XGBoost和LightGBM做小规模超参搜索，再比较12/24/36/45/60个月Train；不能根据这张Test表反复挑参数，参数排序必须主要依据Selection-valid，Test只用于封存验证。

### 12.11 当前项目下一步

截至2026-07-19：

1. 已完成 CSI1000 五类模型、25个 canonical recorder 重训。
2. 已实现 Validation 选模、Top-N 集成、重复 recorder 去重和统一 Test 报告。
3. 已完成 Linear 60个月窗口的三个非重叠 Test Folds，三个 Fold Rank IC/Rank ICIR 均为正，但近期 Fold3 收益转弱。
4. 下一步在完全相同的三个 Fold 上运行 LightGBM 60个月模型。
5. 对比 Linear、LightGBM 和二者等权/Validation加权集成。
6. 随后比较1日、3日、5日收益标签，选择更适合个人数日持仓的预测周期。
7. 最后修复 Linear 数值溢出、评估交易阈值和费用，再决定是否扩展全市场训练及实盘。

## 13. XGBoost-240 每日更新、推理与验证完整教程

本节是2026-07-20后的推荐自助入口。目标是：收盘后更新Tushare日线数据，只加载已训练的CSI1000 XGBoost-240 Fold3，生成最新排名，并在未来收盘价到齐后验证历史信号。

### 13.1 环境与固定变量

```bash
cd /Users/hmax/qlibAssistant
PY=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python
export MLFLOW_ALLOW_FILE_STORE=true
export MPLCONFIGDIR=/Users/hmax/qlibAssistant/.qlibAssistant/matplotlib
```

日常更新和预测都使用 `qlibAssistant` 环境。`qlib_env` 只由 `update-predict.py` 内部调用，用来转换Qlib二进制。Tushare token从 `~/.config/tushare_token` 读取，不应写入Git文件。

### 13.2 XGBoost-240 的训练设计和存储位置

训练条件：CSI1000、Alpha158、1日标签、240个月计划Train窗口，三个固定Fold，Valid用于早停，Selection-valid用于选模，Test只用于最终比较。

| Fold | Experiment ID | Recorder ID | 主要用途 |
|---|---|---|---|
| Fold1 | `215069146775515865` | `89f3ed736bf1497e86aa074bc074581e` | 历史Test 1 |
| Fold2 | `174018209104878076` | `308d37421ddf4c018896de2f512f7d0e` | 历史Test 2 |
| Fold3 | `473901139733640553` | `021853d841234c4fb397e0e919800b58` | 最新数据推理与Test 3 |

Fold3实际目录：

```text
/Users/hmax/qlibAssistant/.qlibAssistant/mlruns/473901139733640553/021853d841234c4fb397e0e919800b58/
```

关键文件：

- `artifacts/params.pkl`：训练好的XGBoost模型；
- `artifacts/task`：数据集、Alpha158、日期切分配置；
- `artifacts/pred.pkl`：原Test区间预测；
- `artifacts/label.pkl`：原Test区间标签；
- `artifacts/valid_sig_analysis/`：Selection-valid选模指标；
- `artifacts/sig_analysis/`：Test信号指标。

如需重新训练同样的三Fold：

```bash
$PY script/run_fixed_folds.py \
  --models XGBoost \
  --train-months 240 \
  --pool csi1000 \
  --tag-prefix windowcmp \
  --date-tag YYMMDD
```

训练会新建三个Experiment和三个Recorder；不会覆盖上表旧结果。可用 `.qlibAssistant/experiment_index.csv` 或 `.qlibAssistant/mlruns_by_name/` 查找新ID。

### 13.3 每日收盘后更新数据

建议在17:00～18:00之后运行，以免Tushare当日数据尚未发布。

```bash
$PY update-predict.py --update-only
```

该命令会自动：

1. 确认 `/Users/hmax/qlib` 和 `/Users/hmax/investment_data` 都在 `local/hmax-fixes` 分支；
2. 检查Dolt本地最新交易日；
3. 通过Tushare代理下载缺失日线；
4. 写入Dolt、转换Qlib二进制；
5. 替换 `~/.qlib/qlib_data/cn_data/`；
6. 重建CSI300/500/800/1000成分股文件；
7. 不运行旧的全模型selection。

完成后验证：

```bash
tail -n 3 ~/.qlib/qlib_data/cn_data/calendars/day.txt
ls ~/.qlib/qlib_data/cn_data/instruments/csi1000.txt
```

完整更新日志位于 `/tmp/update_predict.log`。全量转换通常需要10～20分钟，长时间无终端输出时可用 `tail -f /tmp/update_predict.log` 查看。

### 13.4 用XGBoost-240推理最新日期

假设Qlib已更新到 `YYYY-MM-DD`：

```bash
$PY script/predict_recorder_date.py \
  --experiment-id 473901139733640553 \
  --recorder-id 021853d841234c4fb397e0e919800b58 \
  --date YYYY-MM-DD \
  --topk 10
```

脚本会同时修改Test segment和保存在Recorder中的handler `end_time`，这一点很重要：只改Test segment会得到空预测。240个月handler需重新加载长历史Alpha158，本机一次推理约3分钟，属正常现象。

输出目录：

```text
.qlibAssistant/analysis/recorder_prediction_YYYYMMDD_HHMMSS/ranking.csv
```

CSV包含 `rank,instrument,name,datetime,score`。`score` 是模型排序分数，不是可直接解读为百分比的预测收益。

### 13.5 交易日期口径

当信号日为T时，当前1日标签是：

```text
Ref($close,-2) / Ref($close,-1) - 1
```

含义是T日收盘后产生排名，T+1交易日收盘附近理论买入，T+2交易日收盘附近理论卖出。例如：

- 2026-07-20信号；
- 2026-07-21收盘附近买；
- 2026-07-22收盘附近卖。

不能把此结果直接解读为“次日开盘买入”。当前交易策略网格存在过拟合，所以TopK是模型候选排名，不是自动实盘指令。

### 13.6 用新收盘价验证旧信号

只有T+2收盘价已经入库，T日信号才有完整标签。验证命令：

```bash
$PY script/validate_recorder_signal.py \
  --experiment-id 473901139733640553 \
  --recorder-id 021853d841234c4fb397e0e919800b58 \
  --signal-date YYYY-MM-DD
```

输出包括：

- 当日横截面Rank IC；
- Top1/Top3/Top10平均收益和胜率；
- CSI1000股票平均收益；
- 当日Top10的实际标签。

如果T+2尚未收盘，只能检查T到T+1的盘面变化，这不是模型正式标签。临时观察可指定：

```bash
$PY script/validate_recorder_signal.py \
  --experiment-id 473901139733640553 \
  --recorder-id 021853d841234c4fb397e0e919800b58 \
  --signal-date YYYY-MM-DD \
  --label-expression 'Ref($close,-1)/$close-1'
```

### 13.7 XGBoost-240 Fold3已知指标

Fold3 Test大约为2026-02-24～2026-07-17，99个信号日。

| 指标 | 结果 |
|---|---:|
| Top1上涨胜率 | 46.39% |
| Top3组合上涨胜率 | 53.61% |
| Top5组合上涨胜率 | 52.58% |
| Top10组合上涨胜率 | 48.45% |
| 每日Rank IC为正比例 | 59.79% |
| Top3扣0.15%换手成本后净累计 | Fold平均68.89% |

“胜率不高但累计收益高”表示历史上盈利日的幅度大于亏损日，不代表每天Top1可靠上涨。这些窗口是查看过Test后挑出的，必须用2026-07-20之后的新数据做前向记录，不应再利用同一Test反复改参数。

### 13.8 2026-07-20实际示例

7月20日Tushare下载5524条日线，Qlib日历成功更新到7月20日。XGBoost-240 Fold3 Top10：

| Rank | 股票 |
|---:|---|
| 1 | 长芯博创 SZ300548 |
| 2 | 德科立 SH688205 |
| 3 | 安泰科技 SZ000969 |
| 4 | 西部材料 SZ002149 |
| 5 | 东威科技 SH688700 |
| 6 | 大金重工 SZ002487 |
| 7 | 三祥新材 SH603663 |
| 8 | 华曙高科 SH688433 |
| 9 | 高新发展 SZ000628 |
| 10 | 福晶科技 SZ002222 |

输出：`.qlibAssistant/analysis/recorder_prediction_20260720_180957/ranking.csv`。

7月16日信号在7月20日已可完整验证：当日Rank IC=-0.2142，Top1=+8.71%，Top3平均=+1.68%，Top10平均=-3.59%。这是“Top1命中，整体排序失败”的典型反例，不能只截取Top1盈利宣称模型当日成功。

### 13.9 最短日常操作清单

```bash
cd /Users/hmax/qlibAssistant
PY=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python
export MLFLOW_ALLOW_FILE_STORE=true
export MPLCONFIGDIR=/Users/hmax/qlibAssistant/.qlibAssistant/matplotlib

# 1. 收盘后更新；无新数据时等待1～2小时重试
$PY update-predict.py --update-only

# 2. 把日期换成Qlib日历最后一天
$PY script/predict_recorder_date.py \
  --experiment-id 473901139733640553 \
  --recorder-id 021853d841234c4fb397e0e919800b58 \
  --date YYYY-MM-DD --topk 10

# 3. 验证两个交易日前的信号
$PY script/validate_recorder_signal.py \
  --experiment-id 473901139733640553 \
  --recorder-id 021853d841234c4fb397e0e919800b58 \
  --signal-date YYYY-MM-DD
```

日常记录至少保存：信号日、Top10、Top1/3/10后续实际收益、当日Rank IC、CSI1000/CSI300同期收益、是否停牌/涨跌停/无法成交。

---

## 14. 预测文件、回测参数与评估字段完整说明

### 14.1 单模型和固定集成预测的输出

两个预测脚本现在每次都会同时生成完整股票池和剔除科创板两个版本，不再要求必须传入 `--exclude-star-market`：

```text
recorder_prediction_YYYYMMDD_HHMMSS/
├── ranking_all.csv              # 完整股票池
├── ranking_ex_star.csv          # 自动剔除SH688/SH689
├── ranking.csv                  # 兼容旧脚本的入口
├── prediction_config.json       # 机器可读运行配置
└── README.md                    # 人工可读说明

fixed_ensemble_YYYYMMDD_HHMMSS/
├── ensemble_ranking_all.csv
├── ensemble_ranking_ex_star.csv
├── ensemble_ranking.csv         # 兼容旧脚本的入口
├── prediction_config.json
└── README.md
```

默认情况下兼容入口 `ranking.csv`/`ensemble_ranking.csv` 保存完整股票池；如果仍传入 `--exclude-star-market`，只会让兼容入口和终端TopK改为剔除科创板版本。无论是否传参，`*_all.csv` 和 `*_ex_star.csv` 都会生成。

### 14.2 macOS永久环境变量

当前Mac默认Shell是zsh，对应Linux bash的 `~/.bashrc` 文件是：

```text
~/.zshrc
```

只需执行一次：

```bash
echo 'export PY=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python' >> ~/.zshrc
echo 'export MLFLOW_ALLOW_FILE_STORE=true' >> ~/.zshrc
echo 'export MPLCONFIGDIR=/Users/hmax/qlibAssistant/.qlibAssistant/matplotlib' >> ~/.zshrc
source ~/.zshrc
```

以后新开终端即可直接使用 `$PY`。可以这样检查：

```bash
echo $PY
$PY --version
```

`MLFLOW_ALLOW_FILE_STORE=true` 只是当前项目仍使用旧FileStore时的兼容开关；以后迁移SQLite后应删除这一项。若不希望污染所有终端，也可以只在项目专用启动脚本中设置。

### 14.3 `evaluate_batch.py`参数解释

典型命令：

```bash
$PY script/evaluate_batch.py \
  --experiment-pattern 'EXP_XGBModel_Alpha158_csi1000_custom_step0_windowcmp_fold3_train240m_260720_20260720_16' \
  --exact-experiment-name \
  --model-class XGBModel \
  --topk 3 \
  --weighting equal \
  --cost-rate 0.0015
```

| 参数 | 是否必需 | 默认值 | 解释 |
|---|---:|---:|---|
| `--experiment-pattern` | 是 | 无 | 用来匹配MLflow Experiment名称。默认是“包含匹配”，因此短字符串可能匹配多个实验 |
| `--exact-experiment-name` | 否 | 关闭 | 开启后要求实验名称完全一致。评估单个正式实验时建议始终开启，避免误匹配其他fold或h3/h5实验 |
| `--model-class` | 否 | 不限制 | 只保留指定模型类，例如 `XGBModel`、`LGBModel`、`CatBoostModel`、`LinearModel` |
| `--threshold` | 否 | `0.001` | Validation选模门槛。Validation的IC、ICIR、Rank IC、Rank ICIR必须全部不低于该值。单recorder诊断时若不想过滤，可设很低的值 |
| `--topk` | 否 | `10` | 主策略每天持有预测排名前K只股票；决定主图的 `Executable TopK Equity`。报告仍固定附带Top1/3/5/10指标 |
| `--cost-rate` | 否 | `0.0015` | 完整换仓时的简化综合成本，0.0015即0.15%。实际扣费=`换手率 × cost_rate` |
| `--test-start` | 否 | recorder共同Test起点 | 手工限制评估开始日期，只会在原Test范围内截取 |
| `--test-end` | 否 | recorder共同Test终点 | 手工限制评估结束日期 |
| `--holding-period` | 否 | `1` | 持仓交易日数。大于1时使用H组错峰资金模拟，避免把重叠的多日标签每天完整复利 |
| `--weighting` | 否 | `validation_rank_icir` | 多recorder集成权重。`validation_rank_icir`按Validation Rank ICIR加权；`equal`等权。单recorder建议使用 `equal` |
| `--top-models` | 否 | 全部通过者 | 通过Validation门槛后，只保留Rank ICIR最高的前N个recorder |

共同Test区间指所有入选recorder都拥有预测值的日期交集。这样比较不同模型时不会因为某个模型多预测了几天而获得不公平优势。

### 14.4 常见金融词汇

| 词汇 | 本项目中的含义 |
|---|---|
| Gross / 毛收益 | 尚未扣手续费、印花税、滑点和冲击成本的理论收益 |
| Net / 净收益 | 毛收益减去简化交易成本。本项目按每日TopK实际换手比例扣 `cost_rate` |
| Equity / 净值 | 假设初始资金为1，收益逐期复利后的资金曲线。净值1.50表示累计赚50% |
| Cumulative Return / 累计收益 | `当前净值 - 1`。净值1.50对应累计收益0.50，即50% |
| Executable / 可执行口径 | 针对多日持仓，用H组等资金错峰开仓，避免同一笔资金同时被重复投入多个重叠标签。它仍是研究级模拟，不代表一定成交 |
| Turnover / 换手率 | 今天持仓与上一交易日持仓发生替换的比例。Top3替换1只，换手率约1/3 |
| Universe | 当天测试股票池全部股票的等权平均，不是CSI1000指数本身 |
| Benchmark / 基准 | 用来比较的市场指数，本项目主要使用CSI1000，CSI300作为大盘风格参照 |
| Excess / 超额 | 策略相对于基准的复合净值增幅，公式一般为 `策略净值/基准净值-1` |
| Return Diff / 收益率差 | `策略累计收益-基准累计收益`，是百分点差值；它和复合超额不是同一个公式 |
| IC | 某一天所有股票预测分数与实际收益的Pearson相关系数，关注线性关系 |
| Rank IC | 某一天预测排名与实际收益排名的Spearman相关系数，更符合选股任务 |
| ICIR / Rank ICIR | 每日IC或Rank IC的均值除以标准差，衡量信号相对波动的稳定程度 |

“Executable”不等于券商级真实成交。当前仍未逐笔模拟：A股100股一手、最低5元佣金、买卖侧费率差异、涨跌停无法成交、停牌、尾盘价格偏差和大单冲击成本。

### 14.5 `test_report.png`五条净值线

上半图默认包含五条线，所有线初始值约为1：

| 曲线 | 含义 | 是否扣简化成本 |
|---|---|---:|
| `Executable TopK Equity` | 主TopK策略的错峰毛净值；`holding-period=1`时就是每天TopK毛收益复利 | 否 |
| `TopK Net Equity`，例如 `Top3 Net Equity` | 每天TopK毛收益减去“实际换手率×cost_rate”后的净值 | 是 |
| `Executable Universe Equity` | 整个股票池等权平均收益的错峰净值，用于判断选股是否优于池内平均股票 | 否 |
| `CSI1000 Equity` | 同期中证1000指数净值，是CSI1000选股实验的主基准 | 否 |
| `CSI300 Equity` | 同期沪深300指数净值，用来观察大盘蓝筹风格表现 | 否 |

下半图是每日 `Rank IC` 柱状图：绿色表示当日排序方向正确，红色表示方向相反；黑色虚线是整个Test期间的平均Rank IC。单日柱子波动大是正常现象，应结合平均Rank IC、Rank ICIR、正值比例、净值和最大回撤共同判断。

注意：五条线里只有 `TopK Net Equity` 扣了简化成本，因此最接近当前研究级净收益；其他线主要作为毛收益和市场基准对照。

### 14.6 `daily_metrics.csv`字段字典

每一行代表一个信号日T；实际标签遵循当前任务的T+1收盘附近买入、T+2收盘附近卖出口径。

| 字段或字段模式 | 含义 |
|---|---|
| `datetime` | 信号日T |
| `IC` | 当日全股票预测score与实际label的Pearson相关系数 |
| `Rank IC` | 当日全股票预测排名与实际收益排名的Spearman相关系数 |
| `Universe Mean Return` | 当天股票池所有股票label的等权平均收益 |
| `TopN Mean Return` | 当天预测排名前N只股票的实际平均收益，N会输出1/3/5/10 |
| `Top1 Return` | 第一名股票实际收益；数值与 `Top1 Mean Return` 相同，保留用于兼容旧报告 |
| `Long Excess Return` | 主TopK平均收益减去Universe平均收益，是当日简单收益差 |
| `TopN Turnover` | TopN持仓相对前一日的替换比例。第一天按100%建仓处理 |
| `TopN Net Return` | `TopN Mean Return - TopN Turnover × cost_rate` |
| `TopN Net Equity` | TopN净收益逐日复利后的净值 |
| `TopK Cumulative` | 主TopK毛收益逐日复利后的累计收益，已经减1，因此0.20表示20% |
| `Universe Cumulative` | Universe平均收益逐日复利后的累计收益 |
| `Excess Cumulative` | 主TopK每日收益减Universe每日收益后直接复利的研究值；更严谨的净值超额优先看 `Executable Excess` |
| `Executable TopK Equity` | 按holding-period错峰计算的主TopK毛净值，未减1 |
| `TopN Equity` | 各TopN按holding-period错峰计算的毛净值 |
| `Executable Universe Equity` | Universe按holding-period错峰计算的毛净值 |
| `Executable Excess` | `Executable TopK Equity / Executable Universe Equity - 1` |
| `CSI1000 Return` / `CSI300 Return` | 与模型标签持仓时点一致的指数区间收益 |
| `CSI1000 Equity` / `CSI300 Equity` | 指数收益复利净值 |
| `Excess vs CSI1000/CSI300` | 主TopK毛净值除以对应指数净值再减1 |
| `TopN Excess vs CSI1000/CSI300` | 指定TopN毛净值相对于指数的复合超额 |
| `TopN Return Diff vs CSI1000/CSI300` | TopN累计收益与指数累计收益的直接百分点差 |

### 14.7 `summary.csv`字段字典

`summary.csv`通常只有一行，是整段Test的汇总。动态TopN字段会为Top1/3/5/10分别生成。

#### 实验身份和日期

| 字段 | 含义 |
|---|---|
| `experiment_pattern` | 本次匹配实验所用的字符串 |
| `model` | 入选recorder模型类型；混合时显示 `mixed` |
| `train_months` | 训练集月份长度 |
| `train_start` / `train_end` | 训练集起止日期 |
| `selected_recorders` | 通过Validation门槛并参与集成的recorder数量 |
| `top_models` | 是否只保留Validation排名前N个模型 |
| `weighting` | 集成权重方式 |
| `holding_period` | 持仓交易日数 |
| `cost_rate` | 完整换仓的简化综合成本率 |
| `test_start` / `test_end` | 实际共同Test区间 |
| `test_trading_days` | 实际参与评估的信号日数量 |

#### 预测排序能力

| 字段 | 含义 |
|---|---|
| `mean_IC` / `mean_Rank_IC` | Test期间每日IC/Rank IC平均值 |
| `ICIR` / `Rank_ICIR` | 每日IC/Rank IC均值除以标准差 |
| `Rank_IC_positive_ratio` | 每日Rank IC大于0的天数比例，不等于股票上涨胜率 |
| `TopN_win_rate` | TopN组合实际平均收益大于0的信号日比例 |

#### 收益和基准

| 字段 | 含义 |
|---|---|
| `gross_topk_cumulative` | 主TopK未扣成本的累计毛收益 |
| `gross_universe_cumulative` | 股票池全部股票等权累计毛收益 |
| `gross_excess_cumulative` | 每日TopK减Universe收益差的累计研究值 |
| `executable_topk_cumulative` | 主TopK按holding-period错峰后的毛累计收益 |
| `executable_universe_cumulative` | Universe错峰毛累计收益 |
| `executable_excess_cumulative` | 主TopK相对Universe的复合超额 |
| `csi1000_cumulative` / `csi300_cumulative` | 对应指数累计收益 |
| `csi1000_covered_days` / `csi300_covered_days` | 实际获得指数收益的日期数量 |
| `csi1000_last_valid_date` / `csi300_last_valid_date` | 指数最后一个有完整未来标签的信号日 |
| `excess_vs_csi1000` / `excess_vs_csi300` | 主TopK毛净值相对指数的复合超额 |
| `TopN_cumulative` | 指定TopN错峰毛累计收益 |
| `TopN_average_turnover` | Test期间平均换手率 |
| `TopN_net_cumulative` | 指定TopN扣简化成本后的累计净收益 |
| `TopN_excess_vs_csi1000/300` | 指定TopN毛净值相对指数的复合超额 |
| `TopN_return_diff_vs_csi1000/300` | 指定TopN累计毛收益减指数累计收益的百分点差 |

#### 绝对预测校准

| 字段 | 含义 |
|---|---|
| `prediction_mean` / `prediction_std` | 所有股票score的均值和标准差 |
| `realized_return_mean` / `realized_return_std` | 所有真实label的均值和标准差 |
| `absolute_MAE` | `abs(score-label)`平均值，越低越好；只有raw-label模型的score是收益率单位时才有直观金融意义 |
| `absolute_RMSE` | 均方根误差，对极端错误更敏感 |
| `prediction_bias` | `score-label`平均值；正值表示整体高估，负值表示整体低估 |
| `sample_correlation` | 将所有日期股票样本混合后计算的相关系数；不能替代每日横截面IC |

默认Alpha158标签经过横截面标准化时，score主要表达相对排名，不是“预测上涨3%”。这时MAE、RMSE和prediction bias缺少绝对收益率含义，应主要看Rank IC和TopK表现。

### 14.8 `absolute_return_calibration.csv`字段字典

这张表把所有预测score从低到高分成最多10个等样本分组，用来检查“预测越高，实际收益是否总体越高”。

| 字段 | 含义 |
|---|---|
| `score_bin` | score分组边界，例如 `(-0.0186, -0.00928]` |
| `samples` | 该分组包含的股票-日期样本数 |
| `predicted_mean` | 该组平均预测score |
| `predicted_min` / `predicted_max` | 该组score最小值和最大值 |
| `realized_mean` | 该组股票后续实际平均收益 |
| `realized_win_rate` | 该组实际收益大于0的样本比例 |

理想排序模型应大致表现为：score分组越高，`realized_mean`和`realized_win_rate`整体越高。它不要求每一档严格单调，因为市场噪声很大。对标准化标签模型，应重点看分组之间的相对单调性，而不是拿 `predicted_mean` 当百分比收益。

---

## 15. 原Test之后的前向评估、科创板归因与模型投票

### 15.1 为什么 `evaluate_batch.py --test-end` 不能自动测试新日期

`evaluate_batch.py` 默认读取 recorder 训练时保存的 `pred.pkl` 和 `label.pkl`。`--test-start/--test-end` 只能在这些历史预测的日期范围内截取，不能让模型对原Test结束后的日期重新推理。XGBoost-240 Fold3保存的Test截止2026-07-17，因此把 `--test-end` 写成2026-07-21不会产生7月20/21的新预测。

对原Test之后的日期必须使用：

```bash
$PY script/evaluate_forward_recorder.py \
  --experiment-id 473901139733640553 \
  --recorder-id 021853d841234c4fb397e0e919800b58 \
  --start 2026-07-15 \
  --end 2026-07-21 \
  --cost-rate 0.0015
```

该脚本会重新加载模型和Alpha158处理器，对指定日期重新推理，然后通过Qlib最新收盘价计算真实标签。当前1日标签需要信号日之后两个交易日收盘价，因此截至7月21只能完整评价到7月17；7月20要等7月22收盘入库，7月21要等7月23。

输出目录为 `.qlibAssistant/analysis/forward_evaluation_YYYYMMDD_HH_MM_SS/`，包含 `predictions_with_labels.csv`、`board_variant_summary.csv`、`board_variant_daily.csv` 和 `evaluation_config.json`。

### 15.2 自动生成含/剔除科创板评估

`evaluate_batch.py` 现在每次额外生成 `board_variant_summary.csv` 和 `board_variant_daily.csv`。`universe_variant=all` 是完整股票池，`ex_star` 是先剔除SH688/SH689再重新选TopK。两种口径都输出毛累计收益、扣换手成本净累计收益、换手率、胜率、最大回撤、简单年化Sharpe和最好5天收益集中度。

XGB240 Fold3的Top3结果：

| 股票池 | 毛累计 | 扣0.15%换手成本净累计 | 最大回撤 |
|---|---:|---:|---:|
| 全池 | +63.81% | +44.68% | -26.12% |
| 剔除科创板 | +26.27% | +11.88% | -23.63% |

### 15.3 DoubleEnsemble简介与240个月三折结果

当前 `DEnsembleModel` 内部使用6个LightGBM子模型。训练过程中交替进行样本重加权（后续模型更关注前面模型难以拟合的样本）和特征选择（不同阶段选择更有用的因子），最后按 `1,0.2,0.2,0.2,0.2,0.2` 合成预测。它比单一GBDT更复杂、训练更慢，也可能在不同市场状态下表现不一致。

Top3扣0.15%简化成本：

| Fold | Test | 全池净累计 | 剔除科创板净累计 | 结论 |
|---|---|---:|---:|---|
| Fold1 | 2025-04～2025-09 | +61.16% | +53.23% | 强 |
| Fold2 | 2025-09～2026-02 | +68.14% | +76.44% | 强 |
| Fold3 | 2026-02～2026-07 | -5.93% | -12.87% | 最新市场失效 |

因此DoubleEnsemble-240暂时不应替代XGB240作为当前主模型；它可以作为集成多样性来源，但必须由Selection Validation决定权重，不能因为Fold1/2收益高就忽略最新Fold3。

### 15.4 有限投票和时序持续性探索

```bash
$PY script/evaluate_consensus_strategies.py \
  --component 473901139733640553,021853d841234c4fb397e0e919800b58,XGB240 \
  --component 143011367364704830,cde04bab7b454f2483f71f5281bff571,DE240 \
  --cost-rate 0.0015
```

预先定义的规则包括：两个模型都进Top10、都进Top20、平均排名Top10且排名差不超过5，以及连续两天都满足Top20。当前同一Fold3 Test上的探索中，截面一致投票表现很高，而“连续两天Top20”明显亏损。这些结果已经查看Test，属于假设生成，不是无偏证据；后续只能把少量规则冻结后放到新的历史Walk-forward Fold或未来前向数据验证，不能继续按本Test最优结果反复调阈值。

## 16. TRA（Temporal Routing Adaptor）

### 16.1 TRA解决什么问题

TRA来自KDD 2021论文《Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport》。它不是强化学习。TRA假设市场包含多个随时间切换的交易模式，用一个时序骨干提取股票表示，再由多个预测器分别学习不同模式，路由器根据当前隐表示（LR）和历史预测误差（TPE）选择预测器。最优传输约束用于避免所有预测器退化成同一个模型。

本项目第一版采用Qlib官方Alpha158-full结构：

```text
最近60个交易日 × 158因子
        ↓
两层LSTM（hidden=256）+ 时间注意力
        ↓
3个潜在状态预测器
        ↓
LR_TPE路由器（LSTM hidden=32）
        ↓
最终score
```

训练分为两阶段：

1. `pretrain=True`：用oracle分配预训练LSTM和多个预测头；
2. `transport_method=router`：训练路由器学习在未知未来标签时选择市场模式。

### 16.2 环境安装

TRA依赖PyTorch，安装到日常使用的`qlibAssistant`环境：

```bash
conda run -n qlibAssistant python -m pip install -r requirements-tra.txt
```

当前Mac上的Qlib 0.9.7 TRA实现只自动选择CUDA或CPU，不使用Apple MPS，因此这里实际走CPU。

### 16.3 Smoke测试

先验证完整的数据、预训练、路由训练、Recorder、Selection Validation和重新加载预测链路：

```bash
$PY script/run.py \
  --pool csi300 \
  --run-tag tra_alpha158_smoke_YYMMDD \
  --models TRA \
  --model-preset tra_smoke \
  --segments-json '[{
    "train":["2024-01-02","2024-06-28"],
    "valid":["2024-07-01","2024-08-30"],
    "selection_valid":["2024-09-02","2024-09-30"],
    "test":["2024-10-08","2024-10-31"]
  }]'
```

`tra_smoke`仍然执行两个训练阶段，但每阶段仅2个epoch、每epoch最多2个batch，不能用于判断模型效果。

`tra_pilot`保持相同网络结构，每阶段最多10个epoch、每epoch20个batch、早停5，用于单fold耗时测量和初步有效性检查；它仍不是正式官方训练。

Apple Silicon上长窗口TRA曾在训练子进程进入首个epoch前以原生信号`-11`退出。逐段诊断证明数据准备、首批数据、LSTM前向、反向传播和优化器更新均正常；根因是spawn子进程中的BLAS/OpenMP线程过度订阅，而不是模型内存不足。`script/run.py`现在会为TRA子进程自动设置`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`和`VECLIB_MAXIMUM_THREADS=1`。

`tra_smoke`和两个pilot使用`batch_size=256`。`tra_pilot_20f`采用Qlib官方TRA配置中使用的20个Alpha158因子、`hidden_size=64`，适合Mac CPU上快速判断链路和初步泛化；`tra_official_full`仍保留158因子、hidden 256、batch 1024的官方完整口径。若再次遇到原生崩溃，可运行`script/debug_tra_native.py`逐段检查prepare、batch、forward和backward。

2026-07-23端到端验证结果：

- Experiment ID：`390707062670142558`
- Recorder ID：`472e864d724b4a9f869b4acfd40027af`
- 总耗时：约231秒
- Test Rank IC：约`0.0030`
- Selection Validation Rank IC：约`-0.0493`

指标很差是预期现象：训练集只有半年且每个epoch只更新2个batch。这个实验的结论仅为“实现链路可运行并可保存、重载和预测”，不能与XGBoost-240比较。

24个月20因子pilot：

- Experiment ID：`690553590774368978`
- Recorder ID：`7ddef05be5e146a7b4618a75384352f0`
- Train：2022-06-17～2024-06-16；Test：2025-04-17～2025-09-16
- 训练与Recorder导出耗时：约404秒
- Selection Validation Rank IC / Rank ICIR：`0.0661 / 0.2933`
- Test Rank IC / Rank ICIR：`0.0140 / 0.0646`
- Test Top3毛累计 / 扣简化成本后累计：`5.28% / -0.94%`
- 同期CSI300 / CSI1000累计：`19.23% / 28.20%`

这个pilot在Selection Validation上有较强信号，但到独立Test明显衰减；扣除高换手的简化成本后Top1、Top3均为负，因此暂时不能用于实盘，也不能替代XGBoost-240。完整报告位于`.qlibAssistant/analysis/evaluation_20260723_18_24_19/`。

注意：TRA预训练日志中可能出现约`0.88`的Validation IC。该阶段使用真实标签做oracle专家分配，是训练上界，不是可交易结果。必须看第二阶段router训练结束后的Selection Validation和Test指标。

### 16.4 正式训练

正式官方完整158因子结构使用：

```bash
$PY script/run.py \
  --pool csi1000 \
  --run-tag tra_alpha158_full_YYMMDD \
  --models TRA \
  --model-preset tra_official_full \
  --segments-json '<与其他模型完全相同的固定分段JSON>'
```

正式preset为100个预训练epoch加最多100个TRA epoch，早停20。CPU成本很高；在运行240个月三折前，应先使用一个fold和较短窗口测量每epoch耗时。公平比较时必须与XGBoost使用完全相同的Train、Valid、Selection Validation、Test和交易成本。

### 16.5 Recorder预测

TRA Recorder可使用通用预测脚本：

```bash
$PY script/predict_recorder_date.py \
  --experiment-id 390707062670142558 \
  --recorder-id 472e864d724b4a9f869b4acfd40027af \
  --date 2024-10-31 \
  --topk 10
```

Qlib 0.9.7的TRAModel序列化不会保存TensorBoard `_writer` 字段，本项目已在验证、普通模型选择、指定Recorder预测和前向评测入口统一恢复该运行期字段，不修改Conda环境中的Qlib源码。

### 16.6 Qlib强化学习位置

Qlib强化学习代码位于安装包的`qlib/rl/`，核心模块包括：

- `simulator.py`：环境/市场模拟器接口；
- `interpreter.py`：把交易状态和策略动作转换成RL观测与动作；
- `reward.py`：奖励定义；
- `trainer/`：训练、回测、并行环境和callback；
- `order_execution/`：单资产订单执行环境、策略和奖励；
- `data/`：分钟级执行数据；
- `strategy/base.py`中的`RLStrategy`：将RL策略接入Qlib嵌套执行框架。

Qlib现有RL示例主要用于把一张大订单拆分到多个分钟执行，目标是降低价格冲击和成交偏差；它不是直接替代Alpha158选股模型。TRA虽然有“路由选择”行为，但通过监督学习、历史损失和最优传输训练，不属于强化学习。

### 16.7 TRA正式长训练的最终状态

2026-07-24的CSI1000、240个月Alpha20正式训练最终完成，耗时约2小时14分钟，
但Selection Validation Rank ICIR仅`0.0342`，Test Rank IC / Rank ICIR为
`-0.0036 / -0.0224`，没有可泛化的选股能力。随后启动的Alpha158-full没有形成
完整`params.pkl`和`pred.pkl`，属于未完成recorder，不能预测或回测。

因此16.4中的命令保留为研究入口，不代表当前存在可用TRA正式模型。若以后重试，
应先加入checkpoint和多随机种子短窗口实验，不应直接再次运行240个月全量任务。

---

## 17. 学习资料索引

近期讨论已经从操作日志整理为系统学习资料：

- 总索引：`daily_learning/README.md`
- 系统手册：`daily_learning/2026-07-27_近期量化研究学习手册.md`
- 精确实验结果：`daily_learning/2026-07-27_主板TopN集成与最新预测.md`

系统手册覆盖数据层、Alpha158样本结构、四段数据、Fold、Recorder、MLflow、
模型家族、IC/ICIR、TopK、成交约束、A股指数、市场宽度、小盘实验、回测过拟合
和个人10万元账户的研究边界。学习概念时优先读系统手册，复现实验时再回到本文
和对应CSV。

---

## 18. 每日四路预测与决策Skill

2026-07-28起，日常更新不再只运行旧的CSI300 `model selection`，而是固定为：

1. 更新Tushare、Dolt和Qlib日线；
2. 运行Mainboard20、XGBoost-240、Fixed Ensemble三路个股排名；
3. 运行全A市场宽度Top2/4/6/8/20；
4. 合并三路Top10票数；
5. 对XGBoost/Fixed应用市场宽度Top2不低于40%的候选门槛；
6. 将`event_guard`标记为“待买入日尾盘确认”，不使用前一晚不存在的行情。

### 18.1 一键入口

对Agent说“更新数据并生成今天的决策”，或运行`/update-predict`。对应Skill为：

```text
/Users/hmax/.agents/skills/source-command-update-predict/SKILL.md
```

本地确定性脚本：

```bash
$PY script/run_daily_decision_pipeline.py --update
```

也可以使用Shell入口：

```bash
bash script/run_daily_prediction.sh
```

只预测、不更新数据：

```bash
$PY script/run_daily_decision_pipeline.py
```

复用已有四路预测并重新生成某日汇总：

```bash
$PY script/run_daily_decision_pipeline.py \
  --date 2026-07-28 \
  --reuse-existing
```

市场宽度缓存已经覆盖信号日时，脚本自动跳过耗时约3～4分钟的全历史Dolt聚合；
需要强制重建时添加`--force-breadth-rebuild`。

### 18.2 输出

```text
.qlibAssistant/analysis/daily_decision_YYYYMMDD_HHMMSS/
├── DECISION_SUMMARY.md
├── source_actions.csv
├── consensus_top100.csv
├── decision_config.json
└── 各步骤.log
```

- `source_actions.csv`：三路Top1、市场门槛和初步动作；
- `consensus_top100.csv`：三路名次、Top10/Top20票数；
- `DECISION_SUMMARY.md`：每天最优先阅读的清单；
- `decision_config.json`：信号日、预计买卖日、阈值和原始预测目录。

三路共识尚未单独回测；Mainboard20也尚未测试40%市场门槛。不得把这两个观察
字段描述为已经验证的硬规则。

### 18.3 event_guard不能在前一晚完成

当前信号在T日收盘后产生，预计T+1尾盘买入。`event_guard`需要T+1临近收盘时
可见的当日最低价、当前回撤和可交易状态，所以T日晚间只能标记为待确认：

- 近10日跌停次数不超过1；
- 相对20日最高收盘回撤高于-30%；
- T+1最低价未触及跌停；
- 当前未涨停、未停牌且能够成交。

不通过就留现金，不自动购买下一名。要消除完整日线与同收盘成交的理想化，
后续应接入14:50～14:55分钟行情做前向执行。

### 18.4 Fixed Ensemble是否有未来函数

当前Fixed Ensemble的四个模型在每日预测时只读取信号日及以前的数据，没有
直接把未来价格或Test标签输入模型，因此不能简单称为“有未来函数”。

它的主要问题是**事后选择偏差**：

- XGBoost-60m/120m、LightGBM-84m、CatBoost-120m是在看过多轮历史实验表现后
  冻结的；
- 当前真正冻结的四个组件都来自Fold3；
- Fold1、Fold2报告是事后按同一架构寻找对应Recorder重建，并非当时就冻结的
  一套生产组合；
- 40%市场门槛和`event_guard`也在已经查看过的Test上做过探索。

因此Fixed的+39%～+40%三折平均回测更可能高估未来表现。准确表述是“没有直接
时间泄漏，但存在较强的数据窥探、模型选择和多重比较偏差”。解决方式是在当前
时点冻结组件、权重和交易规则，此后只看全新的前向日期，不能继续根据同一批
Test结果修改后再引用原收益。
