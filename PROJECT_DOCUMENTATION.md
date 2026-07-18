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
全量 5 模型（XGBoost/Linear/DoubleEnsemble/LightGBM/CatBoost，约 40 分钟）：
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
