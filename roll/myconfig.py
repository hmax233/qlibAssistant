#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
from pathlib import Path

import yaml

CSI300_MARKET = "csi300"
CSI100_MARKET = "csi100"

CSI300_BENCH = "SH000300"

DATASET_ALPHA158_CLASS = "Alpha158"
DATASET_ALPHA360_CLASS = "Alpha360"

###################################
# config
###################################

# Supported Models:
model_list = [
    "CatBoost",
    "KRNN",
    "Sandwich",
    "Linear",
    "XGBoost",
    "TFT",
    "TRA",
    "DoubleEnsemble",
    "LightGBM",
]


CATBOOST_MODEL = {
    "class": "CatBoostModel",
    "module_path": "qlib.contrib.model.catboost_model",
    "kwargs": {
        "loss": "RMSE",
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "max_depth": 6,
        "num_leaves": 100,
        "thread_count": 14, # 匹配 M5 Pro 15 核，留 1 核给系统
        "grow_policy": "Lossguide",
        "bootstrap_type": "MVS"
    }
}

KRNN_MODEL = {
   "class": "KRNN",
    "module_path": "qlib.contrib.model.pytorch_krnn",
    "kwargs": {
        "fea_dim": 6,
        "cnn_dim": 8,
        "cnn_kernel_size": 3,
        "rnn_dim": 8,
        "rnn_dups": 2,
        "rnn_layers": 2,
        "n_epochs": 200,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "metric": "loss",
        "GPU": 0
    }
}

SANDWICH_MODEL = {
  "class": "Sandwich",
    "module_path": "qlib.contrib.model.pytorch_sandwich",
    "kwargs": {
        "fea_dim": 6,          # ⚠️ 注意：需修改为实际因子数量 (如 Alpha158 为 158)
        "cnn_dim_1": 16,
        "cnn_dim_2": 16,
        "cnn_kernel_size": 3,
        "rnn_dim_1": 8,
        "rnn_dim_2": 8,
        "rnn_dups": 2,
        "rnn_layers": 2,
        "n_epochs": 200,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "metric": "loss",
        "GPU": 0               # GPU 索引，无显卡改为 -1
    }
}

TRA_MODEL = {
    "class": "TRAModel",
    "module_path": "qlib.contrib.model.pytorch_tra",
    "kwargs": {
        # Official Alpha158-full backbone: a two-layer attentive LSTM.
        "model_config": {
            "input_size": 158,
            "hidden_size": 256,
            "num_layers": 2,
            "rnn_arch": "LSTM",
            "use_attn": True,
            "dropout": 0.2,
        },
        # Three latent predictors represent different temporal trading patterns.
        "tra_config": {
            "num_states": 3,
            "rnn_arch": "LSTM",
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "tau": 1.0,
            "src_info": "LR_TPE",
        },
        "model_type": "RNN",
        "lr": 0.001,
        "n_epochs": 100,
        "early_stop": 20,
        "max_steps_per_epoch": None,
        "lamb": 1.0,
        "rho": 0.99,
        "alpha": 0.5,
        "seed": 0,
        # Keep TRA outputs inside the Qlib/MLflow recorder instead of a shared
        # local output directory that would collide across folds.
        "logdir": None,
        "eval_train": False,
        "eval_test": False,
        "pretrain": True,
        "init_state": None,
        "freeze_model": False,
        "freeze_predictors": False,
        "transport_method": "router",
        "memory_mode": "sample",
    },
}

LINEAR_MODEL = {
  "class": "LinearModel",
    "module_path": "qlib.contrib.model.linear",
    "kwargs": {
        "estimator": "ridge",  # 岭回归
        "alpha": 0.05
    }
}

XGBOOST_MODEL = {
 "class": "XGBModel",
    "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {
        "eval_metric": "rmse",
        "colsample_bytree": 0.8879,
        "eta": 0.0421,
        "max_depth": 8,
        "subsample": 0.8789,
        # Qlib XGBModel.fit 使用 num_boost_round=1000 并在 validation 上早停；
        # n_estimators 属于 sklearn API，在这里会被 xgboost 忽略。
        "nthread": 14
    }
}

DOUBLE_ENSEMBLE_MODEL = {
 "class": "DEnsembleModel",
    "module_path": "qlib.contrib.model.double_ensemble",
    "kwargs": {
        # --- DoubleEnsemble 自身参数 ---
        "base_model": "gbm",      # 内部使用的基础模型类型
        "loss": "mse",            # 损失函数
        "num_models": 6,          # 集成中包含 6 个子模型
        "enable_sr": True,        # 启用子采样 (Sample Reuse)
        "enable_fs": True,        # 启用特征选择 (Feature Selection)
        "alpha1": 1,
        "alpha2": 1,
        "bins_sr": 10,
        "bins_fs": 5,
        "decay": 0.5,
        
        # ⚠️ 注意: sample_ratios 有5个, sub_weights 有6个
        "sample_ratios": [
            0.8,
            0.7,
            0.6,
            0.5,
            0.4
        ],
        "sub_weights": [
            1,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2
        ],
        
        # --- 传递给内部 base_model (gbm) 的参数 ---
        "epochs": 136,
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 14, # 匹配 M5 Pro 15 核，留 1 核给系统
        "verbosity": -1
    }
}

GBDT_MODEL = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 14, # 匹配 M5 Pro 15 核，留 1 核给系统
    },
}


SA_RC = {
    "class": "SigAnaRecord",
    "module_path": "qlib.workflow.record_temp",
}


RECORD_CONFIG = [
    {
        "class": "SignalRecord",
        "module_path": "qlib.workflow.record_temp",
        "kwargs": {
            "dataset": "<DATASET>",
            "model": "<MODEL>",
        },
    },
    SA_RC,
]

def get_data_handler_config(
    start_time="2018-01-01",
    end_time="2066-08-01",
    fit_start_time=None, 
    fit_end_time=None,
    instruments=CSI300_MARKET,
    label_horizon=1,
    normalize_features=False,
    raw_label=False,
):
    label_horizon = int(label_horizon)
    if label_horizon < 1:
        raise ValueError("label_horizon must be >= 1")
    label = (
        [f"Ref($close, -{label_horizon + 1})/Ref($close, -1) - 1"],
        [f"LABEL{label_horizon}D"],
    )
    config = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "instruments": instruments,
        "label": label,
    }
    if normalize_features:
        config["infer_processors"] = [
            {
                "class": "RobustZScoreNorm",
                "kwargs": {"fields_group": "feature", "clip_outlier": True},
            },
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ]
    if raw_label:
        # 仅删除缺失标签，不做默认的横截面 CSZScoreNorm；模型直接学习绝对收益率。
        config["learn_processors"] = [{"class": "DropnaLabel"}]
    return config

def get_dataset_config(
    dataset_class=DATASET_ALPHA158_CLASS,
    train=("2015-01-01", "2016-12-31"),
    valid=("2017-01-01", "2017-02-28"),
    test=("2017-03-01", "2026-12-31"),
    handler_kwargs=None,  # 建议默认值设为 None，避免可变参数陷阱
):
    # 1. 如果没传 handler_kwargs，给个默认字典
    if handler_kwargs is None:
        handler_kwargs = {"instruments": CSI300_MARKET}
    
    # 为了不修改外部传入的字典，建议 copy 一份
    kwargs = handler_kwargs.copy()

    # ================= 关键修改 =================
    # 手动把 train 的时间填进去，替换掉那个 "<...>" 占位符
    # train[0] 就是开始时间，train[1] 就是结束时间
    if "fit_start_time" not in kwargs:
        kwargs["fit_start_time"] = train[0]
    
    if "fit_end_time" not in kwargs:
        kwargs["fit_end_time"] = train[1]
    # ===========================================

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": dataset_class,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": get_data_handler_config(**kwargs), # 使用填充好日期的 kwargs
            },
            "segments": {
                "train": train,
                "valid": valid,
                "test": test,
            },
        },
    }


def get_tra_dataset_config(
    dataset_class=DATASET_ALPHA158_CLASS,
    train=("2015-01-01", "2016-12-31"),
    valid=("2017-01-01", "2017-02-28"),
    test=("2017-03-01", "2026-12-31"),
    handler_kwargs=None,
):
    """Build the memory-augmented time-series dataset required by TRA.

    TRA cannot consume the project's ordinary ``DatasetH``.  The official
    implementation uses ``MTSDatasetH`` to expose both a 60-session feature
    sequence and the historical routing-loss memory, while keeping the same
    Alpha158 handler and date segments used by the rest of this project.
    """

    if dataset_class != DATASET_ALPHA158_CLASS:
        raise ValueError("当前TRA接入仅验证了Alpha158；Alpha360需使用input_size=6的独立配置")
    kwargs = (handler_kwargs or {"instruments": CSI300_MARKET}).copy()
    kwargs["normalize_features"] = True
    handler_config = get_data_handler_config(**kwargs)
    # Match the official TRA Alpha158 workflow: rank-normalized labels are
    # used for routing and signal learning.
    handler_config["learn_processors"] = [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ]
    return {
        "class": "MTSDatasetH",
        "module_path": "qlib.contrib.data.dataset",
        "kwargs": {
            "handler": {
                "class": dataset_class,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_config,
            },
            "segments": {
                "train": train,
                "valid": valid,
                "test": test,
            },
            "seq_len": 60,
            "horizon": int(kwargs.get("label_horizon", 1)),
            "num_states": 3,
            "memory_mode": "sample",
            "batch_size": 1024,
            "n_samples": None,
            "shuffle": True,
            "drop_last": True,
            "input_size": None,
        },
    }


def get_gbdt_task(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        "model": GBDT_MODEL,
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
        "record": RECORD_CONFIG,
    }


def get_record_lgb_config(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
        },
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
        "record": RECORD_CONFIG,
    }


def get_record_xgboost_config(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        "model": {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
        },
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
        "record": RECORD_CONFIG,
    }


CSI300_DATASET_CONFIG = get_dataset_config(handler_kwargs={"instruments": CSI300_MARKET})
CSI300_GBDT_TASK = get_gbdt_task(handler_kwargs={"instruments": CSI300_MARKET})

CSI100_RECORD_XGBOOST_TASK_CONFIG = get_record_xgboost_config(handler_kwargs={"instruments": CSI100_MARKET})
CSI100_RECORD_LGB_TASK_CONFIG = get_record_lgb_config(handler_kwargs={"instruments": CSI100_MARKET})
CSI300_RECORD_LGB_TASK_CONFIG = get_record_lgb_config(handler_kwargs={"instruments": CSI300_MARKET})


def get_model_config(model_name: str, model_preset: str | None = None):
    configs = {
        "XGBoost": XGBOOST_MODEL,
        "CatBoost": CATBOOST_MODEL,
        "KRNN": KRNN_MODEL,
        "Sandwich": SANDWICH_MODEL,
        "TRA": TRA_MODEL,
        "Linear": LINEAR_MODEL,
        "DoubleEnsemble": DOUBLE_ENSEMBLE_MODEL,
        "LightGBM": GBDT_MODEL,
    }
    if model_name not in configs:
        raise ValueError(f"Model {model_name} is not supported.")
    config = copy.deepcopy(configs[model_name])
    if model_name == "TRA":
        if model_preset in (None, "official", "official_full", "tra_official_full"):
            return config
        if model_preset in ("smoke", "tra_smoke"):
            config["kwargs"].update(
                {
                    "n_epochs": 2,
                    "early_stop": 2,
                    "max_steps_per_epoch": 2,
                }
            )
            # Keep the architecture and pretrain stage identical so the smoke
            # test validates routing, memory and serialization end to end.
            return config
        raise ValueError(
            "未知TRA preset: "
            f"{model_preset}; 可选: tra_smoke, tra_official_full"
        )
    if model_name != "LightGBM" or not model_preset:
        return config
    preset_path = Path(__file__).with_name("model_params.yaml")
    presets = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    try:
        overrides = presets["LightGBM"][model_preset]
    except KeyError as error:
        available = ", ".join((presets.get("LightGBM") or {}).keys())
        raise ValueError(f"未知 LightGBM preset: {model_preset}; 可选: {available}") from error
    config["kwargs"].update(overrides)
    return config

def get_my_config(
    model_name: str,
    dataset_name: str,
    stock_pool: str,
    label_horizon: int = 1,
    normalize_features: bool = False,
    raw_label: bool = False,
    model_preset: str | None = None,
):
    handler_kwargs = {
        "instruments": stock_pool,
        "label_horizon": label_horizon,
        "normalize_features": normalize_features,
        "raw_label": raw_label,
    }
    dataset_config_factory = (
        get_tra_dataset_config if model_name == "TRA" else get_dataset_config
    )
    return {
        "model": get_model_config(model_name, model_preset=model_preset),
        "dataset": dataset_config_factory(
            dataset_class=dataset_name, handler_kwargs=handler_kwargs
        ),
        "record": RECORD_CONFIG,
    }
