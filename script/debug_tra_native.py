"""Pinpoint native crashes in Qlib's TRA training path.

This intentionally avoids MLflow and the training subprocess so each checkpoint
identifies the last successful operation before a native crash.
"""

from __future__ import annotations

import argparse
import copy
import faulthandler
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import qlib
import torch
import torch.optim as optim
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "roll"))

from myconfig import get_my_config


def mark(message: str) -> None:
    print(f"[TRA-DIAG] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--preset", default="tra_pilot_20f")
    parser.add_argument("--pool", default="csi300")
    parser.add_argument(
        "--through",
        choices=("prepare", "batch", "forward", "backward", "fit"),
        default="backward",
    )
    args = parser.parse_args()

    faulthandler.enable(all_threads=True)
    mark(f"python={os.sys.executable}")
    mark(f"torch={torch.__version__}, device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    qlib.init(provider_uri=str(Path(args.provider_uri).expanduser()), region=REG_CN)

    task = get_my_config("TRA", "Alpha158", args.pool, model_preset=args.preset)
    segments = {
        "train": ("2022-06-17", "2024-06-16"),
        "valid": ("2024-06-17", "2024-11-16"),
        "test": ("2025-04-17", "2025-09-16"),
    }
    task["dataset"]["kwargs"]["segments"] = segments
    handler_kwargs = task["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_kwargs.update(
        {
            "start_time": segments["train"][0],
            "end_time": segments["test"][1],
            "fit_start_time": segments["train"][0],
            "fit_end_time": segments["train"][1],
        }
    )

    mark("initializing dataset")
    dataset = init_instance_by_config(task["dataset"])
    mark("preparing train/valid/test")
    train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])
    mark(f"prepared lengths: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")

    mark("initializing model")
    model = init_instance_by_config(task["model"])
    mark("deep-copying backbone state")
    copy.deepcopy(model.model.state_dict())
    mark("deep-copying TRA state")
    copy.deepcopy(model.tra.state_dict())
    mark("creating Adam optimizer")
    model.optimizer = optim.Adam(
        list(model.model.parameters()) + list(model.tra.predictors.parameters()),
        lr=model.lr,
    )
    if args.through == "fit":
        mark("running complete model.fit without MLflow/subprocess")
        model.fit(dataset, evals_result={})
        mark("complete model.fit finished")
        return
    if args.through == "prepare":
        return

    mark("fetching first training batch")
    train_set.train()
    batch = next(iter(train_set))
    mark(
        "batch fetched: "
        + ", ".join(f"{key}={tuple(value.shape)}" for key, value in batch.items() if hasattr(value, "shape"))
    )
    if args.through == "batch":
        return

    mark("running one pretrain forward/loss")
    model.model.train()
    model.tra.train()
    data = batch["data"].to(model.model.rnn.weight_ih_l0.device)
    label = batch["label"].to(data.device)
    hidden = model.model(data)
    pred = model.tra.predictors(hidden)
    loss = ((pred - label.unsqueeze(-1)) ** 2).mean()
    mark(f"forward complete: loss={loss.item():.6f}")
    if args.through == "forward":
        return

    mark("running backward")
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    mark("backward and optimizer step complete")


if __name__ == "__main__":
    main()
