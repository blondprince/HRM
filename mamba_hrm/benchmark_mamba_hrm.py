from typing import List
import yaml
import os

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf

from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader


class BenchmarkConfig(pydantic.BaseModel):
    baselines: List[str]
    mamba_hrm: str


def _load_config_from_checkpoint(checkpoint_path: str) -> PretrainConfig:
    with open(os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml"), "r") as f:
        return PretrainConfig(**yaml.safe_load(f))


def _run_eval_single(checkpoint: str):
    RANK = 0
    WORLD_SIZE = 1
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = _load_config_from_checkpoint(checkpoint)
    config.checkpoint_path = os.path.dirname(checkpoint)

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    try:
        train_state.model.load_state_dict(torch.load(checkpoint, map_location="cuda"), assign=True)
    except Exception:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in torch.load(checkpoint, map_location="cuda").items()},
            assign=True,
        )

    train_state.step = 0
    ckpt_filename = os.path.basename(checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    return metrics


def launch():
    bench_cfg = BenchmarkConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore

    print("=== Baseline HRM checkpoints ===")
    for ckpt in bench_cfg.baselines:
        print(f"[Baseline] {ckpt}")
        metrics = _run_eval_single(ckpt)
        if metrics is not None:
            print(metrics)

    print("=== Mamba-HRM checkpoint ===")
    print(f"[Mamba-HRM] {bench_cfg.mamba_hrm}")
    metrics = _run_eval_single(bench_cfg.mamba_hrm)
    if metrics is not None:
        print(metrics)


if __name__ == "__main__":
    launch()

