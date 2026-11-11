"""Distributed Data Parallel training utilities for PyGo."""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional


def init_distributed(backend: str = "nccl") -> tuple:
    """Initialize distributed training from environment variables."""
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    if world_size > 1:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: nn.Module, local_rank: int,
               find_unused: bool = False) -> nn.Module:
    """Wrap model with DistributedDataParallel."""
    model = model.cuda(local_rank)
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
    return model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all ranks."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


def broadcast_object(obj, src: int = 0):
    """Broadcast a Python object from src to all ranks."""
    if not dist.is_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


class GradientSynchronizer:
    """Accumulates gradients locally, then syncs across ranks."""
    def __init__(self, model: nn.Module, accum_steps: int = 1):
        self.model       = model
        self.accum_steps = accum_steps
        self._step       = 0

    def __enter__(self):
        self._step += 1
        if (hasattr(self.model, "no_sync") and
                self._step % self.accum_steps != 0):
            return self.model.no_sync()
        return self._null_ctx()

    def __exit__(self, *args): pass

    class _null_ctx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
