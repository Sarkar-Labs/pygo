"""PyGo FSDP — Fully Sharded Data Parallel for training models larger than GPU memory."""
from __future__ import annotations

import math
import functools
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable, Set, Type, Any

import torch
import torch.distributed as dist
import torch.nn as nn


# ─── Sharding strategies ──────────────────────────────────────────────────────

class ShardingStrategy(Enum):
    FULL_SHARD       = "full_shard"        # ZeRO-3: shard params + grads + optim states
    SHARD_GRAD_OP    = "shard_grad_op"     # ZeRO-2: shard grads + optim states only
    NO_SHARD         = "no_shard"          # DDP-equivalent
    HYBRID_SHARD     = "hybrid_shard"      # shard within node, replicate across nodes


class BackwardPrefetch(Enum):
    BACKWARD_PRE  = "backward_pre"   # prefetch next layer's params before current backward
    BACKWARD_POST = "backward_post"


@dataclass
class FSDPConfig:
    sharding_strategy:     ShardingStrategy   = ShardingStrategy.FULL_SHARD
    backward_prefetch:     BackwardPrefetch   = BackwardPrefetch.BACKWARD_PRE
    mixed_precision:       bool               = True
    param_dtype:           torch.dtype        = torch.bfloat16
    reduce_dtype:          torch.dtype        = torch.float32
    buffer_dtype:          torch.dtype        = torch.bfloat16
    cpu_offload:           bool               = False
    activation_checkpointing: bool            = True
    min_num_params:        int                = 1_000_000   # auto-wrap threshold
    sync_module_states:    bool               = True
    use_orig_params:       bool               = True
    limit_all_gathers:     bool               = True


# ─── Flat Parameter (the core sharding primitive) ─────────────────────────────

class FlatParameter(nn.Parameter):
    """
    A single flattened, 1-D parameter that represents the concatenation of
    multiple original parameters. Only this shard's local chunk lives on
    this rank; full parameter is materialised via all-gather before use.
    """
    def __new__(cls, data, requires_grad=True):
        return super().__new__(cls, data, requires_grad)

    def init_metadata(self, param_infos: List["ParamInfo"], shapes, numels, names):
        self._param_infos = param_infos
        self._shapes       = shapes
        self._numels        = numels
        self._names         = names
        self._is_sharded    = False
        self._local_shard: Optional[torch.Tensor] = None
        self._full_param:  Optional[torch.Tensor] = None


@dataclass
class ParamInfo:
    module:      nn.Module
    param_name:  str
    module_name: str


# ─── FlatParamHandle ──────────────────────────────────────────────────────────

class FlatParamHandle:
    """Manages flattening, sharding, all-gather, and reduce-scatter for one
    FSDP unit's set of parameters."""

    def __init__(self, params: List[nn.Parameter], param_infos: List[ParamInfo],
                 rank: int, world_size: int, cfg: FSDPConfig):
        self.rank       = rank
        self.world_size = world_size
        self.cfg        = cfg

        shapes  = [p.shape for p in params]
        numels  = [p.numel() for p in params]
        names   = [info.param_name for info in param_infos]

        flat_data = torch.cat([p.detach().reshape(-1) for p in params])
        self.flat_param = FlatParameter(flat_data, requires_grad=True)
        self.flat_param.init_metadata(param_infos, shapes, numels, names)

        self._shard_param()

    def _shard_param(self):
        """Split the flat parameter into `world_size` equal chunks; keep only
        this rank's chunk resident in GPU memory."""
        total       = self.flat_param.numel()
        chunk_size  = math.ceil(total / self.world_size)
        padded_size = chunk_size * self.world_size

        if padded_size > total:
            pad = torch.zeros(padded_size - total, dtype=self.flat_param.dtype)
            full = torch.cat([self.flat_param.data, pad])
        else:
            full = self.flat_param.data

        start = self.rank * chunk_size
        end   = start + chunk_size
        self.flat_param._local_shard = full[start:end].clone()
        self.flat_param._is_sharded  = True
        self.flat_param.data = self.flat_param._local_shard  # free full tensor
        self._padded_size = padded_size
        self._chunk_size  = chunk_size

    def unshard(self):
        """All-gather the full parameter from all ranks before forward/backward."""
        if not self.flat_param._is_sharded:
            return  # already unsharded (e.g., NO_SHARD strategy)

        gathered = torch.empty(self._padded_size,
                               dtype=self.flat_param._local_shard.dtype,
                               device=self.flat_param._local_shard.device)
        dist.all_gather_into_tensor(gathered, self.flat_param._local_shard)

        self.flat_param._full_param = gathered[: self.flat_param._numels and sum(self.flat_param._numels)]
        self.flat_param.data = self.flat_param._full_param
        self._write_back_views()

    def reshard(self):
        """Discard the full parameter, keep only this rank's local shard."""
        if self.cfg.sharding_strategy == ShardingStrategy.NO_SHARD:
            return
        self.flat_param.data = self.flat_param._local_shard
        self.flat_param._full_param = None  # free memory

    def reduce_scatter_grad(self):
        """After backward: reduce-scatter gradients so each rank only keeps
        its shard of the gradient (ZeRO-2/3 behaviour)."""
        if self.flat_param.grad is None:
            return

        full_grad = self.flat_param.grad
        if full_grad.numel() < self._padded_size:
            pad = torch.zeros(self._padded_size - full_grad.numel(),
                              dtype=full_grad.dtype, device=full_grad.device)
            full_grad = torch.cat([full_grad.reshape(-1), pad])

        local_grad_shard = torch.empty(self._chunk_size,
                                       dtype=full_grad.dtype, device=full_grad.device)
        dist.reduce_scatter_tensor(local_grad_shard, full_grad)
        local_grad_shard /= self.world_size

        self.flat_param._local_shard.grad = local_grad_shard

    def _write_back_views(self):
        """Re-create views into the unsharded flat_param for each original
        parameter so forward() can use them transparently."""
        offset = 0
        for info, shape, numel in zip(
                self.flat_param._param_infos,
                self.flat_param._shapes,
                self.flat_param._numels):
            view = self.flat_param.data[offset: offset + numel].view(shape)
            setattr(info.module, info.param_name, nn.Parameter(view, requires_grad=True))
            offset += numel


# ─── FullyShardedDataParallel wrapper ─────────────────────────────────────────

class FullyShardedDataParallel(nn.Module):
    """
    Wraps a module for ZeRO-3 style sharded training.
    Recursively wraps submodules matching `auto_wrap_policy`.
    """

    def __init__(
        self,
        module: nn.Module,
        cfg: FSDPConfig = FSDPConfig(),
        auto_wrap_policy: Optional[Callable] = None,
        process_group: Optional[Any] = None,
    ):
        super().__init__()
        self.cfg          = cfg
        self.process_group = process_group or dist.group.WORLD
        self.rank         = dist.get_rank(self.process_group)
        self.world_size   = dist.get_world_size(self.process_group)

        policy = auto_wrap_policy or self._default_wrap_policy
        self._wrap_recursive(module, policy)

        self.module = module
        self._handles: List[FlatParamHandle] = []
        self._collect_handles(module)

        if cfg.sync_module_states:
            self._broadcast_initial_params()

        self._register_hooks()

    # ── Auto-wrap policy ────────────────────────────────────────────────────

    def _default_wrap_policy(self, module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        if recurse:
            return True
        return nonwrapped_numel >= self.cfg.min_num_params

    def _wrap_recursive(self, module: nn.Module, policy: Callable, prefix: str = ""):
        for name, child in module.named_children():
            numel = sum(p.numel() for p in child.parameters())
            if policy(child, False, numel):
                setattr(module, name, _FSDPUnit(child, self.cfg, self.rank, self.world_size))
            else:
                self._wrap_recursive(child, policy, f"{prefix}{name}.")

    def _collect_handles(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, _FSDPUnit):
                self._handles.append(m.handle)

    def _broadcast_initial_params(self):
        for handle in self._handles:
            dist.broadcast(handle.flat_param._local_shard, src=0, group=self.process_group)

    def _register_hooks(self):
        for handle in self._handles:
            handle.flat_param.register_hook(
                functools.partial(self._post_backward_hook, handle))

    def _post_backward_hook(self, handle: FlatParamHandle, grad):
        handle.reduce_scatter_grad()
        if self.cfg.cpu_offload:
            handle.flat_param._local_shard.grad = (
                handle.flat_param._local_shard.grad.cpu())
        return grad

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # ── Checkpoint (sharded state dict) ─────────────────────────────────────

    def state_dict(self, *args, **kwargs):
        """Gathers full parameters (rank 0 only) for checkpointing."""
        full_sd = {}
        for handle in self._handles:
            handle.unshard()
            for info, shape, numel, name in zip(
                    handle.flat_param._param_infos,
                    handle.flat_param._shapes,
                    handle.flat_param._numels,
                    handle.flat_param._names):
                if self.rank == 0:
                    full_sd[name] = getattr(info.module, info.param_name).detach().cpu()
            handle.reshard()
        return full_sd

    def clip_grad_norm_(self, max_norm: float) -> float:
        """Computes global grad norm across all shards (all-reduce required)."""
        local_norm_sq = 0.0
        for handle in self._handles:
            g = handle.flat_param._local_shard.grad
            if g is not None:
                local_norm_sq += g.pow(2).sum().item()

        total = torch.tensor(local_norm_sq, device="cuda")
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self.process_group)
        total_norm = total.sqrt().item()

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for handle in self._handles:
                g = handle.flat_param._local_shard.grad
                if g is not None:
                    g.mul_(clip_coef)
        return total_norm


class _FSDPUnit(nn.Module):
    """One FSDP-wrapped unit: unshard before forward, reshard after."""
    def __init__(self, module: nn.Module, cfg: FSDPConfig, rank: int, world_size: int):
        super().__init__()
        self.module = module
        params      = list(module.parameters())
        param_infos = [ParamInfo(module, n, type(module).__name__)
                       for n, _ in module.named_parameters()]
        self.handle = FlatParamHandle(params, param_infos, rank, world_size, cfg)

    def forward(self, *args, **kwargs):
        self.handle.unshard()
        out = self.module(*args, **kwargs)
        self.handle.reshard()
        return out


# ─── ZeRO Optimizer wrapper ───────────────────────────────────────────────────

class ZeroRedundancyOptimizer:
    """Shards optimiser state (m, v for Adam) across ranks (ZeRO stage-1)."""

    def __init__(self, params, optimizer_class: Type[torch.optim.Optimizer],
                 rank: int, world_size: int, **optim_kwargs):
        self.rank       = rank
        self.world_size = world_size
        params          = list(params)

        # Partition parameters round-robin across ranks
        self.shards: List[List[torch.nn.Parameter]] = [[] for _ in range(world_size)]
        for i, p in enumerate(params):
            self.shards[i % world_size].append(p)

        self.local_optimizer = optimizer_class(self.shards[rank], **optim_kwargs)
        self.all_params = params

    def step(self, closure=None):
        self.local_optimizer.step(closure)
        # broadcast updated params from owning rank to all others
        for r, shard in enumerate(self.shards):
            for p in shard:
                dist.broadcast(p.data, src=r)

    def zero_grad(self, set_to_none=True):
        for p in self.all_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
