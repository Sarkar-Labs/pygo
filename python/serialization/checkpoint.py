"""Model checkpoint save/load utilities for PyGo."""
import os
import json
import time
import glob
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, optimizer=None, scheduler=None,
                    step: int = 0, epoch: int = 0, loss: float = 0.0,
                    path: str = "checkpoint.pt", metadata: Dict[str,Any] = None):
    """Save full training checkpoint."""
    state = {
        "model":     model.state_dict(),
        "step":      step,
        "epoch":     epoch,
        "loss":      loss,
        "timestamp": time.time(),
        "metadata":  metadata or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, model: nn.Module, optimizer=None,
                    scheduler=None, device="cpu") -> Dict[str,Any]:
    """Load checkpoint and restore model/optimizer state."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return {k: v for k, v in state.items() if k not in ("model","optimizer","scheduler")}


class CheckpointManager:
    """Manages multiple checkpoints with rotation and best-model tracking."""
    def __init__(self, directory: str, max_to_keep: int = 3,
                 metric: str = "eval_loss", mode: str = "min"):
        self.directory   = Path(directory)
        self.max_to_keep = max_to_keep
        self.metric      = metric
        self.mode        = mode
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints: list = []
        self.best_path: Optional[str] = None
        self.best_value: Optional[float] = None
        self._load_index()

    def _index_path(self): return self.directory / "checkpoint_index.json"

    def _load_index(self):
        if self._index_path().exists():
            with open(self._index_path()) as f:
                data = json.load(f)
            self.checkpoints = data.get("checkpoints", [])
            self.best_path   = data.get("best_path")
            self.best_value  = data.get("best_value")

    def _save_index(self):
        with open(self._index_path(), "w") as f:
            json.dump({"checkpoints": self.checkpoints,
                      "best_path": self.best_path,
                      "best_value": self.best_value}, f, indent=2)

    def save(self, model, step: int, metrics: Dict[str,float] = None, **kwargs) -> str:
        path = str(self.directory / f"checkpoint-{step}.pt")
        save_checkpoint(model, path=path, step=step, metadata=metrics or {}, **kwargs)
        self.checkpoints.append(path)

        # Check if best
        if metrics and self.metric in metrics:
            val = metrics[self.metric]
            is_better = (self.best_value is None or
                        (self.mode == "min" and val < self.best_value) or
                        (self.mode == "max" and val > self.best_value))
            if is_better:
                self.best_value = val
                self.best_path  = str(self.directory / "best_model.pt")
                shutil.copy2(path, self.best_path)

        # Rotate old checkpoints
        while len(self.checkpoints) > self.max_to_keep:
            old = self.checkpoints.pop(0)
            if os.path.exists(old) and old != self.best_path:
                os.remove(old)

        self._save_index()
        return path

    def latest(self) -> Optional[str]:
        return self.checkpoints[-1] if self.checkpoints else None

    def best(self) -> Optional[str]:
        return self.best_path
