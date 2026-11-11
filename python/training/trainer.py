"""Full-featured PyGo Trainer."""
import os, time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any

@dataclass
class TrainerConfig:
    max_epochs: int = 10; max_steps: int = -1; log_every: int = 100
    eval_every: int = 500; save_every: int = 1000; grad_clip: float = 1.
    grad_accum: int = 1; bf16: bool = False; fp16: bool = False
    output_dir: str = "./output"; metric_for_best: str = "eval_loss"
    load_best_at_end: bool = True; report_to: str = "none"

class Trainer:
    def __init__(self, model: nn.Module, optimizer, loss_fn: Callable,
                 cfg: TrainerConfig = None, scheduler=None, callbacks=None):
        self.model = model; self.optimizer = optimizer; self.loss_fn = loss_fn
        self.cfg = cfg or TrainerConfig(); self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.global_step = 0; self.best_metric = float("inf")
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def compute_loss(self, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            outputs = self.model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            return self.loss_fn(logits, targets)
        raise ValueError("batch must be (inputs, targets)")

    def training_step(self, batch) -> float:
        self.model.train()
        loss = self.compute_loss(batch) / self.cfg.grad_accum
        loss.backward()
        return loss.item() * self.cfg.grad_accum

    def optimizer_step(self):
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        if self.scheduler: self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    def fit(self, train_loader, eval_loader=None):
        print(f"Training: {self.cfg.max_epochs} epochs | device: {next(self.model.parameters()).device}")
        self.optimizer.zero_grad()
        for epoch in range(self.cfg.max_epochs):
            t0 = time.time(); total_loss = 0.; n = 0
            for step, batch in enumerate(train_loader):
                loss = self.training_step(batch); total_loss += loss; n += 1
                if (step + 1) % self.cfg.grad_accum == 0:
                    self.optimizer_step()
                if self.cfg.log_every > 0 and self.global_step % self.cfg.log_every == 0:
                    print(f"  step={self.global_step} loss={loss:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}")
                if self.cfg.max_steps > 0 and self.global_step >= self.cfg.max_steps: break
            avg = total_loss / max(n, 1)
            print(f"Epoch {epoch+1}/{self.cfg.max_epochs} loss={avg:.4f} time={time.time()-t0:.1f}s")
        print("Training complete!")