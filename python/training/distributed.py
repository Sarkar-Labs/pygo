"""Training utility: distributed."""
from __future__ import annotations
import torch

class Distributed:
    """Reusable training helper with explicit train/eval phases."""
    def __init__(self, model=None, optimizer=None, loss_fn=None):
        self.model=model; self.optimizer=optimizer; self.loss_fn=loss_fn
    def step(self, batch):
        if self.model is None or self.optimizer is None or self.loss_fn is None: raise RuntimeError("model, optimizer and loss_fn are required")
        x,y=batch; self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        loss=self.loss_fn(self.model(x),y); loss.backward(); self.optimizer.step(); return float(loss.detach())
    def evaluate(self, loader):
        if self.model is None or self.loss_fn is None: raise RuntimeError("model and loss_fn are required")
        self.model.eval(); total=0.0; count=0
        with torch.no_grad():
            for x,y in loader: total += float(self.loss_fn(self.model(x),y)); count += 1
        return total/max(count,1)

def run(model, loader, optimizer, loss_fn, epochs=1):
    trainer=Distributed(model,optimizer,loss_fn); history=[]
    for _ in range(epochs):
        history.append(sum(trainer.step(b) for b in loader)/max(len(loader),1))
    return history
