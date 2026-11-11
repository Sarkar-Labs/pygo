"""Mixed precision (AMP) training for PyGo."""
import torch, torch.nn as nn, torch.nn.functional as F
from contextlib import contextmanager

class GradScaler:
    def __init__(self, init_scale=65536., growth=2., backoff=0.5, interval=2000, enabled=True):
        self.scale=init_scale; self.growth=growth; self.backoff=backoff
        self.interval=interval; self.enabled=enabled; self._tracker=0
    def scale_loss(self, loss): return loss*self.scale if self.enabled else loss
    def step(self, opt) -> bool:
        if not self.enabled: opt.step(); return True
        inv = 1./self.scale; found_inf = False
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                p.grad.mul_(inv)
                if not torch.isfinite(p.grad).all(): found_inf=True; break
        if found_inf: self.scale*=self.backoff; self._tracker=0; return False
        opt.step(); self._tracker+=1
        if self._tracker>=self.interval: self.scale*=self.growth; self._tracker=0
        return True
    def update(self): pass
    def state_dict(self): return {"scale": self.scale}
    def load_state_dict(self, sd): self.scale = sd.get("scale", self.scale)

@contextmanager
def autocast(enabled=True, dtype=torch.float16):
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype): yield
    else: yield
