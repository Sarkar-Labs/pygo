"""LR schedulers for PyGo."""
import math
from typing import List

class LRScheduler:
    def __init__(self, opt, last_step=-1):
        self.optimizer=opt; self.last_step=last_step
        self.base_lrs=[g["lr"] for g in opt.param_groups]; self.step()
    def get_lr(self): raise NotImplementedError
    def step(self):
        self.last_step+=1; lrs=self.get_lr()
        for g,lr in zip(self.optimizer.param_groups,lrs): g["lr"]=lr
    def get_last_lr(self): return [g["lr"] for g in self.optimizer.param_groups]
    def state_dict(self): return {"last_step": self.last_step}
    def load_state_dict(self, sd): self.last_step=sd["last_step"]

class WarmupCosineDecay(LRScheduler):
    def __init__(self, opt, warmup_steps, total_steps, min_lr_ratio=0.1, last_step=-1):
        self.warmup=warmup_steps; self.total=total_steps; self.min_r=min_lr_ratio
        super().__init__(opt, last_step)
    def get_lr(self):
        s=self.last_step
        return [b*(s/max(self.warmup,1)) if s<self.warmup
                else b*(self.min_r+(1-self.min_r)*0.5*(1+math.cos(math.pi*(s-self.warmup)/max(self.total-self.warmup,1))))
                for b in self.base_lrs]

class OneCycleLR(LRScheduler):
    def __init__(self, opt, max_lr, total_steps, pct_start=0.3, div=25., final_div=1e4, last_step=-1):
        self.max_lr=max_lr; self.total=total_steps; self.pct=pct_start
        self.div=div; self.fdiv=final_div; self.ws=int(pct_start*total_steps)
        super().__init__(opt, last_step)
    def _cos(self, s, e, p): return e+(s-e)/2*(1+math.cos(math.pi*p))
    def get_lr(self):
        s=self.last_step; init=self.max_lr/self.div; mn=init/self.fdiv
        return [self._cos(init,self.max_lr,s/max(self.ws,1)) if s<=self.ws
                else self._cos(self.max_lr,mn,(s-self.ws)/max(self.total-self.ws,1))
                for _ in self.base_lrs]

class NoamLR(LRScheduler):
    def __init__(self, opt, d_model, warmup=4000, last_step=-1):
        self.d=d_model; self.w=warmup; super().__init__(opt, last_step)
    def get_lr(self):
        s=max(self.last_step,1)
        return [self.d**-0.5 * min(s**-0.5, s*self.w**-1.5) for _ in self.base_lrs]
