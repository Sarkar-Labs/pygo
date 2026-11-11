"""RLHF PPO trainer for language models."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Dict

@dataclass
class RLHFConfig:
    kl_coeff:float=0.1; clip_ratio:float=0.2; vf_coeff:float=0.1
    entropy_coeff:float=0.01; gamma:float=1.0; lam:float=0.95
    ppo_epochs:int=4; mini_batch:int=8; normalize_adv:bool=True
    max_grad_norm:float=1.0; reward_clip:float=10.0

class RewardModel(nn.Module):
    def __init__(self, backbone, d_model):
        super().__init__(); self.backbone=backbone
        self.head=nn.Sequential(nn.Linear(d_model,d_model//2),nn.GELU(),nn.Linear(d_model//2,1))
    def forward(self, ids, mask=None):
        out=self.backbone(ids,attention_mask=mask)
        h=out["last_hidden_state"] if isinstance(out,dict) else out
        lens=(mask.sum(-1)-1) if mask is not None else torch.full((h.size(0),),h.size(1)-1,device=h.device)
        return self.head(h[torch.arange(h.size(0),device=h.device),lens]).squeeze(-1)
    def loss(self, chosen, rejected):
        return -F.logsigmoid(self(chosen)-self(rejected)).mean()

class PPOTrainer:
    def __init__(self, policy, ref_policy, reward_model, value_model, optimizer, cfg=None):
        self.policy=policy; self.ref=ref_policy; self.rm=reward_model
        self.vm=value_model; self.opt=optimizer; self.cfg=cfg or RLHFConfig()
    @torch.no_grad()
    def gae(self, r, v, masks):
        B,T=r.shape; adv=torch.zeros_like(r); last=0
        for t in reversed(range(T)):
            nv=v[:,t+1] if t<T-1 else torch.zeros(B,device=r.device)
            delta=r[:,t]+self.cfg.gamma*nv*masks[:,t]-v[:,t]
            last=delta+self.cfg.gamma*self.cfg.lam*masks[:,t]*last; adv[:,t]=last
        ret=adv+v
        if self.cfg.normalize_adv: adv=(adv-adv.mean())/(adv.std()+1e-8)
        return adv,ret
    def step(self, ids, old_lp, adv, ret) -> Dict[str,float]:
        out=self.policy(ids); logits=out["logits"] if isinstance(out,dict) else out
        lp=F.log_softmax(logits,-1).gather(-1,ids.unsqueeze(-1)).squeeze(-1)
        v=self.vm(ids); ratio=torch.exp(lp-old_lp)
        surr=torch.min(ratio*adv,torch.clamp(ratio,1-self.cfg.clip_ratio,1+self.cfg.clip_ratio)*adv)
        pl=-surr.mean(); vl=F.mse_loss(v,ret)
        ent=-(F.softmax(logits,-1)*F.log_softmax(logits,-1)).sum(-1).mean()
        loss=pl+self.cfg.vf_coeff*vl-self.cfg.entropy_coeff*ent
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.max_grad_norm)
        self.opt.step()
        return {"policy_loss":pl.item(),"value_loss":vl.item(),"entropy":ent.item()}
