"""LoRA: Low-Rank Adaptation for efficient fine-tuning."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Dict

class LoRALinear(nn.Module):
    def __init__(self,in_f,out_f,r=4,alpha=1.,dropout=0.,bias=True):
        super().__init__(); self.r=r; self.scale=alpha/r
        self.weight=nn.Parameter(torch.empty(out_f,in_f),requires_grad=False)
        self.bias_=nn.Parameter(torch.zeros(out_f),requires_grad=bias)
        self.lora_A=nn.Parameter(torch.empty(r,in_f))
        self.lora_B=nn.Parameter(torch.zeros(out_f,r))
        self.drop=nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):
        return F.linear(x,self.weight,self.bias_)+F.linear(F.linear(self.drop(x),self.lora_A),self.lora_B)*self.scale
    def merge(self):
        m=nn.Linear(self.lora_A.shape[1],self.weight.shape[0],bias=self.bias_ is not None)
        m.weight.data=self.weight+(self.lora_B@self.lora_A)*self.scale
        if self.bias_ is not None: m.bias.data=self.bias_
        return m

def inject_lora(model,targets=None,r=16,alpha=32.,dropout=0.05):
    targets=targets or ["q_proj","v_proj","k_proj","o_proj"]
    for p in model.parameters(): p.requires_grad_(False)
    def _inject(parent,prefix=""):
        for name,child in list(parent.named_children()):
            full=f"{prefix}.{name}" if prefix else name
            if isinstance(child,nn.Linear) and any(t in full for t in targets):
                lo=LoRALinear(child.in_features,child.out_features,r,alpha,dropout,child.bias is not None)
                lo.weight.data=child.weight.data.clone()
                if child.bias is not None: lo.bias_.data=child.bias.data.clone()
                setattr(parent,name,lo)
            else: _inject(child,full)
    _inject(model); return model

def lora_params(model): return [p for n,p in model.named_parameters() if p.requires_grad]
def param_count(model) -> Dict[str,int]:
    t=sum(p.numel() for p in model.parameters())
    tr=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total":t,"trainable":tr,"pct":100*tr/max(t,1)}
