"""Sparse layers and Mixture of Experts for PyGo."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

class SparseLinear(nn.Module):
    def __init__(self, in_f, out_f, sparsity=0.9, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f,in_f)*0.01)
        self.mask   = nn.Parameter((torch.rand(out_f,in_f)>sparsity).float(), requires_grad=False)
        self.bias_  = nn.Parameter(torch.zeros(out_f)) if bias else None
    def forward(self, x): return F.linear(x, self.weight*self.mask, self.bias_)
    def prune(self, s):
        with torch.no_grad():
            thr=torch.quantile(self.weight.abs(),s); self.mask.data=(self.weight.abs()>thr).float()

class MixtureOfExperts(nn.Module):
    def __init__(self, d, d_ff, n_experts, top_k=2, dropout=0., lb_coeff=0.01):
        super().__init__(); self.n_experts=n_experts; self.top_k=top_k; self.lb=lb_coeff
        self.gate       = nn.Linear(d, n_experts, bias=False)
        self.expert_w1  = nn.Parameter(torch.randn(n_experts,d,d_ff)*0.01)
        self.expert_w2  = nn.Parameter(torch.randn(n_experts,d_ff,d)*0.01)
        self.drop       = nn.Dropout(dropout)
    def forward(self, x: Tensor):
        B,S,D=x.shape; xf=x.view(-1,D); N=xf.shape[0]
        rp=F.softmax(self.gate(xf),dim=-1)
        tv,ti=torch.topk(rp,self.top_k,dim=-1)
        tw=tv/(tv.sum(-1,keepdim=True)+1e-8)
        out=torch.zeros_like(xf)
        for k in range(self.top_k):
            ei=ti[:,k]; w=tw[:,k]
            for e in range(self.n_experts):
                m=(ei==e)
                if not m.any(): continue
                h=F.silu(xf[m]@self.expert_w1[e])
                out[m]+=w[m,None]*(self.drop(h)@self.expert_w2[e])
        tpe=torch.zeros(self.n_experts,device=x.device)
        for e in range(self.n_experts): tpe[e]=(ti==e).float().sum()
        tpe/=(N*self.top_k)
        lb=self.lb*self.n_experts*(rp.mean(0)*tpe).sum()
        return out.view(B,S,D), lb
