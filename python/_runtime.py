"""Shared PyGo Python runtime backed by PyTorch."""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as functional
Tensor=torch.Tensor
Module=nn.Module
Parameter=nn.Parameter

def make_module(factory,*args,**kwargs): return factory(*args,**kwargs)
def make_loss(name="mse",**kwargs):
    table={"mse":nn.MSELoss,"l1":nn.L1Loss,"cross_entropy":nn.CrossEntropyLoss,"bce":nn.BCELoss,"bce_logits":nn.BCEWithLogitsLoss}
    return table.get(name,nn.MSELoss)(**kwargs)
def make_optimizer(params,name="adam",lr=1e-3,**kwargs):
    table={"adam":torch.optim.Adam,"adamw":torch.optim.AdamW,"sgd":torch.optim.SGD,"rmsprop":torch.optim.RMSprop}
    return table.get(name,torch.optim.Adam)(params,lr=lr,**kwargs)
def save(obj,path,**kwargs): Path(path).parent.mkdir(parents=True,exist_ok=True); return torch.save(obj,path,**kwargs)
def load(path,map_location="cpu",**kwargs): return torch.load(path,map_location=map_location,**kwargs)

def train_step(model,batch,optimizer,loss_fn):
    model.train(); optimizer.zero_grad(set_to_none=True); x,y=batch; loss=loss_fn(model(x),y); loss.backward(); optimizer.step(); return loss.detach()
