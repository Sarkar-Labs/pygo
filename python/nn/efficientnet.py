"""EfficientNet B0-B7 — compound scaled MobileNetV3 blocks."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass

PARAMS = {"b0":(1.,1.,224,.2),"b1":(1.,1.1,240,.2),"b2":(1.1,1.2,260,.3),
          "b3":(1.2,1.4,300,.3),"b4":(1.4,1.8,380,.4),"b5":(1.6,2.2,456,.4),
          "b6":(1.8,2.6,528,.5),"b7":(2.,3.1,600,.5)}

class SEBlock(nn.Module):
    def __init__(self,c,r): super().__init__(); self.se=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,r,1),nn.SiLU(),nn.Conv2d(r,c,1),nn.Sigmoid())
    def forward(self,x): return x*self.se(x)

class MBConv(nn.Module):
    def __init__(self,ic,oc,k,s,er,se_r,drop_p=0.):
        super().__init__(); self.skip=s==1 and ic==oc; self.drop_p=drop_p
        hc=ic*er; layers=[]
        if er!=1: layers+=[nn.Conv2d(ic,hc,1,bias=False),nn.BatchNorm2d(hc),nn.SiLU()]
        layers+=[nn.Conv2d(hc,hc,k,stride=s,padding=k//2,groups=hc,bias=False),nn.BatchNorm2d(hc),nn.SiLU()]
        if se_r>0: layers.append(SEBlock(hc,max(1,int(ic*se_r))))
        layers+=[nn.Conv2d(hc,oc,1,bias=False),nn.BatchNorm2d(oc)]
        self.block=nn.Sequential(*layers)
    def forward(self,x):
        out=self.block(x)
        if self.skip:
            if self.training and self.drop_p>0:
                mask=torch.rand(x.shape[0],1,1,1,device=x.device)>self.drop_p
                return out*mask/max(1-self.drop_p,1e-8)+x
            return out+x
        return out

class EfficientNet(nn.Module):
    BASE=[(3,32,16,1,1,1,.25),(3,16,24,6,2,2,.25),(5,24,40,6,2,2,.25),
          (3,40,80,6,2,3,.25),(5,80,112,6,1,3,.25),(5,112,192,6,2,4,.25),(3,192,320,6,1,1,.25)]
    def __init__(self, v="b0", nc=1000):
        super().__init__(); w,d,_,dr=PARAMS[v]
        sc=lambda c: int(math.ceil(c*w/8))*8; sl=lambda n: int(math.ceil(n*d))
        self.stem=nn.Sequential(nn.Conv2d(3,sc(32),3,2,1,bias=False),nn.BatchNorm2d(sc(32)),nn.SiLU())
        blocks=[]; ic=sc(32)
        n_total=sum(sl(b[5]) for b in self.BASE); drop_i=0
        for k,cin,cout,er,s,n,se_r in self.BASE:
            n=sl(n)
            for i in range(n):
                oc=sc(cout); s_=s if i==0 else 1; dp=dr*drop_i/max(n_total-1,1)
                blocks.append(MBConv(ic,oc,k,s_,er,se_r,dp)); ic=oc; drop_i+=1
        self.blocks=nn.Sequential(*blocks)
        hc=sc(1280)
        self.head=nn.Sequential(nn.Conv2d(ic,hc,1,bias=False),nn.BatchNorm2d(hc),nn.SiLU())
        self.pool=nn.AdaptiveAvgPool2d(1); self.drop=nn.Dropout(dr); self.fc=nn.Linear(hc,nc)
    def forward(self,x):
        x=self.stem(x); x=self.blocks(x); x=self.head(x)
        return self.fc(self.drop(self.pool(x).flatten(1)))

def efficientnet_b0(nc=1000): return EfficientNet("b0",nc)
def efficientnet_b4(nc=1000): return EfficientNet("b4",nc)
def efficientnet_b7(nc=1000): return EfficientNet("b7",nc)
