"""Advanced convolution modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch); self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x): return F.relu(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, expand=6):
        super().__init__(); hidden = in_ch * expand; self.use_skip = s==1 and in_ch==out_ch
        layers = []
        if expand != 1: layers += [nn.Conv2d(in_ch,hidden,1,bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(True)]
        layers += [nn.Conv2d(hidden,hidden,3,s,1,groups=hidden,bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(True),
                   nn.Conv2d(hidden,out_ch,1,bias=False), nn.BatchNorm2d(out_ch)]
        self.conv = nn.Sequential(*layers)
    def forward(self, x): return x + self.conv(x) if self.use_skip else self.conv(x)

class FusedMBConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, expand=4):
        super().__init__(); hidden = in_ch * expand; self.use_skip = s==1 and in_ch==out_ch
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch,hidden,k,s,k//2,bias=False), nn.BatchNorm2d(hidden), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv2d(hidden,out_ch,1,bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x): out = self.conv2(self.conv1(x)); return x+out if self.use_skip else out

class SEBlock(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__(); r = max(1, ch//ratio)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch,r), nn.ReLU(), nn.Linear(r,ch), nn.Sigmoid())
    def forward(self, x): return x * self.se(x).view(x.shape[0], -1, 1, 1)