"""Data utility: cifar."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader

class Cifar(Dataset):
    def __init__(self, data=None, targets=None, transform=None):
        self.data = data if data is not None else []
        self.targets = targets; self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, index):
        x=self.data[index]; y=self.targets[index] if self.targets is not None else None
        if self.transform is not None: x=self.transform(x)
        return (x,y) if y is not None else x

def loader(data, batch_size=32, shuffle=True, **kwargs):
    ds=data if isinstance(data, Dataset) else Cifar(data)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,**kwargs)
