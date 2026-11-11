"""Device abstraction for PyGo."""
from dataclasses import dataclass
import torch
from typing import Optional

@dataclass(frozen=True)
class Device:
    type:str; index:int=0
    @property
    def is_cuda(self): return self.type=="cuda"
    @property
    def is_cpu(self): return self.type=="cpu"
    @property
    def is_mps(self): return self.type=="mps"
    def to_torch(self): return torch.device(f"{self.type}:{self.index}" if self.is_cuda else self.type)
    def __str__(self): return f"{self.type}:{self.index}" if self.is_cuda else self.type

cpu=Device("cpu"); mps=Device("mps")
def cuda(i=0): return Device("cuda",i)
def xla(i=0):  return Device("xla",i)

class DeviceManager:
    @staticmethod
    def best():
        if torch.cuda.is_available(): return cuda(0)
        if torch.backends.mps.is_available(): return mps
        return cpu
    @staticmethod
    def all():
        devs=[cpu]
        if torch.cuda.is_available(): devs+=[cuda(i) for i in range(torch.cuda.device_count())]
        if torch.backends.mps.is_available(): devs.append(mps)
        return devs
    @staticmethod
    def mem(dev:Device):
        if dev.is_cuda:
            return {"allocated_mb":torch.cuda.memory_allocated(dev.index)/1e6,
                    "reserved_mb":torch.cuda.memory_reserved(dev.index)/1e6}
        return {}
    @staticmethod
    def sync(dev:Optional[Device]=None):
        if dev is None or (dev and dev.is_cuda): torch.cuda.synchronize()
