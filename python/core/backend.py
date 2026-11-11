"""Runtime and backend configuration for PyGo."""
import os, torch
from enum import Enum
from typing import Optional

class Backend(Enum):
    PYTORCH="pytorch"; PYGO_CPP="pygo_cpp"; XLA="xla"; ONNX="onnx"

class Runtime:
    _backend=Backend.PYTORCH; _device=torch.device("cpu")
    _dtype=torch.float32; _threads=os.cpu_count() or 4; _seed=42

    @classmethod
    def set_backend(cls,b): cls._backend=b
    @classmethod
    def set_device(cls,d): cls._device=torch.device(d) if isinstance(d,str) else d
    @classmethod
    def set_dtype(cls,dt): cls._dtype=dt; torch.set_default_dtype(dt)
    @classmethod
    def set_seed(cls,s):
        cls._seed=s; torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
        try: import numpy as np; np.random.seed(s)
        except: pass
    @classmethod
    def set_threads(cls,n): cls._threads=n; torch.set_num_threads(n)
    @classmethod
    def set_deterministic(cls,v): torch.use_deterministic_algorithms(v)
    @classmethod
    def info(cls):
        d={"backend":cls._backend.value,"device":str(cls._device),"dtype":str(cls._dtype),
           "cuda":torch.cuda.is_available(),"mps":torch.backends.mps.is_available(),
           "torch":torch.__version__}
        if torch.cuda.is_available():
            d["cuda_ver"]=torch.version.cuda; d["n_gpus"]=torch.cuda.device_count()
        return d
    @classmethod
    def print_info(cls):
        for k,v in cls.info().items(): print(f"  {k:20s}: {v}")

def setup(seed=42,device="auto",dtype=torch.float32,deterministic=False,threads=None):
    Runtime.set_seed(seed)
    if device=="auto":
        device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    Runtime.set_device(device); Runtime.set_dtype(dtype); Runtime.set_deterministic(deterministic)
    if threads: Runtime.set_threads(threads)
    return Runtime._device
