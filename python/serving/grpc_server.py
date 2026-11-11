"""gRPC/REST model serving for PyGo."""
import time, json, queue, threading, logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List

log = logging.getLogger("pygo.serve")

@dataclass
class Request:
    request_id: str; model_name: str; model_version: str="latest"
    inputs: Dict[str,Any]=field(default_factory=dict)
    params: Dict[str,Any]=field(default_factory=dict); timeout: float=30.0

@dataclass
class Response:
    request_id: str; outputs: Dict[str,Any]=field(default_factory=dict)
    latency_ms: float=0.; error: Optional[str]=None

class DynamicBatcher:
    def __init__(self, max_batch=32, timeout_ms=10.):
        self.max_batch=max_batch; self.timeout=timeout_ms/1000.
        self._q: queue.Queue=queue.Queue(); self._res: Dict[str,queue.Queue]={}
        self._lock=threading.Lock(); self._running=True
        threading.Thread(target=self._loop, daemon=True).start()
    def submit(self, req: Request) -> Response:
        rq: queue.Queue=queue.Queue(1)
        with self._lock: self._res[req.request_id]=rq
        self._q.put(req)
        try: return rq.get(timeout=req.timeout)
        except queue.Empty: return Response(req.request_id, error="timeout")
    def _loop(self):
        while self._running:
            batch=[]
            try: batch.append(self._q.get(timeout=self.timeout))
            except queue.Empty: continue
            while len(batch)<self.max_batch:
                try: batch.append(self._q.get_nowait())
                except queue.Empty: break
            for req in batch:
                start=time.time()
                resp=Response(req.request_id,outputs={"predictions":[]},latency_ms=(time.time()-start)*1000)
                with self._lock: q=self._res.pop(req.request_id,None)
                if q: q.put(resp)
    def stop(self): self._running=False

class PyGoServer:
    def __init__(self, max_workers=4, max_batch=32, timeout_ms=10.):
        self._models: Dict[str,Callable]={}; self._lock=threading.Lock()
        self.batcher=DynamicBatcher(max_batch, timeout_ms)
        self._stats={"n":0,"err":0,"ms":0.}
    def register(self, name, version="v1"):
        def deco(fn):
            with self._lock: self._models[f"{name}:{version}"]=fn
            return fn
        return deco
    def predict(self, req: Request) -> Response:
        start=time.time()
        with self._lock: fn=self._models.get(f"{req.model_name}:{req.model_version}")
        if not fn: return Response(req.request_id, error="model not found")
        try:
            out=fn(req.inputs, **req.params)
            r=Response(req.request_id,outputs=out,latency_ms=(time.time()-start)*1000)
        except Exception as e:
            r=Response(req.request_id, error=str(e))
        with self._lock:
            self._stats["n"]+=1; self._stats["ms"]+=r.latency_ms
            if r.error: self._stats["err"]+=1
        return r
    def health(self):
        n=self._stats["n"]
        return {"status":"healthy","requests":n,"avg_ms":self._stats["ms"]/max(n,1)}
    def shutdown(self): self.batcher.stop()
