"""Training profiler with Chrome trace export for PyGo."""
import time, json, threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ProfileEvent:
    name: str; start_us: float; end_us: float
    ms: float = 0.; meta: dict = field(default_factory=dict)

class Profiler:
    def __init__(self, enabled: bool = True):
        self._enabled = enabled; self._lock = threading.Lock()
        self._events: List[ProfileEvent] = []; self._active: Dict[str, float] = {}

    def start(self, name: str):
        if not self._enabled: return
        with self._lock: self._active[name] = time.perf_counter() * 1e6

    def end(self, name: str) -> float:
        if not self._enabled: return 0.
        now = time.perf_counter() * 1e6
        with self._lock:
            start = self._active.pop(name, now)
            ms = (now - start) / 1000.
            self._events.append(ProfileEvent(name, start, now, ms))
        return ms

    @contextmanager
    def profile(self, name: str):
        self.start(name)
        try: yield
        finally: self.end(name)

    def summary(self, k: int = 20) -> str:
        tots: Dict[str, float] = {}
        cnts: Dict[str, int] = {}
        with self._lock:
            for e in self._events:
                tots[e.name] = tots.get(e.name, 0) + e.ms
                cnts[e.name] = cnts.get(e.name, 0) + 1
        lines = ["PyGo Profiler", "=" * 50]
        for n, t in sorted(tots.items(), key=lambda x: -x[1])[:k]:
            lines.append(f"  {n:<40s} total={t:8.2f}ms  avg={t/cnts[n]:7.3f}ms  n={cnts[n]}")
        return "\n".join(lines)

    def export_chrome_trace(self, path: str):
        with self._lock:
            evs = [{"name": e.name, "ph": "X", "pid": 0, "tid": 0,
                    "ts": e.start_us, "dur": e.end_us - e.start_us,
                    "args": e.meta} for e in self._events]
        with open(path, "w") as f:
            json.dump({"traceEvents": evs}, f)

    def reset(self):
        with self._lock: self._events.clear(); self._active.clear()

    def enable(self):  self._enabled = True
    def disable(self): self._enabled = False
    def __len__(self): return len(self._events)

profiler = Profiler(enabled=False)
