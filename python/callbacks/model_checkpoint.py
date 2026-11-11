"""Callbacks component: model_checkpoint."""
from __future__ import annotations
import time

class ModelCheckpoint:
    def __init__(self, **kwargs):
        self.options=dict(kwargs); self.events=[]; self.started=None
    def on_start(self, **state):
        self.started=time.perf_counter(); self.events.append(("start",state)); return state
    def on_step(self, **state):
        self.events.append(("step",state)); return state
    def on_end(self, **state):
        state=dict(state); state.setdefault("elapsed", time.perf_counter()-self.started if self.started else 0.0)
        self.events.append(("end",state)); return state
    def __call__(self, event="step", **state):
        return getattr(self, "on_"+event, self.on_step)(**state)
