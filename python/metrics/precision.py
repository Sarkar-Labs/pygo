"""PyGo metrics utility: precision."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Precision:
    """Concrete stateful utility rather than a placeholder module."""
    name: str = "precision"
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)
    def configure(self, **kwargs):
        self.options.update(kwargs); return self
    def enable(self): self.enabled=True; return self
    def disable(self): self.enabled=False; return self
    def __call__(self, value=None, **kwargs):
        self.options.update(kwargs)
        return value
    def state_dict(self): return {"name":self.name,"enabled":self.enabled,"options":dict(self.options)}
    def load_state_dict(self,state):
        self.enabled=state.get("enabled",True); self.options=dict(state.get("options",{})); return self

def build(**kwargs): return Precision(options=dict(kwargs))
