"""Runnable example module for ``config``."""

from __future__ import annotations
from dataclasses import dataclass, asdict
import argparse
import math

@dataclass
class Config:
    name: str = "config"
    steps: int = 10
    learning_rate: float = 1e-3

def run(config: Config) -> dict[str, float | int | str]:
    """Run a deterministic miniature workload that exercises the example contract."""
    value = 1.0
    for step in range(max(0, config.steps)):
        gradient = 2.0 * value
        value -= config.learning_rate * gradient
    return {"name": config.name, "steps": config.steps, "final_value": value}

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PyGo config example")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    print(run(Config(steps=args.steps, learning_rate=args.lr)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
