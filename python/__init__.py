"""PyGo: production Python entry point.

The package exposes the framework ecosystem while keeping heavyweight optional
backends lazy.  Native execution is preferred when a built PyGo shared library
is available; PyTorch remains an optional compatibility backend for modules
that explicitly request it.
"""
from .tensor import Tensor, Module, Parameter, functional, make_module, make_loss, make_optimizer, save, load
from ._runtime import train_step

__version__ = "0.1.0"
__all__ = ["Tensor", "Module", "Parameter", "functional", "make_module", "make_loss", "make_optimizer", "save", "load", "train_step", "__version__"]
