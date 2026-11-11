"""PyGo Python package exports."""
from .._runtime import Tensor, Module, Parameter, functional, make_module, make_loss, make_optimizer, save, load
__all__ = ["Tensor", "Module", "Parameter", "functional", "make_module", "make_loss", "make_optimizer", "save", "load"]

def version():
    """Return the Python API version exposed by this package."""
    return "0.1.0"

def capabilities():
    """Report optional backend capabilities without importing heavyweight modules."""
    return {"torch": True, "autograd": True, "nn": True, "serialization": True}
