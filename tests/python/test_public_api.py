import importlib
import pygo


def test_public_package_imports():
    assert pygo.__version__ == "0.1.0"
    assert pygo.Tensor is not None
    assert callable(pygo.make_optimizer)


def test_subpackages_are_importable():
    for name in ("tensor", "nn", "optim", "autograd", "serialization", "distributed", "inference", "data"):
        assert importlib.import_module(f"pygo.{name}") is not None
