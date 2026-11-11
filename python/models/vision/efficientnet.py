"""EfficientNet models — vision classification."""
from python.nn.efficientnet import EfficientNet

def efficientnet_b0(nc=1000): return EfficientNet("b0", nc)
def efficientnet_b1(nc=1000): return EfficientNet("b1", nc)
def efficientnet_b2(nc=1000): return EfficientNet("b2", nc)
def efficientnet_b3(nc=1000): return EfficientNet("b3", nc)
def efficientnet_b4(nc=1000): return EfficientNet("b4", nc)
def efficientnet_b5(nc=1000): return EfficientNet("b5", nc)
def efficientnet_b6(nc=1000): return EfficientNet("b6", nc)
def efficientnet_b7(nc=1000): return EfficientNet("b7", nc)

__all__ = [f"efficientnet_b{i}" for i in range(8)]
