"""Python entry points for Cinder's C++ extension module."""

try:
    from .core import Tensor
except ImportError:
    __all__: list[str] = []
else:
    __all__ = ["Tensor"]
