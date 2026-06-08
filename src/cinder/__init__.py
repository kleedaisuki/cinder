"""Python entry points for Cinder's C++ extension module."""

from .core import add

try:
    from .core import Tensor, divide, multiply, subtract
except ImportError:
    __all__ = ["add"]
else:
    __all__ = ["Tensor", "add", "divide", "multiply", "subtract"]
