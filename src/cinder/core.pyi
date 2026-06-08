from __future__ import annotations

from collections.abc import Sequence
from typing import overload


class Tensor:
    """Host-owned CUDA dense tensor with float32 storage.

    Tensor owns a single packed CUDA allocation containing metadata and data.
    Python exposes only the high-level tensor object: construction, shape
    inspection, host readback, elementwise arithmetic operators, tensor product,
    and contraction.
    """

    @overload
    def __init__(self, shape: Sequence[int]) -> None:
        """Create a zero-initialized tensor with dense row-major layout."""
        ...

    @overload
    def __init__(self, shape: Sequence[int], values: Sequence[float]) -> None:
        """Create a tensor from dense row-major host values."""
        ...

    @property
    def shape(self) -> list[int]:
        """Per-axis extents copied from the host-owned tensor metadata."""
        ...

    @property
    def rank(self) -> int:
        """Number of tensor axes."""
        ...

    @property
    def size(self) -> int:
        """Total number of dense elements."""
        ...

    def __len__(self) -> int:
        """Return ``size``."""
        ...

    def to_list(self) -> list[float]:
        """Copy dense row-major tensor data from CUDA device memory to Python."""
        ...

    def tensor_product(self, other: Tensor) -> Tensor:
        """Return the tensor product of ``self`` and ``other``."""
        ...

    def contract(self, other: Tensor, axes: Sequence[int], other_axes: Sequence[int]) -> Tensor:
        """Return the tensor contraction over paired axes."""
        ...

    def __add__(self, other: Tensor) -> Tensor:
        """Return elementwise ``self + other``."""
        ...

    def __sub__(self, other: Tensor) -> Tensor:
        """Return elementwise ``self - other``."""
        ...

    def __mul__(self, other: Tensor) -> Tensor:
        """Return elementwise ``self * other``."""
        ...

    def __truediv__(self, other: Tensor) -> Tensor:
        """Return elementwise ``self / other``."""
        ...

    def __repr__(self) -> str:
        """Return a compact debugging representation."""
        ...
