from __future__ import annotations

from collections.abc import Sequence
from typing import overload


class Tensor:
    """Host-owned CUDA dense tensor with float32 storage.

    Tensor owns a single packed CUDA allocation containing metadata and data.
    Python exposes only the high-level tensor object: construction, shape
    inspection, host readback, elementwise arithmetic operators, tensor product,
    transpose, contraction, mode multiplication, inner products, and norms.
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

    def reshape(self, shape: Sequence[int]) -> Tensor:
        """Return a dense tensor with ``shape`` and the same row-major data order."""
        ...

    def broadcast(self, shape: Sequence[int]) -> Tensor:
        """Return a dense tensor broadcast to ``shape``."""
        ...

    def slice(self, starts: Sequence[int], shape: Sequence[int]) -> Tensor:
        """Return a dense slice copy starting at ``starts`` with ``shape`` extents."""
        ...

    @overload
    def concat(self, other: Tensor, axis: int) -> Tensor:
        """Concatenate ``self`` and ``other`` along ``axis``."""
        ...

    @overload
    def concat(self, other: Sequence[Tensor], axis: int) -> Tensor:
        """Concatenate ``self`` and a sequence of tensors along ``axis``."""
        ...

    def tensor_product(self, other: Tensor) -> Tensor:
        """Return the tensor product of ``self`` and ``other``."""
        ...

    @overload
    def transpose(self) -> Tensor:
        """Return a transposed tensor with all axes reversed."""
        ...

    @overload
    def transpose(self, axes: Sequence[int]) -> Tensor:
        """Return a transposed tensor using the given axis permutation."""
        ...

    def contract(self, other: Tensor, axes: Sequence[int], other_axes: Sequence[int]) -> Tensor:
        """Return the tensor contraction over paired axes."""
        ...

    def mode_multiply(self, matrix: Tensor, mode: int) -> Tensor:
        """Return the mode multiplication of ``self`` by ``matrix`` along ``mode``."""
        ...

    def inner(self, other: Tensor) -> Tensor:
        """Return the inner product of ``self`` and ``other`` as a rank-0 tensor."""
        ...

    def dot(self, other: Tensor) -> Tensor:
        """Return the dot-product alias of ``inner`` as a rank-0 tensor."""
        ...

    def norm(self) -> Tensor:
        """Return the L2 norm of ``self`` as a rank-0 tensor."""
        ...

    @overload
    def __add__(self, other: Tensor) -> Tensor:
        """Return elementwise ``self + other``."""
        ...

    @overload
    def __add__(self, other: float) -> Tensor:
        """Return elementwise ``self + other``."""
        ...

    def __radd__(self, other: float) -> Tensor:
        """Return elementwise ``other + self``."""
        ...

    @overload
    def __sub__(self, other: Tensor) -> Tensor:
        """Return elementwise ``self - other``."""
        ...

    @overload
    def __sub__(self, other: float) -> Tensor:
        """Return elementwise ``self - other``."""
        ...

    def __rsub__(self, other: float) -> Tensor:
        """Return elementwise ``other - self``."""
        ...

    @overload
    def __mul__(self, other: Tensor) -> Tensor:
        """Return elementwise ``self * other``."""
        ...

    @overload
    def __mul__(self, other: float) -> Tensor:
        """Return elementwise ``self * other``."""
        ...

    def __rmul__(self, other: float) -> Tensor:
        """Return elementwise ``other * self``."""
        ...

    @overload
    def __truediv__(self, other: Tensor) -> Tensor:
        """Return elementwise ``self / other``."""
        ...

    @overload
    def __truediv__(self, other: float) -> Tensor:
        """Return elementwise ``self / other``."""
        ...

    def __rtruediv__(self, other: float) -> Tensor:
        """Return elementwise ``other / self``."""
        ...

    def __repr__(self) -> str:
        """Return a compact debugging representation."""
        ...
