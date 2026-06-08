"""
@brief cinder.core 原生扩展模块的公开类型契约。Public type contract for cinder.core.

@note 该 stub（type stub）只描述 pybind11 导出的稳定 Python API。
      This stub describes only the stable Python API exported by pybind11.
@note CUDA kernel 和 C++ 自由函数不属于 Python 公开面。
      CUDA kernels and C++ free functions are not part of the Python public surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

# @brief shape-like 输入；每个元素是非负 axis extent。
#        Shape-like input; each item is a non-negative axis extent.
_ShapeLike = Sequence[int]

# @brief axes-like 输入；每个元素是 axis index。
#        Axes-like input; each item is an axis index.
_AxesLike = Sequence[int]

# @brief dense row-major float32 输入；Python int 按 float-compatible 值接收。
#        Dense row-major float32 input; Python int is accepted as a float-compatible value.
_DenseValuesLike = Sequence[float]

# @brief 标量输入；运行时转换为 Tensor 的 float32 value_type。
#        Scalar input; converted at runtime to the Tensor float32 value_type.
_ScalarLike = float


class Tensor:
    """
    @brief CUDA 支持的 dense float32 Tensor。CUDA-backed dense float32 Tensor.

    @note 数据按 row-major order 存储；公开操作返回新 Tensor，不会原地修改输入。
          Data is stored in row-major order; public operations return new Tensors and do not mutate
          their inputs.
    @note rank-0 Tensor 使用空 shape ``[]`` 表示 scalar；zero extent shape 合法且 size 为 0。
          Rank-0 Tensors use ``[]`` for scalars; zero-extent shapes are valid and have size 0.
    """

    # Construction

    @overload
    def __init__(self, shape: _ShapeLike) -> None:
        """
        @brief 创建零初始化 Tensor。Create a zero-initialized Tensor.
        @param shape 每个轴的 extent；``[]`` 表示 rank-0 scalar。
                     Per-axis extents; ``[]`` denotes a rank-0 scalar.
        """
        ...

    @overload
    def __init__(self, shape: _ShapeLike, values: _DenseValuesLike) -> None:
        """
        @brief 从 dense row-major host 数据创建 Tensor。Create a Tensor from dense row-major host data.
        @param shape 每个轴的 extent；其乘积必须等于 ``len(values)``。
                     Per-axis extents; their product must equal ``len(values)``.
        @param values 输入数据，会转换为 float32 存储。
                      Input values converted to float32 storage.
        """
        ...

    # Metadata

    @property
    def shape(self) -> list[int]:
        """
        @brief 返回 Tensor shape 的 Python list 副本。Return a Python list copy of the Tensor shape.
        @return 每个轴的 extent。Per-axis extents.
        @note 修改返回的 list 不会改变 Tensor。Mutating the returned list does not change the Tensor.
        """
        ...

    @property
    def rank(self) -> int:
        """
        @brief 返回 Tensor rank。Return the Tensor rank.
        @return axis 数量；rank-0 scalar 返回 0。Number of axes; rank-0 scalars return 0.
        """
        ...

    @property
    def size(self) -> int:
        """
        @brief 返回 dense 元素总数。Return the dense element count.
        @return ``shape`` 中所有 extent 的乘积；zero extent shape 返回 0。
                Product of all extents in ``shape``; zero-extent shapes return 0.
        """
        ...

    def __len__(self) -> int:
        """
        @brief 返回 dense 元素总数。Return the dense element count.
        @return 与 ``size`` 相同的值。The same value as ``size``.
        """
        ...

    # Host transfer

    def to_list(self) -> list[float]:
        """
        @brief 将 device 数据复制为 Python list。Copy device data into a Python list.
        @return dense row-major float32 数据的 host 副本。
                Host copy of dense row-major float32 values.
        """
        ...

    # Shape operations

    def reshape(self, shape: _ShapeLike) -> Tensor:
        """
        @brief 以相同 row-major 数据顺序返回新 shape 的 Tensor。
               Return a Tensor with a new shape and the same row-major data order.
        @param shape 目标 shape；元素总数必须等于当前 ``size``。
                     Target shape; its element count must equal the current ``size``.
        @return reshape 结果 Tensor。Reshaped Tensor.
        """
        ...

    def broadcast(self, shape: _ShapeLike) -> Tensor:
        """
        @brief 按 trailing-axis broadcasting 规则返回目标 shape 的 Tensor。
               Return a Tensor with the target shape using trailing-axis broadcasting.
        @param shape 目标 shape，rank 必须不小于当前 rank。
                     Target shape whose rank must be at least the current rank.
        @return broadcast 结果 Tensor。Broadcast result Tensor.
        @note 每个输入轴必须与对应输出轴相等，或输入 extent 为 1。
              Each input axis must either match the corresponding output axis or have extent 1.
        """
        ...

    def slice(self, starts: _AxesLike, shape: _ShapeLike) -> Tensor:
        """
        @brief 返回 dense slice 副本。Return a dense slice copy.
        @param starts 每个轴的起始坐标，长度必须等于 ``rank``。
                      Per-axis start coordinates; length must equal ``rank``.
        @param shape 输出 slice 的 shape，长度必须等于 ``rank``。
                     Output slice shape; length must equal ``rank``.
        @return slice 结果 Tensor。Slice result Tensor.
        """
        ...

    @overload
    def concat(self, other: Tensor, axis: int) -> Tensor:
        """
        @brief 沿指定轴拼接另一个 Tensor。Concatenate another Tensor along an axis.
        @param other 右侧 Tensor；rank 必须相同，非拼接轴 extent 必须匹配。
                      Right-hand Tensor; rank and non-concat extents must match.
        @param axis 拼接轴。Concatenation axis.
        @return concat 结果 Tensor。Concatenation result Tensor.
        """
        ...

    @overload
    def concat(self, others: Sequence[Tensor], axis: int) -> Tensor:
        """
        @brief 沿指定轴拼接多个 Tensor。Concatenate multiple Tensors along an axis.
        @param others 追加到当前 Tensor 后面的 Tensor 序列。
                       Tensors appended after the current Tensor.
        @param axis 拼接轴。Concatenation axis.
        @return concat 结果 Tensor。Concatenation result Tensor.
        """
        ...

    # Tensor algebra

    def tensor_product(self, other: Tensor) -> Tensor:
        """
        @brief 返回两个 Tensor 的 tensor product。Return the tensor product of two Tensors.
        @param other 右侧 Tensor。Right-hand Tensor.
        @return shape 为 ``self.shape + other.shape`` 的 Tensor。
                Tensor whose shape is ``self.shape + other.shape``.
        """
        ...

    @overload
    def transpose(self) -> Tensor:
        """
        @brief 反转所有轴顺序。Reverse all axes.
        @return 默认 transpose 结果 Tensor。Default transposed Tensor.
        """
        ...

    @overload
    def transpose(self, axes: _AxesLike) -> Tensor:
        """
        @brief 按给定 axis permutation 转置。Transpose by an explicit axis permutation.
        @param axes 长度必须等于 ``rank`` 且每个 axis 必须唯一。
                     Permutation whose length must equal ``rank`` and whose axes must be unique.
        @return transpose 结果 Tensor。Transposed Tensor.
        """
        ...

    def contract(self, other: Tensor, axes: _AxesLike, other_axes: _AxesLike) -> Tensor:
        """
        @brief 沿成对轴执行 Tensor contraction。Contract two Tensors along paired axes.
        @param other 右侧 Tensor。Right-hand Tensor.
        @param axes 当前 Tensor 中参与 contraction 的轴。Axes in this Tensor to contract.
        @param other_axes ``other`` 中参与 contraction 的轴。Axes in ``other`` to contract.
        @return 输出 Tensor；free axes 顺序为当前 Tensor 的 free axes 后接 ``other`` 的 free axes。
                Output Tensor; free axes are ordered as this Tensor's free axes followed by
                ``other`` free axes.
        """
        ...

    def mode_multiply(self, matrix: Tensor, mode: int) -> Tensor:
        """
        @brief 沿指定 mode 执行矩阵乘法。Multiply this Tensor by a matrix along one mode.
        @param matrix rank-2 Tensor，shape 为 ``[out_extent, self.shape[mode]]``。
                       Rank-2 Tensor with shape ``[out_extent, self.shape[mode]]``.
        @param mode 被替换的输入轴。Input axis to replace.
        @return 输出 shape 将 ``self.shape[mode]`` 替换为 ``matrix.shape[0]``。
                Output shape replaces ``self.shape[mode]`` with ``matrix.shape[0]``.
        """
        ...

    # Reductions

    def inner(self, other: Tensor) -> Tensor:
        """
        @brief 返回 full inner product。Return the full inner product.
        @param other shape 必须与当前 Tensor 相同。Tensor whose shape must match this Tensor.
        @return rank-0 Tensor，包含逐元素乘积之和。
                Rank-0 Tensor containing the sum of elementwise products.
        """
        ...

    def dot(self, other: Tensor) -> Tensor:
        """
        @brief ``inner`` 的别名。Alias for ``inner``.
        @param other shape 必须与当前 Tensor 相同。Tensor whose shape must match this Tensor.
        @return rank-0 Tensor，包含逐元素乘积之和。
                Rank-0 Tensor containing the sum of elementwise products.
        """
        ...

    def norm(self) -> Tensor:
        """
        @brief 返回 L2 norm。Return the L2 norm.
        @return rank-0 Tensor；zero-extent Tensor 的 norm 为 0。
                Rank-0 Tensor; zero-extent Tensors have norm 0.
        """
        ...

    # Elementwise operators

    @overload
    def __add__(self, other: Tensor) -> Tensor:
        """
        @brief 逐元素 Tensor 加法。Elementwise Tensor addition.
        @param other shape 必须匹配当前 Tensor。Tensor whose shape must match this Tensor.
        @return 加法结果 Tensor。Addition result Tensor.
        """
        ...

    @overload
    def __add__(self, other: _ScalarLike) -> Tensor:
        """
        @brief 逐元素 scalar 加法。Elementwise scalar addition.
        @param other scalar 值，会转换为 float32。Scalar value converted to float32.
        @return 加法结果 Tensor。Addition result Tensor.
        """
        ...

    def __radd__(self, other: _ScalarLike) -> Tensor:
        """
        @brief 逐元素 reverse scalar 加法。Elementwise reverse scalar addition.
        @param other scalar 值，会转换为 float32。Scalar value converted to float32.
        @return 加法结果 Tensor。Addition result Tensor.
        """
        ...

    @overload
    def __sub__(self, other: Tensor) -> Tensor:
        """
        @brief 逐元素 Tensor 减法。Elementwise Tensor subtraction.
        @param other shape 必须匹配当前 Tensor。Tensor whose shape must match this Tensor.
        @return 减法结果 Tensor。Subtraction result Tensor.
        """
        ...

    @overload
    def __sub__(self, other: _ScalarLike) -> Tensor:
        """
        @brief 逐元素 scalar 减法。Elementwise scalar subtraction.
        @param other scalar 值，会转换为 float32。Scalar value converted to float32.
        @return 减法结果 Tensor。Subtraction result Tensor.
        """
        ...

    def __rsub__(self, other: _ScalarLike) -> Tensor:
        """
        @brief 逐元素 reverse scalar 减法。Elementwise reverse scalar subtraction.
        @param other scalar 值，会转换为 float32。Scalar value converted to float32.
        @return 减法结果 Tensor。Subtraction result Tensor.
        """
        ...

    @overload
    def __mul__(self, other: Tensor) -> Tensor:
        """
        @brief 逐元素 Tensor 乘法。Elementwise Tensor multiplication.
        @param other shape 必须匹配当前 Tensor。Tensor whose shape must match this Tensor.
        @return 乘法结果 Tensor。Multiplication result Tensor.
        """
        ...

    @overload
    def __mul__(self, other: _ScalarLike) -> Tensor:
        """
        @brief 逐元素 scalar 乘法。Elementwise scalar multiplication.
        @param other scalar 值，会转换为 float32。Scalar value converted to float32.
        @return 乘法结果 Tensor。Multiplication result Tensor.
        """
        ...

    def __rmul__(self, other: _ScalarLike) -> Tensor:
        """
        @brief 逐元素 reverse scalar 乘法。Elementwise reverse scalar multiplication.
        @param other scalar 值，会转换为 float32。Scalar value converted to float32.
        @return 乘法结果 Tensor。Multiplication result Tensor.
        """
        ...

    @overload
    def __truediv__(self, other: Tensor) -> Tensor:
        """
        @brief 逐元素 Tensor 除法。Elementwise Tensor division.
        @param other shape 必须匹配当前 Tensor。Tensor whose shape must match this Tensor.
        @return 除法结果 Tensor，遵循 CUDA float32 除零语义。
                Division result Tensor following CUDA float32 division-by-zero semantics.
        """
        ...

    @overload
    def __truediv__(self, other: _ScalarLike) -> Tensor:
        """
        @brief 逐元素 scalar 除法。Elementwise scalar division.
        @param other scalar 值，会转换为 float32。Scalar value converted to float32.
        @return 除法结果 Tensor，遵循 CUDA float32 除零语义。
                Division result Tensor following CUDA float32 division-by-zero semantics.
        """
        ...

    def __rtruediv__(self, other: _ScalarLike) -> Tensor:
        """
        @brief 逐元素 reverse scalar 除法。Elementwise reverse scalar division.
        @param other scalar 值，会转换为 float32。Scalar value converted to float32.
        @return 除法结果 Tensor，遵循 CUDA float32 除零语义。
                Division result Tensor following CUDA float32 division-by-zero semantics.
        """
        ...

    # Representation

    def __repr__(self) -> str:
        """
        @brief 返回调试用 repr 字符串。Return a debugging repr string.
        @return 包含 shape、size、dtype 与 device 的字符串。
                String containing shape, size, dtype, and device.
        """
        ...
