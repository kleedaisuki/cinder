import math

import pytest

import cinder

try:
    import cinder.core as core
except ImportError:
    core = None


Tensor = getattr(cinder, "Tensor", None)

pytestmark = pytest.mark.skipif((Tensor is None) or (core is None), reason="CUDA Tensor bindings are not enabled")


def assert_tensor(tensor: object, shape: list[int], values: list[float]) -> None:
    assert isinstance(tensor, Tensor)
    assert tensor.shape == shape
    assert tensor.rank == len(shape)
    assert tensor.size == len(values)
    assert len(tensor) == len(values)
    assert tensor.to_list() == pytest.approx(values)


def row_major_offset(shape: list[int], indices: list[int]) -> int:
    offset = 0

    for extent, index in zip(shape, indices):
        offset = (offset * extent) + index

    return offset


def unravel_row_major(shape: list[int], linear_index: int) -> list[int]:
    coordinates: list[int] = []

    for extent in reversed(shape):
        coordinates.append(linear_index % extent)
        linear_index //= extent

    coordinates.reverse()
    return coordinates


def transpose_reference(shape: list[int], values: list[float], axes: list[int]) -> tuple[list[int], list[float]]:
    output_shape = [shape[axis] for axis in axes]
    output_values: list[float] = []

    for output_linear in range(math.prod(output_shape)):
        output_coordinates = unravel_row_major(output_shape, output_linear)
        input_indices = [0] * len(shape)

        for coordinate, input_axis in zip(output_coordinates, axes):
            input_indices[input_axis] = coordinate

        output_values.append(values[row_major_offset(shape, input_indices)])

    return output_shape, output_values


def broadcast_reference(input_shape: list[int], values: list[float], output_shape: list[int]) -> list[float]:
    output_values: list[float] = []
    leading_rank = len(output_shape) - len(input_shape)

    for output_linear in range(math.prod(output_shape)):
        output_coordinates = unravel_row_major(output_shape, output_linear)
        input_indices: list[int] = []

        for input_axis, input_extent in enumerate(input_shape):
            output_coordinate = output_coordinates[leading_rank + input_axis]
            input_indices.append(0 if input_extent == 1 else output_coordinate)

        output_values.append(values[row_major_offset(input_shape, input_indices)])

    return output_values


def slice_reference(
    input_shape: list[int],
    values: list[float],
    starts: list[int],
    output_shape: list[int],
) -> list[float]:
    output_values: list[float] = []

    for output_linear in range(math.prod(output_shape)):
        output_coordinates = unravel_row_major(output_shape, output_linear)
        input_indices = [start + coordinate for start, coordinate in zip(starts, output_coordinates)]
        output_values.append(values[row_major_offset(input_shape, input_indices)])

    return output_values


def concat_reference(
    input_shapes: list[list[int]],
    input_values: list[list[float]],
    axis: int,
) -> tuple[list[int], list[float]]:
    output_shape = input_shapes[0].copy()
    output_shape[axis] = sum(shape[axis] for shape in input_shapes)
    axis_offsets = [0]

    for shape in input_shapes:
        axis_offsets.append(axis_offsets[-1] + shape[axis])

    output_values: list[float] = []

    for output_linear in range(math.prod(output_shape)):
        output_coordinates = unravel_row_major(output_shape, output_linear)
        axis_coordinate = output_coordinates[axis]
        input_index = 0

        while (input_index + 1) < len(input_shapes) and axis_coordinate >= axis_offsets[input_index + 1]:
            input_index += 1

        input_coordinates = output_coordinates.copy()
        input_coordinates[axis] -= axis_offsets[input_index]
        output_values.append(input_values[input_index][row_major_offset(input_shapes[input_index], input_coordinates)])

    return output_shape, output_values


def contract_reference(
    lhs_shape: list[int],
    lhs_values: list[float],
    rhs_shape: list[int],
    rhs_values: list[float],
    lhs_axes: list[int],
    rhs_axes: list[int],
) -> tuple[list[int], list[float]]:
    lhs_free_axes = [axis for axis in range(len(lhs_shape)) if axis not in lhs_axes]
    rhs_free_axes = [axis for axis in range(len(rhs_shape)) if axis not in rhs_axes]
    output_shape = [lhs_shape[axis] for axis in lhs_free_axes] + [rhs_shape[axis] for axis in rhs_free_axes]
    contraction_shape = [lhs_shape[axis] for axis in lhs_axes]
    output_values: list[float] = []

    for output_linear in range(math.prod(output_shape)):
        output_coordinates = unravel_row_major(output_shape, output_linear)
        lhs_indices = [0] * len(lhs_shape)
        rhs_indices = [0] * len(rhs_shape)

        for coordinate, axis in zip(output_coordinates[: len(lhs_free_axes)], lhs_free_axes):
            lhs_indices[axis] = coordinate

        for coordinate, axis in zip(output_coordinates[len(lhs_free_axes) :], rhs_free_axes):
            rhs_indices[axis] = coordinate

        total = 0.0

        for contraction_linear in range(math.prod(contraction_shape)):
            contraction_coordinates = unravel_row_major(contraction_shape, contraction_linear)

            for coordinate, lhs_axis, rhs_axis in zip(contraction_coordinates, lhs_axes, rhs_axes):
                lhs_indices[lhs_axis] = coordinate
                rhs_indices[rhs_axis] = coordinate

            total += lhs_values[row_major_offset(lhs_shape, lhs_indices)] * rhs_values[
                row_major_offset(rhs_shape, rhs_indices)
            ]

        output_values.append(total)

    return output_shape, output_values


def mode_multiply_reference(
    input_shape: list[int],
    input_values: list[float],
    matrix_shape: list[int],
    matrix_values: list[float],
    mode: int,
) -> tuple[list[int], list[float]]:
    output_shape = input_shape.copy()
    output_shape[mode] = matrix_shape[0]
    output_values: list[float] = []

    for output_linear in range(math.prod(output_shape)):
        output_coordinates = unravel_row_major(output_shape, output_linear)
        input_indices = output_coordinates.copy()
        matrix_row = output_coordinates[mode]
        total = 0.0

        for mode_coordinate in range(input_shape[mode]):
            input_indices[mode] = mode_coordinate
            matrix_offset = (matrix_row * matrix_shape[1]) + mode_coordinate
            total += input_values[row_major_offset(input_shape, input_indices)] * matrix_values[matrix_offset]

        output_values.append(total)

    return output_shape, output_values


def test_public_api_exposes_tensor_only() -> None:
    assert cinder.__all__ == ["Tensor"]
    assert cinder.Tensor is core.Tensor
    assert hasattr(Tensor, "tensor_product")
    assert hasattr(Tensor, "reshape")
    assert hasattr(Tensor, "broadcast")
    assert hasattr(Tensor, "slice")
    assert hasattr(Tensor, "concat")
    assert hasattr(Tensor, "transpose")
    assert hasattr(Tensor, "contract")
    assert hasattr(Tensor, "mode_multiply")
    assert hasattr(Tensor, "inner")
    assert hasattr(Tensor, "dot")
    assert hasattr(Tensor, "norm")

    for name in (
        "add",
        "subtract",
        "multiply",
        "divide",
        "reshape",
        "broadcast",
        "slice",
        "concat",
        "tensor_product",
        "transpose",
        "contract",
        "mode_multiply",
        "inner",
        "dot",
        "norm",
    ):
        assert not hasattr(cinder, name)
        assert not hasattr(core, name)

    for name in ("add", "subtract", "multiply", "divide"):
        assert not hasattr(Tensor, name)


def test_tensor_constructs_from_shape_and_dense_values() -> None:
    tensor = Tensor([2, 3], [1.0, 2.5, -3.0, 4.0, 5.25, 6.0])

    assert_tensor(tensor, [2, 3], [1.0, 2.5, -3.0, 4.0, 5.25, 6.0])
    assert repr(tensor) == "Tensor(shape=[2, 3], size=6, dtype=float32, device=cuda)"


def test_tensor_accepts_integer_values_as_float32_inputs() -> None:
    tensor = Tensor([4], [1, 2, 3, 4])

    assert_tensor(tensor, [4], [1.0, 2.0, 3.0, 4.0])


def test_tensor_shape_constructor_zero_initializes_data() -> None:
    tensor = Tensor([4])

    assert_tensor(tensor, [4], [0.0, 0.0, 0.0, 0.0])


def test_tensor_preserves_dense_row_major_order_for_higher_rank_shape() -> None:
    values = [float(index) for index in range(24)]
    tensor = Tensor([2, 3, 4], values)

    assert_tensor(tensor, [2, 3, 4], values)


def test_scalar_tensor_has_rank_zero_and_one_element() -> None:
    lhs = Tensor([], [6.0])
    rhs = Tensor([], [4.0])

    assert_tensor(lhs, [], [6.0])
    assert_tensor(lhs + rhs, [], [10.0])
    assert_tensor(lhs - rhs, [], [2.0])
    assert_tensor(lhs * rhs, [], [24.0])
    assert_tensor(lhs / rhs, [], [1.5])


def test_zero_extent_tensor_has_no_values_but_preserves_shape() -> None:
    lhs = Tensor([2, 0, 3])
    rhs = Tensor([2, 0, 3])

    assert_tensor(lhs, [2, 0, 3], [])
    assert_tensor(lhs + rhs, [2, 0, 3], [])
    assert_tensor(lhs - rhs, [2, 0, 3], [])
    assert_tensor(lhs * rhs, [2, 0, 3], [])
    assert_tensor(lhs / rhs, [2, 0, 3], [])


def test_tensor_elementwise_operators() -> None:
    lhs = Tensor([2, 2], [8.0, -6.0, 4.5, 2.0])
    rhs = Tensor([2, 2], [2.0, 3.0, -1.5, 5.0])

    assert_tensor(lhs + rhs, [2, 2], [10.0, -3.0, 3.0, 7.0])
    assert_tensor(lhs - rhs, [2, 2], [6.0, -9.0, 6.0, -3.0])
    assert_tensor(lhs * rhs, [2, 2], [16.0, -18.0, -6.75, 10.0])
    assert_tensor(lhs / rhs, [2, 2], [4.0, -2.0, -3.0, 0.4])


def test_tensor_scalar_operators_support_both_operand_orders() -> None:
    tensor = Tensor([2, 2], [8.0, -6.0, 4.5, 2.0])

    assert_tensor(tensor + 2.5, [2, 2], [10.5, -3.5, 7.0, 4.5])
    assert_tensor(2.5 + tensor, [2, 2], [10.5, -3.5, 7.0, 4.5])
    assert_tensor(tensor - 1.5, [2, 2], [6.5, -7.5, 3.0, 0.5])
    assert_tensor(10.0 - tensor, [2, 2], [2.0, 16.0, 5.5, 8.0])
    assert_tensor(tensor * -2.0, [2, 2], [-16.0, 12.0, -9.0, -4.0])
    assert_tensor(-2.0 * tensor, [2, 2], [-16.0, 12.0, -9.0, -4.0])
    assert_tensor(tensor / 2.0, [2, 2], [4.0, -3.0, 2.25, 1.0])
    assert_tensor(18.0 / tensor, [2, 2], [2.25, -3.0, 4.0, 9.0])


def test_tensor_scalar_operators_preserve_zero_extent_shape() -> None:
    tensor = Tensor([2, 0, 3])

    assert_tensor(tensor + 1.0, [2, 0, 3], [])
    assert_tensor(1.0 - tensor, [2, 0, 3], [])
    assert_tensor(tensor * 2.0, [2, 0, 3], [])
    assert_tensor(2.0 / tensor, [2, 0, 3], [])


def test_tensor_chained_operations_reuse_kernel_written_metadata() -> None:
    lhs = Tensor([2], [5.0, 7.0])
    rhs = Tensor([2], [2.0, 3.0])

    assert_tensor(((lhs + rhs) * (lhs - rhs)) + 1.0, [2], [22.0, 41.0])


def test_tensor_operations_do_not_mutate_inputs() -> None:
    lhs = Tensor([3], [1.0, 2.0, 3.0])
    rhs = Tensor([3], [10.0, 20.0, 30.0])

    _ = (lhs + rhs) * rhs

    assert_tensor(lhs, [3], [1.0, 2.0, 3.0])
    assert_tensor(rhs, [3], [10.0, 20.0, 30.0])


def test_tensor_reshape_preserves_dense_row_major_order() -> None:
    values = [float(index + 1) for index in range(6)]
    tensor = Tensor([2, 3], values)

    assert_tensor(tensor.reshape([3, 2]), [3, 2], values)
    assert_tensor(tensor.reshape([6]), [6], values)


def test_tensor_reshape_supports_scalar_and_zero_extent_shapes() -> None:
    scalar = Tensor([1], [9.0]).reshape([])
    zero_extent = Tensor([2, 0, 3]).reshape([0, 6])

    assert_tensor(scalar, [], [9.0])
    assert_tensor(zero_extent, [0, 6], [])


def test_tensor_broadcast_vector_to_matrix() -> None:
    tensor = Tensor([3], [1.0, 2.0, 3.0])

    assert_tensor(tensor.broadcast([2, 3]), [2, 3], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


def test_tensor_broadcast_higher_rank_with_singleton_axes() -> None:
    input_shape = [2, 1, 3]
    output_shape = [2, 4, 3]
    values = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]

    assert_tensor(
        Tensor(input_shape, values).broadcast(output_shape),
        output_shape,
        broadcast_reference(input_shape, values, output_shape),
    )


def test_tensor_broadcast_scalar_and_zero_extent_shapes() -> None:
    assert_tensor(Tensor([], [5.0]).broadcast([2, 2]), [2, 2], [5.0, 5.0, 5.0, 5.0])
    assert_tensor(Tensor([1, 0, 3]).broadcast([2, 0, 3]), [2, 0, 3], [])


def test_tensor_slice_matrix_window() -> None:
    shape = [3, 4]
    values = [float(index + 1) for index in range(math.prod(shape))]
    starts = [1, 1]
    output_shape = [2, 2]

    assert_tensor(
        Tensor(shape, values).slice(starts, output_shape),
        output_shape,
        slice_reference(shape, values, starts, output_shape),
    )


def test_tensor_slice_supports_scalar_and_zero_extent_outputs() -> None:
    assert_tensor(Tensor([], [7.0]).slice([], []), [], [7.0])
    assert_tensor(Tensor([2, 0, 3]).slice([0, 0, 1], [2, 0, 1]), [2, 0, 1], [])


def test_tensor_concat_binary_axis_zero() -> None:
    lhs = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    rhs = Tensor([1, 2], [5.0, 6.0])

    assert_tensor(lhs.concat(rhs, 0), [3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_tensor_concat_sequence_uses_single_public_operation() -> None:
    lhs_shape = [2, 2]
    middle_shape = [2, 1]
    rhs_shape = [2, 3]
    lhs_values = [1.0, 2.0, 3.0, 4.0]
    middle_values = [5.0, 6.0]
    rhs_values = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    output_shape, output_values = concat_reference(
        [lhs_shape, middle_shape, rhs_shape],
        [lhs_values, middle_values, rhs_values],
        1,
    )

    result = Tensor(lhs_shape, lhs_values).concat(
        [Tensor(middle_shape, middle_values), Tensor(rhs_shape, rhs_values)],
        1,
    )

    assert_tensor(result, output_shape, output_values)


def test_tensor_concat_skips_zero_extent_inputs_on_concat_axis() -> None:
    lhs = Tensor([2, 0])
    rhs = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert_tensor(lhs.concat(rhs, 1), [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_tensor_transpose_matrix_default_reverses_axes() -> None:
    tensor = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert_tensor(tensor.transpose(), [3, 2], [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])


def test_tensor_transpose_higher_rank_axis_permutation() -> None:
    shape = [2, 3, 4]
    values = [float(index + 1) for index in range(math.prod(shape))]
    axes = [1, 2, 0]
    output_shape, output_values = transpose_reference(shape, values, axes)

    assert_tensor(Tensor(shape, values).transpose(axes), output_shape, output_values)


def test_tensor_transpose_scalar_and_vector_are_identity() -> None:
    scalar = Tensor([], [9.0])
    vector = Tensor([3], [1.0, -2.0, 4.0])

    assert_tensor(scalar.transpose(), [], [9.0])
    assert_tensor(scalar.transpose([]), [], [9.0])
    assert_tensor(vector.transpose(), [3], [1.0, -2.0, 4.0])


def test_tensor_transpose_zero_extent_preserves_permuted_shape() -> None:
    tensor = Tensor([2, 0, 3])

    assert_tensor(tensor.transpose([2, 1, 0]), [3, 0, 2], [])


def test_tensor_transpose_chains_with_elementwise_operations() -> None:
    lhs = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    rhs = Tensor([2, 2], [10.0, 20.0, 30.0, 40.0])

    assert_tensor(lhs.transpose() + rhs.transpose(), [2, 2], [11.0, 33.0, 22.0, 44.0])


def test_tensor_product_vector_outer_product() -> None:
    lhs = Tensor([2], [1.0, 2.0])
    rhs = Tensor([3], [10.0, 20.0, 30.0])

    assert_tensor(lhs.tensor_product(rhs), [2, 3], [10.0, 20.0, 30.0, 20.0, 40.0, 60.0])


def test_tensor_product_concatenates_higher_rank_shapes() -> None:
    lhs = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    rhs = Tensor([2], [-1.0, 5.0])

    assert_tensor(lhs.tensor_product(rhs), [2, 2, 2], [-1.0, 5.0, -2.0, 10.0, -3.0, 15.0, -4.0, 20.0])


def test_tensor_product_with_scalar_tensor() -> None:
    scalar = Tensor([], [3.0])
    tensor = Tensor([2, 2], [1.0, -2.0, 4.0, 0.5])

    assert_tensor(scalar.tensor_product(tensor), [2, 2], [3.0, -6.0, 12.0, 1.5])
    assert_tensor(tensor.tensor_product(scalar), [2, 2], [3.0, -6.0, 12.0, 1.5])


def test_tensor_product_zero_extent_preserves_concatenated_shape() -> None:
    lhs = Tensor([2, 0])
    rhs = Tensor([3])

    assert_tensor(lhs.tensor_product(rhs), [2, 0, 3], [])


def test_tensor_contract_vector_dot_product() -> None:
    lhs = Tensor([3], [1.0, 2.0, 3.0])
    rhs = Tensor([3], [10.0, 20.0, 30.0])

    assert_tensor(lhs.contract(rhs, [0], [0]), [], [140.0])


def test_tensor_contract_matrix_multiplication() -> None:
    lhs = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rhs = Tensor([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    assert_tensor(lhs.contract(rhs, [1], [0]), [2, 2], [58.0, 64.0, 139.0, 154.0])


def test_tensor_contract_multiple_axes_preserves_free_axis_order() -> None:
    lhs_shape = [2, 3, 2]
    rhs_shape = [2, 3, 4]
    lhs_values = [float(index + 1) for index in range(math.prod(lhs_shape))]
    rhs_values = [float((index % 7) - 3) for index in range(math.prod(rhs_shape))]
    output_shape, output_values = contract_reference(lhs_shape, lhs_values, rhs_shape, rhs_values, [2, 1], [0, 1])

    result = Tensor(lhs_shape, lhs_values).contract(Tensor(rhs_shape, rhs_values), [2, 1], [0, 1])

    assert_tensor(result, output_shape, output_values)


def test_tensor_contract_without_axes_matches_tensor_product_shape_and_values() -> None:
    lhs = Tensor([2], [2.0, -1.0])
    rhs = Tensor([2, 2], [3.0, 4.0, 5.0, 6.0])

    assert_tensor(lhs.contract(rhs, [], []), [2, 2, 2], [6.0, 8.0, 10.0, 12.0, -3.0, -4.0, -5.0, -6.0])


def test_tensor_contract_zero_contracted_extent_returns_zero_sum() -> None:
    lhs = Tensor([2, 0])
    rhs = Tensor([0, 3])

    assert_tensor(lhs.contract(rhs, [1], [0]), [2, 3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_tensor_contract_zero_free_extent_preserves_shape() -> None:
    lhs = Tensor([2, 0, 3])
    rhs = Tensor([3])

    assert_tensor(lhs.contract(rhs, [2], [0]), [2, 0], [])


def test_tensor_inner_product_returns_rank_zero_tensor() -> None:
    lhs = Tensor([2, 2], [1.0, -2.0, 3.5, 4.0])
    rhs = Tensor([2, 2], [5.0, 6.0, -1.0, 2.0])

    assert_tensor(lhs.inner(rhs), [], [-2.5])
    assert_tensor(lhs.dot(rhs), [], [-2.5])


def test_tensor_inner_product_zero_extent_returns_zero_scalar() -> None:
    lhs = Tensor([2, 0, 3])
    rhs = Tensor([2, 0, 3])

    assert_tensor(lhs.inner(rhs), [], [0.0])


def test_tensor_norm_returns_rank_zero_l2_norm() -> None:
    tensor = Tensor([3], [3.0, 4.0, 12.0])

    assert_tensor(tensor.norm(), [], [13.0])


def test_tensor_norm_zero_extent_returns_zero_scalar() -> None:
    tensor = Tensor([2, 0, 3])

    assert_tensor(tensor.norm(), [], [0.0])


def test_tensor_reductions_support_multi_block_inputs() -> None:
    size = 2049
    lhs = Tensor([size], [1.0] * size)
    rhs = Tensor([size], [2.0] * size)

    assert_tensor(lhs.inner(rhs), [], [2.0 * size])
    assert_tensor(lhs.norm(), [], [math.sqrt(size)])


def test_tensor_mode_multiply_matrix_mode_1() -> None:
    tensor = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    matrix = Tensor([2, 3], [10.0, 20.0, 30.0, -1.0, 0.0, 2.0])

    assert_tensor(tensor.mode_multiply(matrix, 1), [2, 2], [140.0, 5.0, 320.0, 8.0])


def test_tensor_mode_multiply_higher_rank_preserves_axis_order() -> None:
    input_shape = [2, 3, 4]
    matrix_shape = [5, 3]
    input_values = [float(index + 1) for index in range(math.prod(input_shape))]
    matrix_values = [float((index % 7) - 3) for index in range(math.prod(matrix_shape))]
    output_shape, output_values = mode_multiply_reference(input_shape, input_values, matrix_shape, matrix_values, 1)

    result = Tensor(input_shape, input_values).mode_multiply(Tensor(matrix_shape, matrix_values), 1)

    assert_tensor(result, output_shape, output_values)


def test_tensor_mode_multiply_mode_0_matches_left_matrix_multiply() -> None:
    tensor = Tensor([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    matrix = Tensor([2, 3], [7.0, 8.0, 9.0, -1.0, 0.5, 2.0])

    assert_tensor(tensor.mode_multiply(matrix, 0), [2, 2], [76.0, 100.0, 10.5, 12.0])


def test_tensor_mode_multiply_zero_mode_extent_returns_zero_sum() -> None:
    tensor = Tensor([2, 0, 3])
    matrix = Tensor([4, 0])

    assert_tensor(tensor.mode_multiply(matrix, 1), [2, 4, 3], [0.0] * 24)


def test_tensor_mode_multiply_zero_output_extent_preserves_shape() -> None:
    tensor = Tensor([2, 3])
    matrix = Tensor([0, 3])

    assert_tensor(tensor.mode_multiply(matrix, 1), [2, 0], [])


def test_tensor_division_follows_float32_cuda_semantics_for_zero_divisor() -> None:
    result = Tensor([3], [1.0, -1.0, 0.0]) / Tensor([3], [0.0, 0.0, 0.0])
    values = result.to_list()

    assert math.isinf(values[0]) and values[0] > 0.0
    assert math.isinf(values[1]) and values[1] < 0.0
    assert math.isnan(values[2])


def test_tensor_scalar_division_follows_float32_cuda_semantics_for_zero_divisor() -> None:
    tensor_divided_by_zero = Tensor([3], [1.0, -1.0, 0.0]) / 0.0
    values = tensor_divided_by_zero.to_list()

    assert math.isinf(values[0]) and values[0] > 0.0
    assert math.isinf(values[1]) and values[1] < 0.0
    assert math.isnan(values[2])

    zero_divided_by_tensor = 0.0 / Tensor([3], [1.0, -1.0, 0.0])
    reverse_values = zero_divided_by_tensor.to_list()

    assert reverse_values[0] == pytest.approx(0.0)
    assert reverse_values[1] == pytest.approx(-0.0)
    assert math.isnan(reverse_values[2])


@pytest.mark.parametrize(
    ("shape", "values"),
    [
        ([2, 2], [1.0, 2.0, 3.0]),
        ([2, 2], [1.0, 2.0, 3.0, 4.0, 5.0]),
        ([], []),
    ],
)
def test_tensor_rejects_value_count_mismatch(shape: list[int], values: list[float]) -> None:
    with pytest.raises(ValueError, match="values size"):
        Tensor(shape, values)


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape"),
    [
        ([2], [1, 2]),
        ([2, 3], [3, 2]),
        ([], [1]),
    ],
)
def test_tensor_rejects_shape_mismatch(lhs_shape: list[int], rhs_shape: list[int]) -> None:
    lhs = Tensor(lhs_shape, [1.0] * math.prod(lhs_shape))
    rhs = Tensor(rhs_shape, [1.0] * math.prod(rhs_shape))

    with pytest.raises(ValueError, match="shapes must match"):
        _ = lhs + rhs

    with pytest.raises(ValueError, match="shapes must match"):
        lhs.inner(rhs)


@pytest.mark.parametrize(
    "expression",
    [
        lambda tensor: tensor + object(),
        lambda tensor: object() + tensor,
        lambda tensor: tensor - object(),
        lambda tensor: object() * tensor,
        lambda tensor: tensor.reshape(object()),
        lambda tensor: tensor.broadcast(object()),
        lambda tensor: tensor.slice(object(), []),
        lambda tensor: tensor.slice([], object()),
        lambda tensor: tensor.concat(object(), 0),
        lambda tensor: tensor.concat([object()], 0),
        lambda tensor: tensor.tensor_product(object()),
        lambda tensor: tensor.transpose(object()),
        lambda tensor: tensor.contract(object(), [], []),
        lambda tensor: tensor.mode_multiply(object(), 0),
        lambda tensor: tensor.inner(object()),
        lambda tensor: tensor.dot(object()),
    ],
)
def test_tensor_rejects_non_tensor_operands(expression) -> None:
    tensor = Tensor([1], [1.0])

    with pytest.raises(TypeError):
        expression(tensor)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((3,), TypeError),
        (([1], None), TypeError),
        (([1], [object()]), TypeError),
    ],
)
def test_tensor_rejects_invalid_constructor_arguments(args: tuple[object, ...], error: type[Exception]) -> None:
    with pytest.raises(error):
        Tensor(*args)


def test_tensor_reshape_rejects_element_count_mismatch() -> None:
    tensor = Tensor([2, 3], [1.0] * 6)

    with pytest.raises(ValueError, match="element count"):
        tensor.reshape([5])


@pytest.mark.parametrize(
    ("shape", "error"),
    [
        ([2], "target rank"),
        ([2, 2], "incompatible"),
        ([1, 2, 4], "incompatible"),
    ],
)
def test_tensor_broadcast_rejects_invalid_shapes(shape: list[int], error: str) -> None:
    tensor = Tensor([2, 3], [1.0] * 6)

    with pytest.raises(ValueError, match=error):
        tensor.broadcast(shape)


@pytest.mark.parametrize(
    ("starts", "shape", "error"),
    [
        ([0], [1], "match Tensor rank"),
        ([3, 0], [0, 1], "start"),
        ([1, 2], [2, 2], "extent"),
    ],
)
def test_tensor_slice_rejects_invalid_windows(starts: list[int], shape: list[int], error: str) -> None:
    tensor = Tensor([2, 3], [1.0] * 6)

    with pytest.raises(ValueError, match=error):
        tensor.slice(starts, shape)


@pytest.mark.parametrize(
    ("rhs_shape", "axis", "error"),
    [
        ([2, 3], 2, "axis"),
        ([2], 0, "same rank"),
        ([3, 4], 0, "non-axis"),
    ],
)
def test_tensor_concat_rejects_invalid_inputs(rhs_shape: list[int], axis: int, error: str) -> None:
    lhs = Tensor([2, 3], [1.0] * 6)
    rhs = Tensor(rhs_shape, [1.0] * math.prod(rhs_shape))

    with pytest.raises(ValueError, match=error):
        lhs.concat(rhs, axis)


@pytest.mark.parametrize(
    ("axes", "error"),
    [
        ([0], "match Tensor rank"),
        ([0, 2], "out of range"),
        ([0, 0], "unique"),
    ],
)
def test_tensor_transpose_rejects_invalid_axes(axes: list[int], error: str) -> None:
    tensor = Tensor([2, 3], [1.0] * 6)

    with pytest.raises(ValueError, match=error):
        tensor.transpose(axes)


@pytest.mark.parametrize(
    ("lhs_axes", "rhs_axes", "error"),
    [
        ([0], [], "same length"),
        ([2], [0], "out of range"),
        ([0, 0], [0, 1], "unique"),
        ([0], [1], "extents must match"),
    ],
)
def test_tensor_contract_rejects_invalid_axes(lhs_axes: list[int], rhs_axes: list[int], error: str) -> None:
    lhs = Tensor([2, 3], [1.0] * 6)
    rhs = Tensor([2, 4], [1.0] * 8)

    with pytest.raises(ValueError, match=error):
        lhs.contract(rhs, lhs_axes, rhs_axes)


@pytest.mark.parametrize(
    ("matrix_shape", "mode", "error"),
    [
        ([3], 0, "rank 2"),
        ([2, 4], 1, "input extent"),
        ([2, 2], 2, "out of range"),
    ],
)
def test_tensor_mode_multiply_rejects_invalid_inputs(matrix_shape: list[int], mode: int, error: str) -> None:
    tensor = Tensor([2, 3], [1.0] * 6)
    matrix = Tensor(matrix_shape, [1.0] * math.prod(matrix_shape))

    with pytest.raises(ValueError, match=error):
        tensor.mode_multiply(matrix, mode)
