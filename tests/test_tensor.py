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


def test_public_api_exposes_tensor_only() -> None:
    assert cinder.__all__ == ["Tensor"]
    assert cinder.Tensor is core.Tensor

    for name in ("add", "subtract", "multiply", "divide"):
        assert not hasattr(cinder, name)
        assert not hasattr(core, name)
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


def test_tensor_chained_operations_reuse_kernel_written_metadata() -> None:
    lhs = Tensor([2], [5.0, 7.0])
    rhs = Tensor([2], [2.0, 3.0])

    assert_tensor((lhs + rhs) * (lhs - rhs), [2], [21.0, 40.0])


def test_tensor_operations_do_not_mutate_inputs() -> None:
    lhs = Tensor([3], [1.0, 2.0, 3.0])
    rhs = Tensor([3], [10.0, 20.0, 30.0])

    _ = (lhs + rhs) * rhs

    assert_tensor(lhs, [3], [1.0, 2.0, 3.0])
    assert_tensor(rhs, [3], [10.0, 20.0, 30.0])


def test_tensor_division_follows_float32_cuda_semantics_for_zero_divisor() -> None:
    result = Tensor([3], [1.0, -1.0, 0.0]) / Tensor([3], [0.0, 0.0, 0.0])
    values = result.to_list()

    assert math.isinf(values[0]) and values[0] > 0.0
    assert math.isinf(values[1]) and values[1] < 0.0
    assert math.isnan(values[2])


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


@pytest.mark.parametrize(
    "expression",
    [
        lambda tensor: tensor + 1.0,
        lambda tensor: 1.0 + tensor,
        lambda tensor: tensor - object(),
        lambda tensor: object() * tensor,
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
