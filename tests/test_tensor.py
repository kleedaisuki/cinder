import pytest

import cinder


Tensor = getattr(cinder, "Tensor", None)

pytestmark = pytest.mark.skipif(Tensor is None, reason="CUDA Tensor bindings are not enabled")


def test_tensor_constructs_from_shape_and_values() -> None:
    tensor = Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert tensor.shape == [2, 3]
    assert tensor.rank == 2
    assert tensor.size == 6
    assert len(tensor) == 6
    assert tensor.to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_tensor_shape_constructor_zero_initializes_data() -> None:
    tensor = Tensor([4])

    assert tensor.shape == [4]
    assert tensor.to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_tensor_elementwise_operators() -> None:
    lhs = Tensor([2, 2], [8.0, 6.0, 4.0, 2.0])
    rhs = Tensor([2, 2], [2.0, 3.0, 4.0, 5.0])

    assert (lhs + rhs).to_list() == pytest.approx([10.0, 9.0, 8.0, 7.0])
    assert (lhs - rhs).to_list() == pytest.approx([6.0, 3.0, 0.0, -3.0])
    assert (lhs * rhs).to_list() == pytest.approx([16.0, 18.0, 16.0, 10.0])
    assert (lhs / rhs).to_list() == pytest.approx([4.0, 2.0, 1.0, 0.4])


def test_tensor_module_functions() -> None:
    lhs = Tensor([3], [1.0, 2.0, 3.0])
    rhs = Tensor([3], [4.0, 5.0, 6.0])

    assert cinder.add(lhs, rhs).to_list() == pytest.approx([5.0, 7.0, 9.0])
    assert cinder.subtract(lhs, rhs).to_list() == pytest.approx([-3.0, -3.0, -3.0])
    assert cinder.multiply(lhs, rhs).to_list() == pytest.approx([4.0, 10.0, 18.0])
    assert cinder.divide(rhs, lhs).to_list() == pytest.approx([4.0, 2.5, 2.0])


def test_tensor_chained_operations_reuse_kernel_written_metadata() -> None:
    lhs = Tensor([2], [5.0, 7.0])
    rhs = Tensor([2], [2.0, 3.0])

    assert ((lhs + rhs) * (lhs - rhs)).to_list() == pytest.approx([21.0, 40.0])


def test_tensor_rejects_value_count_mismatch() -> None:
    with pytest.raises(ValueError):
        Tensor([2, 2], [1.0, 2.0, 3.0])


def test_tensor_rejects_shape_mismatch() -> None:
    lhs = Tensor([2], [1.0, 2.0])
    rhs = Tensor([1, 2], [1.0, 2.0])

    with pytest.raises(ValueError):
        _ = lhs + rhs
