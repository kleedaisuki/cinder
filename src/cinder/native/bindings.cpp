#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(CINDER_HAS_CUDA)
#include "cinder/tensor.cuh"
#endif

#include <sstream>
#include <string>

namespace py = pybind11;

#if defined(CINDER_HAS_CUDA)
namespace
{

  /**
   * @brief 格式化 Tensor shape 供 repr 使用。Format a Tensor shape for repr.
   *
   * @param tensor Tensor 对象。Tensor object.
   * @return shape 字符串。Shape string.
   */
  [[nodiscard]] auto tensor_shape_repr(const cinder::Tensor &tensor) -> std::string
  {
    /**
     * @brief 字符串输出流。String output stream.
     */
    std::ostringstream stream;

    stream << '[';

    /**
     * @brief host 侧 shape 元信息。Host-side shape metadata.
     */
    const auto &shape = tensor.shape();

    for (std::size_t axis = 0U; axis < shape.size(); ++axis)
    {
      if (axis != 0U)
      {
        stream << ", ";
      }

      stream << shape[axis];
    }

    stream << ']';

    return stream.str();
  }

  /**
   * @brief 格式化 Tensor repr 字符串。Format the Tensor repr string.
   *
   * @param tensor Tensor 对象。Tensor object.
   * @return repr 字符串。repr string.
   */
  [[nodiscard]] auto tensor_repr(const cinder::Tensor &tensor) -> std::string
  {
    /**
     * @brief 字符串输出流。String output stream.
     */
    std::ostringstream stream;

    stream << "Tensor(shape=" << tensor_shape_repr(tensor) << ", size=" << tensor.size() << ", dtype=float32, device=cuda)";

    return stream.str();
  }

  /**
   * @brief 拼接当前 Tensor 与 Python 序列中的多个 Tensor。Concatenate this Tensor with multiple Tensors from a Python sequence.
   *
   * @param tensor 当前 Tensor。Current Tensor.
   * @param others 其他 Tensor 序列。Sequence of other Tensors.
   * @param axis 拼接轴。Concatenation axis.
   * @return concat 结果 Tensor。Concatenation result Tensor.
   */
  [[nodiscard]] auto tensor_concat_sequence(const cinder::Tensor &tensor,
                                            py::sequence others,
                                            cinder::Tensor::size_type axis) -> cinder::Tensor
  {
    /**
     * @brief 非拥有输入 Tensor 指针列表。Non-owning input Tensor pointer list.
     */
    std::vector<const cinder::Tensor *> inputs;

    inputs.reserve(1U + static_cast<std::size_t>(py::len(others)));
    inputs.push_back(&tensor);

    for (const auto item : others)
    {
      /**
       * @brief 当前 Python 对象中持有的 Tensor 引用。Tensor reference held by the current Python object.
       */
      const cinder::Tensor *other = nullptr;

      try
      {
        other = &item.cast<const cinder::Tensor &>();
      }
      catch (const py::cast_error &)
      {
        throw py::type_error("Tensor concat sequence entries must be Tensor objects");
      }

      inputs.push_back(other);
    }

    return cinder::concat(inputs, axis);
  }

  /**
   * @brief 注册 Tensor Python binding。Register Tensor Python bindings.
   *
   * @param module Python 模块对象。Python module object.
   */
  auto bind_tensor(py::module_ &module) -> void
  {
    py::class_<cinder::Tensor>(
        module,
        "Tensor",
        R"pbdoc(
        host-owned CUDA dense Tensor。

        Host-owned CUDA dense Tensor.
        )pbdoc")
        .def(
            py::init<std::vector<cinder::Tensor::size_type>>(),
            py::arg("shape"),
            R"pbdoc(
            创建零初始化 Tensor。

            Create a zero-initialized Tensor.
            )pbdoc")
        .def(
            py::init<std::vector<cinder::Tensor::size_type>, const std::vector<cinder::Tensor::value_type> &>(),
            py::arg("shape"),
            py::arg("values"),
            R"pbdoc(
            从 dense row-major host 数据创建 Tensor。

            Create a Tensor from dense row-major host data.
            )pbdoc")
        .def_property_readonly(
            "shape",
            [](const cinder::Tensor &tensor) {
              return tensor.shape();
            },
            R"pbdoc(
            Tensor shape。
            )pbdoc")
        .def_property_readonly(
            "rank",
            &cinder::Tensor::rank,
            R"pbdoc(
            Tensor rank。
            )pbdoc")
        .def_property_readonly(
            "size",
            &cinder::Tensor::size,
            R"pbdoc(
            dense 元素总数。

            Dense element count.
            )pbdoc")
        .def(
            "__len__",
            &cinder::Tensor::size,
            R"pbdoc(
            返回 dense 元素总数。

            Return the dense element count.
            )pbdoc")
        .def(
            "to_list",
            &cinder::Tensor::to_vector,
            R"pbdoc(
            把 Tensor data 拷回 Python list。

            Copy Tensor data back to a Python list.
            )pbdoc")
        .def(
            "reshape",
            &cinder::Tensor::reshape,
            py::arg("shape"),
            R"pbdoc(
            返回具有新 shape 的 dense Tensor。

            Return a dense Tensor with a new shape.
            )pbdoc")
        .def(
            "broadcast",
            &cinder::Tensor::broadcast,
            py::arg("shape"),
            R"pbdoc(
            按广播规则返回具有目标 shape 的 dense Tensor。

            Return a dense Tensor with the target shape by broadcasting.
            )pbdoc")
        .def(
            "slice",
            &cinder::Tensor::slice,
            py::arg("starts"),
            py::arg("shape"),
            R"pbdoc(
            返回当前 Tensor 的 dense 切片副本。

            Return a dense slice copy of the current Tensor.
            )pbdoc")
        .def(
            "concat",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs, cinder::Tensor::size_type axis) {
              return lhs.concat(rhs, axis);
            },
            py::arg("other"),
            py::arg("axis"),
            R"pbdoc(
            沿指定轴与另一个 Tensor 拼接。

            Concatenate this Tensor with another Tensor along an axis.
            )pbdoc")
        .def(
            "concat",
            &tensor_concat_sequence,
            py::arg("others"),
            py::arg("axis"),
            R"pbdoc(
            沿指定轴与多个 Tensor 拼接。

            Concatenate this Tensor with multiple Tensors along an axis.
            )pbdoc")
        .def(
            "tensor_product",
            &cinder::Tensor::tensor_product,
            py::arg("other"),
            R"pbdoc(
            返回当前 Tensor 与另一个 Tensor 的张量积。

            Return the tensor product of this Tensor and another Tensor.
            )pbdoc")
        .def(
            "transpose",
            static_cast<cinder::Tensor (cinder::Tensor::*)() const>(&cinder::Tensor::transpose),
            R"pbdoc(
            反转所有轴顺序并返回转置 Tensor。

            Return a transposed Tensor with all axes reversed.
            )pbdoc")
        .def(
            "transpose",
            static_cast<cinder::Tensor (cinder::Tensor::*)(const std::vector<cinder::Tensor::size_type> &) const>(
                &cinder::Tensor::transpose),
            py::arg("axes"),
            R"pbdoc(
            按给定轴置换返回转置 Tensor。

            Return a transposed Tensor using the given axis permutation.
            )pbdoc")
        .def(
            "contract",
            &cinder::Tensor::contract,
            py::arg("other"),
            py::arg("axes"),
            py::arg("other_axes"),
            R"pbdoc(
            沿指定轴对返回当前 Tensor 与另一个 Tensor 的缩并。

            Return the contraction of this Tensor and another Tensor along axis pairs.
            )pbdoc")
        .def(
            "mode_multiply",
            &cinder::Tensor::mode_multiply,
            py::arg("matrix"),
            py::arg("mode"),
            R"pbdoc(
            沿指定 mode 返回当前 Tensor 与矩阵的乘法。

            Return the mode multiplication of this Tensor and a matrix.
            )pbdoc")
        .def(
            "inner",
            &cinder::Tensor::inner,
            py::arg("other"),
            R"pbdoc(
            返回当前 Tensor 与另一个 Tensor 的内积，结果为 rank-0 Tensor。

            Return the inner product of this Tensor and another Tensor as a rank-0 Tensor.
            )pbdoc")
        .def(
            "dot",
            &cinder::Tensor::dot,
            py::arg("other"),
            R"pbdoc(
            返回当前 Tensor 与另一个 Tensor 的点积别名，结果为 rank-0 Tensor。

            Return the dot-product alias of this Tensor and another Tensor as a rank-0 Tensor.
            )pbdoc")
        .def(
            "norm",
            &cinder::Tensor::norm,
            R"pbdoc(
            返回当前 Tensor 的 L2 范数，结果为 rank-0 Tensor。

            Return the L2 norm of this Tensor as a rank-0 Tensor.
            )pbdoc")
        .def(
            "__add__",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return lhs + rhs;
            },
            py::is_operator())
        .def(
            "__add__",
            [](const cinder::Tensor &lhs, cinder::Tensor::value_type rhs) {
              return lhs + rhs;
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const cinder::Tensor &rhs, cinder::Tensor::value_type lhs) {
              return lhs + rhs;
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return lhs - rhs;
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const cinder::Tensor &lhs, cinder::Tensor::value_type rhs) {
              return lhs - rhs;
            },
            py::is_operator())
        .def(
            "__rsub__",
            [](const cinder::Tensor &rhs, cinder::Tensor::value_type lhs) {
              return lhs - rhs;
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return lhs * rhs;
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const cinder::Tensor &lhs, cinder::Tensor::value_type rhs) {
              return lhs * rhs;
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const cinder::Tensor &rhs, cinder::Tensor::value_type lhs) {
              return lhs * rhs;
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return lhs / rhs;
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const cinder::Tensor &lhs, cinder::Tensor::value_type rhs) {
              return lhs / rhs;
            },
            py::is_operator())
        .def(
            "__rtruediv__",
            [](const cinder::Tensor &rhs, cinder::Tensor::value_type lhs) {
              return lhs / rhs;
            },
            py::is_operator())
        .def("__repr__", &tensor_repr);
  }

} // namespace
#endif

/**
 * @brief 定义 Python 扩展模块入口。Define the Python extension module entry point.
 *
 * @param module Python 模块对象（Python module object）。The Python module object.
 * @note 该函数由 PYBIND11_MODULE 宏（macro）生成并导出给 CPython。
 *       This function is generated by the PYBIND11_MODULE macro and exported to CPython.
 */
PYBIND11_MODULE(core, module)
{
  module.doc() = "Native C++ extension module for cinder.";

#if defined(CINDER_HAS_CUDA)
  bind_tensor(module);
#endif
}
