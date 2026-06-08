#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cinder/add.hpp"

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
            "add",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return cinder::add(lhs, rhs);
            },
            py::arg("other"),
            R"pbdoc(
            逐元素加法。

            Elementwise addition.
            )pbdoc")
        .def(
            "subtract",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return cinder::subtract(lhs, rhs);
            },
            py::arg("other"),
            R"pbdoc(
            逐元素减法。

            Elementwise subtraction.
            )pbdoc")
        .def(
            "multiply",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return cinder::multiply(lhs, rhs);
            },
            py::arg("other"),
            R"pbdoc(
            逐元素乘法。

            Elementwise multiplication.
            )pbdoc")
        .def(
            "divide",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return cinder::divide(lhs, rhs);
            },
            py::arg("other"),
            R"pbdoc(
            逐元素除法。

            Elementwise division.
            )pbdoc")
        .def(
            "__add__",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
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
            "__mul__",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return lhs * rhs;
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const cinder::Tensor &lhs, const cinder::Tensor &rhs) {
              return lhs / rhs;
            },
            py::is_operator())
        .def("__repr__", &tensor_repr);

    module.def(
        "add",
        static_cast<cinder::Tensor (*)(const cinder::Tensor &, const cinder::Tensor &)>(&cinder::add),
        py::arg("lhs"),
        py::arg("rhs"),
        R"pbdoc(
        逐元素 Tensor 加法。

        Elementwise Tensor addition.
        )pbdoc");

    module.def(
        "subtract",
        &cinder::subtract,
        py::arg("lhs"),
        py::arg("rhs"),
        R"pbdoc(
        逐元素 Tensor 减法。

        Elementwise Tensor subtraction.
        )pbdoc");

    module.def(
        "multiply",
        &cinder::multiply,
        py::arg("lhs"),
        py::arg("rhs"),
        R"pbdoc(
        逐元素 Tensor 乘法。

        Elementwise Tensor multiplication.
        )pbdoc");

    module.def(
        "divide",
        &cinder::divide,
        py::arg("lhs"),
        py::arg("rhs"),
        R"pbdoc(
        逐元素 Tensor 除法。

        Elementwise Tensor division.
        )pbdoc");
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

  module.def(
      "add",
      static_cast<int (*)(int, int) noexcept>(&cinder::add),
      py::arg("lhs"),
      py::arg("rhs"),
      R"pbdoc(
      返回两个整数的和。

      Return the sum of two integers.
      )pbdoc");

#if defined(CINDER_HAS_CUDA)
  bind_tensor(module);
#endif
}
