#include "cinder/mapping.cuh"
#include "cinder/shape.cuh"
#include "cinder/tensor_view.cuh"

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace
{

  /**
   * @brief 将标量 shape 映射到第一个元素的标量映射。Scalar mapping that maps a scalar shape to the first element.
   */
  struct ScalarMapping final
  {
    /**
     * @brief 返回标量元素的线性偏移。Return the linear offset for a scalar element.
     *
     * @tparam Shape 形状类型（shape type）。The shape type.
     * @tparam Index 索引元素类型（index element type）。The index element type.
     * @param shape 标量张量形状（scalar tensor shape）。The scalar tensor shape.
     * @param indices 索引元组指针（index tuple pointer）。The index tuple pointer.
     * @return 标量元素偏移 0。Scalar element offset 0.
     */
    template <cinder::TensorShape Shape, cinder::IntegralIndex Index>
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator()(const Shape &shape,
                                                               const Index *indices) const noexcept -> std::size_t
    {
      static_cast<void>(shape);
      static_cast<void>(indices);
      return 0;
    }
  };

  /**
   * @brief 只使用 shape 的最后一维 extent 的二维映射。2D mapping that uses the last shape extent.
   */
  struct LastExtent2DMapping final
  {
    /**
     * @brief 返回二维索引在线性内存中的偏移。Return the linear-memory offset for a 2D index.
     *
     * @tparam Shape 形状类型（shape type）。The shape type.
     * @tparam Index 索引元素类型（index element type）。The index element type.
     * @param shape 张量形状（tensor shape）。The tensor shape.
     * @param indices 二维索引元组指针（2D index tuple pointer）。The pointer to the 2D index tuple.
     * @return 线性偏移（linear offset）。The linear offset.
     */
    template <cinder::TensorShape Shape, cinder::IntegralIndex Index>
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator()(const Shape &shape,
                                                               const Index *indices) const noexcept -> std::size_t
    {
      return (static_cast<std::size_t>(indices[0]) * static_cast<std::size_t>(shape.extent(1U))) +
             static_cast<std::size_t>(indices[1]);
    }
  };

} // namespace

/**
 * @brief 运行 TensorView 的 CUDA C++ smoke test。Run the CUDA C++ smoke test for TensorView.
 *
 * @return 进程退出码（process exit code）。The process exit code.
 */
auto main() -> int
{
  /**
   * @brief row-major 测试存储。Storage for the row-major test.
   */
  int row_storage[6] = {0, 1, 2, 3, 4, 5};

  /**
   * @brief row-major 张量每轴 extent。Per-axis extents for the row-major tensor.
   */
  const std::size_t row_extents[2] = {2U, 3U};

  /**
   * @brief row-major 张量 shape。Shape for the row-major tensor.
   */
  const cinder::Shape row_shape(2U, row_extents);

  /**
   * @brief row-major 元素索引。Element index for the row-major tensor.
   */
  const std::size_t row_index[2] = {1U, 2U};

  /**
   * @brief dense row-major 张量视图。Dense row-major tensor view.
   */
  cinder::TensorView<int, cinder::DenseRowMajorMapping> row_view(row_storage, row_shape);

  static_assert(std::is_trivially_copyable_v<decltype(row_view)>);
  static_assert(cinder::TensorMappingFor<cinder::DenseRowMajorMapping, cinder::Shape<>, std::size_t>);
  assert(row_view.shape().rank() == 2U);
  assert(row_view.shape().extent(0U) == 2U);
  assert(row_view.shape().extent(1U) == 3U);
  assert(row_view.shape().element_count() == 6U);
  assert(row_view(row_index) == 5);

  /**
   * @brief row-major 已修改元素索引。Modified element index for the row-major tensor.
   */
  const std::size_t row_modified_index[2] = {0U, 1U};

  row_view(row_modified_index) = 42;
  assert(row_storage[1] == 42);
  assert(row_view.linear(1U) == 42);

  /**
   * @brief const 视图对象仍保持 span 风格的元素可变性。A const view object still keeps span-like mutable element access.
   */
  const cinder::TensorView<int, cinder::DenseRowMajorMapping> const_view_object(row_storage, row_shape);

  /**
   * @brief row-major 第一列索引。First-column index for the row-major tensor.
   */
  const std::size_t row_first_column_index[2] = {1U, 0U};

  const_view_object(row_first_column_index) = 7;
  assert(row_storage[3] == 7);

  /**
   * @brief 只读元素视图。Tensor view with const-qualified elements.
   */
  const cinder::TensorView<const int, cinder::DenseRowMajorMapping> const_element_view(row_view);
  assert(const_element_view(row_modified_index) == 42);

  /**
   * @brief column-major 测试存储。Storage for the column-major test.
   */
  int column_storage[6] = {0, 1, 2, 3, 4, 5};

  /**
   * @brief column-major 张量每轴 extent。Per-axis extents for the column-major tensor.
   */
  const std::size_t column_extents[2] = {2U, 3U};

  /**
   * @brief column-major 张量 shape。Shape for the column-major tensor.
   */
  const cinder::Shape column_shape(2U, column_extents);

  /**
   * @brief column-major 元素索引。Element index for the column-major tensor.
   */
  const std::size_t column_index[2] = {1U, 2U};

  /**
   * @brief dense column-major 张量视图。Dense column-major tensor view.
   */
  cinder::TensorView<int, cinder::DenseColumnMajorMapping> column_view(column_storage, column_shape);

  assert(column_view(column_index) == 5);

  /**
   * @brief 自定义映射测试存储。Storage for the custom mapping test.
   */
  int custom_storage[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  /**
   * @brief 自定义映射张量每轴 extent。Per-axis extents for the custom mapping tensor.
   */
  const std::size_t custom_extents[2] = {2U, 4U};

  /**
   * @brief 自定义映射张量 shape。Shape for the custom mapping tensor.
   */
  const cinder::Shape custom_shape(2U, custom_extents);

  /**
   * @brief 自定义映射索引。Index for the custom mapping tensor.
   */
  const std::size_t custom_index[2] = {1U, 2U};

  /**
   * @brief 自定义映射张量视图。Tensor view with a custom mapping policy.
   */
  cinder::TensorView<int, LastExtent2DMapping> custom_view(custom_storage, custom_shape);

  assert(custom_view(custom_index) == 6);

  /**
   * @brief 标量映射测试存储。Storage for the scalar mapping test.
   */
  int scalar_storage[1] = {99};

  /**
   * @brief 标量张量 shape。Shape for the scalar tensor.
   */
  const cinder::Shape scalar_shape(0U, static_cast<const std::size_t *>(nullptr));

  /**
   * @brief 标量张量空索引指针。Empty index pointer for the scalar tensor.
   */
  const std::size_t *scalar_indices = nullptr;

  /**
   * @brief 标量张量视图。Scalar tensor view.
   */
  cinder::TensorView<int, ScalarMapping> scalar_view(scalar_storage, scalar_shape);

  assert(scalar_view(scalar_indices) == 99);

  return 0;
}
