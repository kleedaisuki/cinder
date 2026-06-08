#pragma once

#include "cinder/concepts.cuh"

#include <cstddef>

namespace cinder
{

  /**
   * @brief 无状态 dense row-major 映射纯函数。Stateless pure function for dense row-major mapping.
   *
   * @note 该 policy 不保存 shape 或索引元数据；shape 和 indices 由 TensorView 在访问时传入。
   *       This policy stores no shape or index metadata; shape and indices are passed by TensorView during access.
   */
  class DenseRowMajorMapping final
  {
  public:
    /**
     * @brief 将 shape 和索引元组映射为 dense row-major 线性偏移。
     *        Map a shape and index tuple to a dense row-major linear offset.
     *
     * @tparam Shape 形状类型（shape type）。The shape type.
     * @tparam Index 索引元素类型（index element type）。The index element type.
     * @param shape 张量形状（tensor shape）。The tensor shape.
     * @param indices 非负整数索引元组指针（non-negative integer index tuple pointer）。The pointer to the non-negative integer index tuple.
     * @return 线性偏移（linear offset）。The linear offset.
     *
     * @note 不检查 extent 边界；调用方必须保证 shape 和索引有效。
     *       Extent bounds are not checked; the caller must ensure the shape and indices are valid.
     */
    template <TensorShape Shape, IntegralIndex Index>
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator()(const Shape &shape, const Index *indices) const noexcept
        -> std::size_t
    {
      std::size_t offset = 0;

      for (std::size_t axis = 0; axis < shape.rank(); ++axis)
      {
        offset = (offset * static_cast<std::size_t>(shape.extent(axis))) + static_cast<std::size_t>(indices[axis]);
      }

      return offset;
    }
  };

  /**
   * @brief 无状态 dense column-major 映射纯函数。Stateless pure function for dense column-major mapping.
   *
   * @note 该 policy 不保存 shape 或索引元数据；shape 和 indices 由 TensorView 在访问时传入。
   *       This policy stores no shape or index metadata; shape and indices are passed by TensorView during access.
   */
  class DenseColumnMajorMapping final
  {
  public:
    /**
     * @brief 将 shape 和索引元组映射为 dense column-major 线性偏移。
     *        Map a shape and index tuple to a dense column-major linear offset.
     *
     * @tparam Shape 形状类型（shape type）。The shape type.
     * @tparam Index 索引元素类型（index element type）。The index element type.
     * @param shape 张量形状（tensor shape）。The tensor shape.
     * @param indices 非负整数索引元组指针（non-negative integer index tuple pointer）。The pointer to the non-negative integer index tuple.
     * @return 线性偏移（linear offset）。The linear offset.
     *
     * @note 不检查 extent 边界；调用方必须保证 shape 和索引有效。
     *       Extent bounds are not checked; the caller must ensure the shape and indices are valid.
     */
    template <TensorShape Shape, IntegralIndex Index>
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator()(const Shape &shape, const Index *indices) const noexcept
        -> std::size_t
    {
      std::size_t offset = 0;
      std::size_t scale = 1;

      for (std::size_t axis = 0; axis < shape.rank(); ++axis)
      {
        offset += static_cast<std::size_t>(indices[axis]) * scale;
        scale *= static_cast<std::size_t>(shape.extent(axis));
      }

      return offset;
    }
  };

} // namespace cinder
