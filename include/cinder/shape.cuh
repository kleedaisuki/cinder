#pragma once

#include "cinder/concepts.cuh"

#include <cstddef>

namespace cinder
{

  /**
   * @brief 非拥有张量形状视图。Non-owning tensor shape view.
   *
   * @tparam Index extent 的整数类型（integer type for extents）。The integer type used for extents.
   *
   * @note 该类型只保存 rank 和 extent 指针；若在 device 上使用，extent 指针也必须指向 device 可访问内存。
   *       This type only stores a rank and extent pointer; if used on device, the extent pointer must also refer to device-accessible memory.
   */
  template <IntegralIndex Index = std::size_t>
  class Shape final
  {
  public:
    /**
     * @brief extent 整数类型别名。Alias for the extent integer type.
     */
    using index_type = Index;

    /**
     * @brief 大小类型别名。Alias for size values.
     */
    using size_type = std::size_t;

    /**
     * @brief 构造空形状。Construct an empty shape.
     */
    CINDER_HOST_DEVICE constexpr Shape() noexcept = default;

    /**
     * @brief 从 rank 和 extent 指针构造形状。Construct a shape from a rank and extent pointer.
     *
     * @param rank 张量秩（tensor rank）。The tensor rank.
     * @param extents 每个轴的 extent 指针（per-axis extent pointer）。The extent pointer for each axis.
     *
     * @note extents 必须至少包含 rank 个元素；rank 为 0 时不会读取 extents。
     *       extents must contain at least rank elements; extents is not read when rank is 0.
     */
    CINDER_HOST_DEVICE constexpr Shape(size_type rank, const index_type *extents) noexcept
        : rank_(rank),
          extents_(extents)
    {
    }

    /**
     * @brief 返回张量秩。Return the tensor rank.
     *
     * @return 运行时 rank 值。The runtime rank value.
     */
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto rank() const noexcept -> size_type
    {
      return rank_;
    }

    /**
     * @brief 返回 extent 元数据指针。Return the extent metadata pointer.
     *
     * @return 非拥有 extent 指针（non-owning extent pointer）。The non-owning extent pointer.
     */
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto extents() const noexcept -> const index_type *
    {
      return extents_;
    }

    /**
     * @brief 返回指定轴的 extent。Return the extent of an axis.
     *
     * @param axis 轴编号（axis index）。The axis index.
     * @return 指定轴的 extent。The extent of the requested axis.
     *
     * @note 不做边界检查；axis 必须小于 rank()。
     *       No bounds checking is performed; axis must be less than rank().
     */
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto extent(size_type axis) const noexcept -> index_type
    {
      return extents_[axis];
    }

    /**
     * @brief 返回 dense 张量的元素总数。Return the total element count of a dense tensor.
     *
     * @return 所有 extent 的乘积（product of all extents）。The product of all extents.
     *
     * @note rank 为 0 时返回 1，对应标量张量。
     *       Returns 1 when rank is 0, corresponding to a scalar tensor.
     */
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto element_count() const noexcept -> size_type
    {
      size_type count = 1;

      for (size_type axis = 0; axis < rank_; ++axis)
      {
        count *= static_cast<size_type>(extents_[axis]);
      }

      return count;
    }

  private:
    /**
     * @brief 运行时张量秩。Runtime tensor rank.
     */
    size_type rank_{};

    /**
     * @brief 非拥有 extent 元数据指针。Non-owning pointer to extent metadata.
     */
    const index_type *extents_{};
  };

} // namespace cinder
