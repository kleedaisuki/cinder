#pragma once

#include "cinder/concepts.cuh"
#include "cinder/shape.cuh"

#include <cstddef>
#include <type_traits>

namespace cinder
{

  /**
   * @brief 将非拥有线性内存解释为由 shape 和映射策略定义的张量视图。
   *        Interpret non-owning linear memory as a tensor view defined by a shape and mapping policy.
   *
   * @tparam Element 元素类型（element type），可以带 const 限定。The element type, optionally const-qualified.
   * @tparam Mapping 映射策略（mapping policy），把 shape 和索引元组映射到线性偏移。
   *         The mapping policy that maps a shape and index tuple to a linear offset.
   *
   * @note 该类型只保存数据指针和 shape，不保存 mapping 对象，不分配、不释放、不搬移底层内存。
   *       This type only stores a data pointer and shape, stores no mapping object, and never allocates, frees, or moves the underlying memory.
   */
  template <TensorElement Element, TensorMapping Mapping>
  class TensorView final
  {
  public:
    /**
     * @brief 元素类型别名。Alias for the element type.
     */
    using element_type = Element;

    /**
     * @brief 去除 cv 限定后的值类型。Value type with cv-qualification removed.
     */
    using value_type = std::remove_cv_t<element_type>;

    /**
     * @brief 映射策略类型别名。Alias for the mapping policy type.
     */
    using mapping_type = Mapping;

    /**
     * @brief shape 类型别名。Alias for the shape type.
     */
    using shape_type = Shape<>;

    /**
     * @brief 大小与偏移类型别名。Alias for size and offset values.
     */
    using size_type = std::size_t;

    /**
     * @brief 指向元素的指针类型。Pointer type to an element.
     */
    using pointer = element_type *;

    /**
     * @brief 元素引用类型。Reference type to an element.
     */
    using reference = element_type &;

    /**
     * @brief 构造空视图。Construct an empty view.
     */
    CINDER_HOST_DEVICE constexpr TensorView() noexcept = default;

    /**
     * @brief 从线性内存指针和 shape 构造视图。Construct a view from a linear memory pointer and shape.
     *
     * @param data 非拥有内存起点（non-owning memory base）。The base address of non-owning memory.
     * @param shape 张量形状（tensor shape）。The tensor shape.
     */
    CINDER_HOST_DEVICE constexpr TensorView(pointer data, shape_type shape) noexcept
        : data_(data),
          shape_(shape)
    {
    }

    /**
     * @brief 从兼容元素类型的视图构造视图。Construct a view from another view with a compatible element type.
     *
     * @tparam OtherElement 源视图的元素类型（source element type）。The element type of the source view.
     * @param other 源张量视图（source tensor view）。The source tensor view.
     *
     * @note 典型用途是从 TensorView<T, M> 转换为 TensorView<const T, M>。
     *       A typical use is converting TensorView<T, M> to TensorView<const T, M>.
     */
    template <typename OtherElement>
      requires(std::is_convertible_v<OtherElement *, pointer>)
    CINDER_HOST_DEVICE constexpr TensorView(const TensorView<OtherElement, mapping_type> &other) noexcept
        : data_(other.data()),
          shape_(other.shape())
    {
    }

    /**
     * @brief 返回非拥有内存起点。Return the base address of the non-owning memory.
     *
     * @return 非拥有内存指针（non-owning memory pointer）。The non-owning memory pointer.
     */
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto data() const noexcept -> pointer
    {
      return data_;
    }

    /**
     * @brief 返回张量形状。Return the tensor shape.
     *
     * @return shape 对象（shape object）。The shape object.
     */
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto shape() const noexcept -> shape_type
    {
      return shape_;
    }

    /**
     * @brief 判断视图是否持有非空指针。Check whether the view holds a non-null pointer.
     *
     * @return 若数据指针非空则为 true。True when the data pointer is non-null.
     */
    [[nodiscard]] CINDER_HOST_DEVICE constexpr explicit operator bool() const noexcept
    {
      return data_ != nullptr;
    }

    /**
     * @brief 用 shape 和索引元组访问张量元素。Access a tensor element with the shape and an index tuple.
     *
     * @tparam Index 索引元组元素类型（index tuple element type）。The element type of the index tuple.
     * @param indices 指向索引元组的指针（pointer to index tuple）。The pointer to the index tuple.
     * @return 被映射元素的引用。A reference to the mapped element.
     *
     * @note 索引元组长度由 shape().rank() 决定；调用方必须保证 indices 至少包含 shape().rank() 个元素。
     *       The tuple length is defined by shape().rank(); the caller must ensure indices contains at least shape().rank() elements.
     */
    template <typename Index>
      requires TensorMappingFor<mapping_type, shape_type, Index>
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator()(const Index *indices) const
        noexcept(noexcept(mapping_type{}(shape_, indices))) -> reference
    {
      return data_[static_cast<size_type>(mapping_type{}(shape_, indices))];
    }

    /**
     * @brief 直接按线性偏移访问元素。Access an element directly by linear offset.
     *
     * @tparam Offset 线性偏移类型（linear offset type）。The linear offset type.
     * @param offset 非负线性偏移（non-negative linear offset）。The non-negative linear offset.
     * @return 对应线性位置的元素引用。A reference to the element at the linear position.
     *
     * @note 该函数绕过映射策略，便于与 span-like 代码互操作。
     *       This function bypasses the mapping policy for span-like interoperability.
     */
    template <typename Offset>
      requires IntegralIndex<Offset>
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto linear(Offset offset) const noexcept -> reference
    {
      return data_[static_cast<size_type>(offset)];
    }

  private:
    /**
     * @brief 非拥有线性内存起点。Base address of the non-owning linear memory.
     */
    pointer data_{};

    /**
     * @brief 非拥有张量形状视图。Non-owning tensor shape view.
     */
    shape_type shape_{};

  };

} // namespace cinder
