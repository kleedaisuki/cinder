#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>

/**
 * @brief 主机/设备双端调用标注（host/device callable annotation）。Host/device callable annotation.
 *
 * @note C++ concept 无法反射 CUDA/HIP 函数属性（function attribute）。使用该宏标注模型类型的
 *       `shape()`、`offset()` 和求值函数，并在同样标注的调用点实例化模板；如果实现不能在 device
 *       侧调用，CUDA/HIP 编译器会在 device pass 报错。C++ concepts cannot reflect CUDA/HIP
 *       function attributes. Mark model functions with this macro and instantiate them from similarly
 *       annotated call sites; non-device-callable implementations fail during the compiler's device pass.
 */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CINDER_HOST_DEVICE __host__ __device__
#else
#define CINDER_HOST_DEVICE
#endif

/**
 * @brief 断言检查宏（assertion macro）。Assertion macro.
 *
 * @param condition 必须为真的条件（condition that must hold）。The condition that must hold.
 *
 * @note CUDA/HIP device 侧也支持 `assert` 的受限形式；这里统一用一个宏表达 Tensor 的防御性不变量
 *       （defensive invariant）。CUDA/HIP device code supports a limited form of `assert`; this macro gives
 *       Tensor one spelling for defensive invariants across execution spaces.
 */
#define CINDER_ASSERT(condition) assert(condition)

namespace cinder
{
    /**
     * @brief 多重索引视图（multi-index view）。A host/device-safe view over a multi-index.
     *
     * @tparam Index 自然数索引类型（natural-number index type）。The natural-number index type.
     *
     * @note 该类型只保存指针和长度，避免把 device-facing 契约绑定到某个标准库 `std::span`
     *       实现。This type stores only a pointer and a length, avoiding a device-facing dependency on
     *       a particular standard-library `std::span` implementation.
     */
    template <std::unsigned_integral Index>
    class IndexView
    {
    public:
        /**
         * @brief 索引值类型（index value type）。The index value type.
         */
        using value_type = Index;

        /**
         * @brief 创建空索引视图（empty index view）。Create an empty index view.
         */
        CINDER_HOST_DEVICE constexpr IndexView() noexcept
        {
        }

        /**
         * @brief 从指针和长度创建索引视图。Create an index view from a pointer and a length.
         *
         * @param data 索引数据首地址（index data pointer）。The pointer to the first index value.
         * @param size 索引数量（number of indices）。The number of indices.
         */
        CINDER_HOST_DEVICE constexpr IndexView(const Index *data, std::size_t size) noexcept
            : data_{data},
              size_{size}
        {
        }

        /**
         * @brief 返回索引数据首地址（index data pointer）。Return the pointer to the first index value.
         *
         * @return 索引数据首地址（index data pointer）。The pointer to the first index value.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto data() const noexcept -> const Index *
        {
            return data_;
        }

        /**
         * @brief 返回索引数量（number of indices）。Return the number of indices.
         *
         * @return 索引数量（number of indices）。The number of indices.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto size() const noexcept -> std::size_t
        {
            return size_;
        }

        /**
         * @brief 判断视图是否为空（empty view check）。Return whether the view is empty.
         *
         * @return 若没有索引则为 true。True when the view contains no indices.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto empty() const noexcept -> bool
        {
            return size_ == 0U;
        }

        /**
         * @brief 按轴读取索引（axis index access）。Read the index at an axis.
         *
         * @param axis 轴编号（axis number）。The axis number.
         * @return 指定轴上的索引值（index value on the requested axis）。The index value at the requested axis.
         *
         * @note 该函数不做边界检查（bounds checking），调用者必须保证 `axis < size()`。This function
         *       performs no bounds checking; callers must ensure `axis < size()`.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator[](std::size_t axis) const noexcept -> Index
        {
            return data_[axis];
        }

    private:
        /**
         * @brief 索引数据首地址（index data pointer）。The pointer to the first index value.
         */
        const Index *data_{nullptr};

        /**
         * @brief 索引数量（number of indices）。The number of indices.
         */
        std::size_t size_{0U};
    };

    namespace detail
    {

        /**
         * @brief 张量标量值类型（tensor scalar value type）。Tensor scalar value type.
         *
         * @tparam Tensor 张量候选类型（tensor candidate type）。The tensor candidate type.
         */
        template <typename Tensor>
        using tensor_value_t = typename std::remove_cvref_t<Tensor>::value_type;

        /**
         * @brief 张量自然数索引类型（natural-number index type）。Tensor natural-number index type.
         *
         * @tparam Tensor 张量候选类型（tensor candidate type）。The tensor candidate type.
         */
        template <typename Tensor>
        using tensor_index_t = typename std::remove_cvref_t<Tensor>::index_type;

        /**
         * @brief 张量形状范围类型（tensor shape range type）。Tensor shape range type.
         *
         * @tparam Tensor 张量候选类型（tensor candidate type）。The tensor candidate type.
         */
        template <typename Tensor>
        using tensor_shape_t = decltype(std::declval<const std::remove_cvref_t<Tensor> &>().shape());

        /**
         * @brief 布局自然数索引类型（layout natural-number index type）。Layout natural-number index type.
         *
         * @tparam Layout 布局候选类型（layout candidate type）。The layout candidate type.
         */
        template <typename Layout>
        using layout_index_t = typename std::remove_cvref_t<Layout>::index_type;

        /**
         * @brief 布局线性偏移类型（layout linear offset type）。Layout linear offset type.
         *
         * @tparam Layout 布局候选类型（layout candidate type）。The layout candidate type.
         */
        template <typename Layout>
        using layout_offset_t = typename std::remove_cvref_t<Layout>::offset_type;

        /**
         * @brief 布局形状范围类型（layout shape range type）。Layout shape range type.
         *
         * @tparam Layout 布局候选类型（layout candidate type）。The layout candidate type.
         */
        template <typename Layout>
        using layout_shape_t = decltype(std::declval<const std::remove_cvref_t<Layout> &>().shape());

        /**
         * @brief 分配器 traits 类型（allocator traits type）。Allocator traits type.
         *
         * @tparam Allocator 分配器候选类型（allocator candidate type）。The allocator candidate type.
         */
        template <typename Allocator>
        using allocator_traits_t = std::allocator_traits<std::remove_cvref_t<Allocator>>;

    } // namespace detail

    /**
     * @brief 可算术运算值类型（arithmetic-like value type）。Arithmetic-like value type.
     *
     * @tparam Value 值类型候选（value type candidate）。The candidate value type.
     *
     * @note 该概念覆盖内建算术类型（built-in arithmetic types）以及用户自定义的数值类型
     *       （user-defined numeric types）。Tensor 存储需要值类型可默认构造、复制和析构；数值语义需要
     *       `+`、`-`、`*`、`/` 的结果可转换回该值类型。This concept covers both built-in arithmetic
     *       types and user-defined numeric types. Tensor storage needs default construction, copying,
     *       and destruction; numeric semantics require `+`, `-`, `*`, and `/` to produce values
     *       convertible back to the value type.
     */
    template <typename Value>
    concept ArithmeticLike =
        std::semiregular<std::remove_cvref_t<Value>> &&
        (!std::is_const_v<std::remove_cvref_t<Value>>) &&
        requires(const std::remove_cvref_t<Value> &lhs,
                 const std::remove_cvref_t<Value> &rhs) {
            { lhs + rhs } -> std::convertible_to<std::remove_cvref_t<Value>>;
            { lhs - rhs } -> std::convertible_to<std::remove_cvref_t<Value>>;
            { lhs * rhs } -> std::convertible_to<std::remove_cvref_t<Value>>;
            { lhs / rhs } -> std::convertible_to<std::remove_cvref_t<Value>>;
        };

    /**
     * @brief 张量式类型（Tensor-like type）：表示有限自然数多重索引到可算术值的映射。Tensor-like type: a map from finite natural-number multi-indices to arithmetic-like values.
     *
     * @tparam Tensor 张量候选类型（tensor candidate type）。The tensor candidate type.
     *
     * @note 数学契约是 `shape()[0] x ... x shape()[n - 1] -> V`：`shape()` 给出每一维的有限定义域
     *       （finite domain），其长度就是阶数/维数（rank/dimension）`n`；`operator()(IndexView<index_type>)`
     *       在长度为 `n` 的多重索引（multi-index）上求值并返回可算术值（arithmetic-like value）。The
     *       mathematical contract is `shape()[0] x ... x shape()[n - 1] -> V`: `shape()` gives each finite
     *       domain extent, its length is the rank/dimension `n`, and `operator()(IndexView<index_type>)`
     *       evaluates the tensor at a multi-index of length `n` and returns an arithmetic-like value.
     * @note 该概念故意不要求存储布局、连续内存或可变性。Those are implementation details, not part of the
     *       first-principles definition of a tensor as a function.
     * @note 对于运行时阶数（runtime rank），C++ concept 无法静态表达“调用时必须刚好传入 `n` 个独立参数”。
     *       因此核心求值接口使用 `IndexView<index_type>` 承载这 `n` 个数；具体张量类型仍可额外提供
     *       `tensor(i, j, k)` 这类便利重载（convenience overload）。
     */
    template <typename Tensor>
    concept TensorLike =
        requires {
            typename detail::tensor_value_t<Tensor>;
            typename detail::tensor_index_t<Tensor>;
        } &&
        ArithmeticLike<detail::tensor_value_t<Tensor>> &&
        std::unsigned_integral<detail::tensor_index_t<Tensor>> &&
        requires(const std::remove_cvref_t<Tensor> &tensor) {
            { tensor.shape() } -> std::ranges::sized_range;
        } &&
        std::ranges::forward_range<detail::tensor_shape_t<Tensor>> &&
        std::convertible_to<std::ranges::range_reference_t<detail::tensor_shape_t<Tensor>>,
                            detail::tensor_index_t<Tensor>> &&
        requires(const std::remove_cvref_t<Tensor> &tensor,
                 IndexView<detail::tensor_index_t<Tensor>> index) {
            { tensor(index) } -> std::convertible_to<detail::tensor_value_t<Tensor>>;
        };

    /**
     * @brief 布局式类型（layout-like type）：表示多重索引到线性偏移的映射。Layout-like type: a map from multi-indices to linear offsets.
     *
     * @tparam Layout 布局候选类型（layout candidate type）。The layout candidate type.
     *
     * @note 数学契约是 `shape()[0] x ... x shape()[n - 1] -> N`：`shape()` 定义张量的有限索引域，
     *       `offset(IndexView<index_type>)` 把长度为 `n` 的多重索引（multi-index）映射到线性偏移
     *       （linear offset）。The mathematical contract is `shape()[0] x ... x shape()[n - 1] -> N`:
     *       `shape()` defines the finite index domain, and `offset(IndexView<index_type>)` maps a
     *       multi-index of length `n` to a linear offset.
     * @note 该概念只描述地址计算，不要求行主序（row-major）、列主序（column-major）、步幅
     *       （stride）、连续性（contiguity）或无别名（alias-free）布局。This concept describes address
     *       calculation only; it does not require row-major order, column-major order, strides, contiguity,
     *       or alias freedom.
     * @note 若布局要在 host 与 device 共用，请用 `CINDER_HOST_DEVICE` 标注 `shape()` 和 `offset()`。
     *       The concept cannot directly inspect that annotation, but a device-side call site will enforce it.
     */
    template <typename Layout>
    concept LayoutLike =
        requires {
            typename detail::layout_index_t<Layout>;
            typename detail::layout_offset_t<Layout>;
        } &&
        std::unsigned_integral<detail::layout_index_t<Layout>> &&
        std::unsigned_integral<detail::layout_offset_t<Layout>> &&
        requires(const std::remove_cvref_t<Layout> &layout) {
            { layout.shape() } -> std::ranges::sized_range;
        } &&
        std::ranges::forward_range<detail::layout_shape_t<Layout>> &&
        std::convertible_to<std::ranges::range_reference_t<detail::layout_shape_t<Layout>>,
                            detail::layout_index_t<Layout>> &&
        requires(const std::remove_cvref_t<Layout> &layout,
                 IndexView<detail::layout_index_t<Layout>> index) {
            { layout.offset(index) } -> std::convertible_to<detail::layout_offset_t<Layout>>;
        };

    /**
     * @brief 标准库分配器式类型（standard-library allocator-like type）。Standard-library allocator-like type.
     *
     * @tparam Allocator 分配器候选类型（allocator candidate type）。The allocator candidate type.
     *
     * @note 该概念通过 `std::allocator_traits` 建模标准库 allocator 契约，而不是硬编码成员函数形状。
     *       This concept models the standard-library allocator contract through `std::allocator_traits`
     *       instead of hard-coding member function shapes.
     * @note `Tensor` 只需要当前 `value_type` 的 `allocate`/`deallocate` 最小内存契约，不要求
     *       `rebind`/`rebind_alloc`。对象构造性（object constructibility）属于被分配对象的契约，
     *       不属于 allocator 本身的契约。`Tensor` only needs the minimal memory contract for the current
     *       `value_type`; it does not require `rebind`/`rebind_alloc`.
     */
    template <typename Allocator>
    concept AllocatorLike =
        requires {
            typename std::remove_cvref_t<Allocator>::value_type;
            typename detail::allocator_traits_t<Allocator>::pointer;
            typename detail::allocator_traits_t<Allocator>::const_pointer;
            typename detail::allocator_traits_t<Allocator>::void_pointer;
            typename detail::allocator_traits_t<Allocator>::const_void_pointer;
            typename detail::allocator_traits_t<Allocator>::difference_type;
            typename detail::allocator_traits_t<Allocator>::size_type;
        } &&
        std::is_object_v<typename std::remove_cvref_t<Allocator>::value_type> &&
        (!std::is_const_v<typename std::remove_cvref_t<Allocator>::value_type>) &&
        std::copy_constructible<std::remove_cvref_t<Allocator>> &&
        requires(std::remove_cvref_t<Allocator> &allocator,
                 typename detail::allocator_traits_t<Allocator>::pointer pointer,
                 typename detail::allocator_traits_t<Allocator>::size_type count) {
            {
                detail::allocator_traits_t<Allocator>::allocate(allocator, count)
            } -> std::same_as<typename detail::allocator_traits_t<Allocator>::pointer>;
            detail::allocator_traits_t<Allocator>::deallocate(allocator, pointer, count);
            {
                detail::allocator_traits_t<Allocator>::max_size(allocator)
            } -> std::convertible_to<typename detail::allocator_traits_t<Allocator>::size_type>;
        };

} // namespace cinder
