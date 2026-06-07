#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

namespace cinder
{
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
         * @brief 分配器 traits 类型（allocator traits type）。Allocator traits type.
         *
         * @tparam Allocator 分配器候选类型（allocator candidate type）。The allocator candidate type.
         */
        template <typename Allocator>
        using allocator_traits_t = std::allocator_traits<std::remove_cvref_t<Allocator>>;

    } // namespace detail

    /**
     * @brief 张量式类型（Tensor-like type）：表示有限自然数多重索引到实数的映射。Tensor-like type: a map from finite natural-number multi-indices to real values.
     *
     * @tparam Tensor 张量候选类型（tensor candidate type）。The tensor candidate type.
     *
     * @note 数学契约是 `shape()[0] x ... x shape()[n - 1] -> R`：`shape()` 给出每一维的有限定义域
     *       （finite domain），其长度就是阶数/维数（rank/dimension）`n`；`operator()(std::span<const index_type>)`
     *       在长度为 `n` 的多重索引（multi-index）上求值并返回实数（real number）。The mathematical contract is
     *       `shape()[0] x ... x shape()[n - 1] -> R`: `shape()` gives each finite domain extent, its length is
     *       the rank/dimension `n`, and `operator()(std::span<const index_type>)` evaluates the tensor at a
     *       multi-index of length `n` and returns a real number.
     * @note 该概念故意不要求存储布局、连续内存或可变性。Those are implementation details, not part of the
     *       first-principles definition of a tensor as a function.
     * @note 对于运行时阶数（runtime rank），C++ concept 无法静态表达“调用时必须刚好传入 `n` 个独立参数”。
     *       因此核心求值接口使用 `std::span<const index_type>` 承载这 `n` 个数；具体张量类型仍可额外提供
     *       `tensor(i, j, k)` 这类便利重载（convenience overload）。
     */
    template <typename Tensor>
    concept TensorLike =
        requires {
            typename detail::tensor_value_t<Tensor>;
            typename detail::tensor_index_t<Tensor>;
        } &&
        std::floating_point<detail::tensor_value_t<Tensor>> &&
        std::unsigned_integral<detail::tensor_index_t<Tensor>> &&
        requires(const std::remove_cvref_t<Tensor> &tensor) {
            { tensor.shape() } -> std::ranges::sized_range;
        } &&
        std::ranges::forward_range<detail::tensor_shape_t<Tensor>> &&
        std::convertible_to<std::ranges::range_reference_t<detail::tensor_shape_t<Tensor>>,
                            detail::tensor_index_t<Tensor>> &&
        requires(const std::remove_cvref_t<Tensor> &tensor,
                 std::span<const detail::tensor_index_t<Tensor>> index) {
            { tensor(index) } -> std::convertible_to<detail::tensor_value_t<Tensor>>;
        };

    /**
     * @brief 标准库分配器式类型（standard-library allocator-like type）。Standard-library allocator-like type.
     *
     * @tparam Allocator 分配器候选类型（allocator candidate type）。The allocator candidate type.
     *
     * @note 该概念通过 `std::allocator_traits` 建模标准库 allocator 契约，而不是硬编码成员函数形状。
     *       This concept models the standard-library allocator contract through `std::allocator_traits`
     *       instead of hard-coding member function shapes.
     * @note `allocate`/`deallocate` 是最小内存契约；`rebind_alloc` 保证该类型能参与标准容器期望的
     *       重绑定（rebinding）流程。对象构造性（object constructibility）属于被分配对象的契约，
     *       不属于 allocator 本身的契约。
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
            typename detail::allocator_traits_t<Allocator>::template rebind_alloc<std::byte>;
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
