#pragma once

#include <memory>
#include <type_traits>

namespace cinder
{
    /**
     * @brief 默认线性存储分配器策略（default linear-storage allocator policy）。Default linear-storage allocator policy.
     *
     * @tparam Value 被分配的值类型（allocated value type）。The allocated value type.
     *
     * @note 这是标准库分配器（standard-library allocator）的薄别名，作为 `Tensor` 的默认 allocator policy。
     *       It is a thin alias over the standard-library allocator and serves as the default allocator
     *       policy for `Tensor`.
     */
    template <typename Value>
    using DefaultAllocator = std::allocator<std::remove_cvref_t<Value>>;

} // namespace cinder
