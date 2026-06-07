#pragma once

#include "cinder/concepts.cuh"

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>

namespace cinder
{
    /**
     * @brief 默认线性存储分配器策略（default linear-storage allocator policy）。Default linear-storage allocator policy.
     *
     * @tparam Value 被分配的值类型（allocated value type）。The allocated value type.
     *
     * @note 该分配器在当前执行域（execution space）里分配本地堆内存：host 侧使用 C++ `operator new`，
     *       CUDA/HIP device 侧使用 device `malloc/free`。它不承诺跨 host/device 可见性；跨域可见性应由
     *       自定义 allocator policy 表达。This allocator uses the local heap of the current execution
     *       space: C++ `operator new` on host and device `malloc/free` on CUDA/HIP device. It does not
     *       promise host/device cross-visibility; custom allocator policies should model that.
     */
    template <typename Value>
    class DefaultAllocator
    {
    public:
        /**
         * @brief 值类型（value type）。The allocated value type.
         */
        using value_type = std::remove_cvref_t<Value>;

        /**
         * @brief 指针类型（pointer type）。The pointer type.
         */
        using pointer = value_type *;

        /**
         * @brief 常量指针类型（const pointer type）。The const pointer type.
         */
        using const_pointer = const value_type *;

        /**
         * @brief 空指针类型（void pointer type）。The void pointer type.
         */
        using void_pointer = void *;

        /**
         * @brief 常量空指针类型（const void pointer type）。The const void pointer type.
         */
        using const_void_pointer = const void *;

        /**
         * @brief 大小类型（size type）。The size type.
         */
        using size_type = std::size_t;

        /**
         * @brief 差值类型（difference type）。The difference type.
         */
        using difference_type = std::ptrdiff_t;

        /**
         * @brief 拷贝赋值时传播分配器（copy-assignment propagation）。Allocator propagation on copy assignment.
         */
        using propagate_on_container_copy_assignment = std::false_type;

        /**
         * @brief 移动赋值时传播分配器（move-assignment propagation）。Allocator propagation on move assignment.
         */
        using propagate_on_container_move_assignment = std::true_type;

        /**
         * @brief 分配器总是等价（always-equal allocator）。Always-equal allocator marker.
         */
        using is_always_equal = std::true_type;

        /**
         * @brief 创建默认分配器（default allocator construction）。Create a default allocator.
         */
        CINDER_HOST_DEVICE constexpr DefaultAllocator() noexcept = default;

        /**
         * @brief 从其他值类型的默认分配器创建（rebound allocator construction）。Create from a rebound allocator.
         *
         * @tparam Other 其他值类型（other value type）。The other value type.
         * @param other 其他分配器（other allocator）。The other allocator.
         */
        template <typename Other>
        CINDER_HOST_DEVICE constexpr DefaultAllocator(const DefaultAllocator<Other> &other) noexcept
        {
            (void)other;
        }

        /**
         * @brief 分配未初始化对象存储（allocate uninitialized object storage）。Allocate uninitialized object storage.
         *
         * @param count 对象数量（object count）。The object count.
         * @return 分配得到的对象指针（allocated object pointer）。The allocated object pointer.
         */
        [[nodiscard]] CINDER_HOST_DEVICE auto allocate(size_type count) -> pointer
        {
            if (count == size_type{0U})
            {
                return nullptr;
            }

            if (count > max_size())
            {
                CINDER_ASSERT(false);
                return nullptr;
            }
            const size_type byte_count = static_cast<size_type>(count * sizeof(value_type));

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            void *const memory = ::malloc(byte_count);
#else
            void *const memory = ::operator new(byte_count, std::nothrow);
#endif
            CINDER_ASSERT(memory != nullptr);
            return static_cast<pointer>(memory);
        }

        /**
         * @brief 释放未初始化或已销毁对象存储（deallocate object storage）。Deallocate object storage.
         *
         * @param storage 对象存储指针（object storage pointer）。The object storage pointer.
         * @param count 对象数量（object count）。The object count.
         */
        CINDER_HOST_DEVICE void deallocate(pointer storage, size_type count) noexcept
        {
            (void)count;
            if (storage == nullptr)
            {
                return;
            }

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            ::free(storage);
#else
            ::operator delete(storage, std::nothrow);
#endif
        }

        /**
         * @brief 返回最大可分配元素数（maximum allocatable element count）。Return max allocatable element count.
         *
         * @return 最大元素数量（maximum element count）。The maximum element count.
         */
        [[nodiscard]] CINDER_HOST_DEVICE static constexpr auto max_size() noexcept -> size_type
        {
            return static_cast<size_type>(std::numeric_limits<size_type>::max() / sizeof(value_type));
        }
    };

    /**
     * @brief 比较默认分配器（default allocator equality）。Compare default allocators.
     *
     * @tparam Lhs 左值类型（left value type）。The left value type.
     * @tparam Rhs 右值类型（right value type）。The right value type.
     * @param lhs 左分配器（left allocator）。The left allocator.
     * @param rhs 右分配器（right allocator）。The right allocator.
     * @return 总是 true，因为默认分配器无状态。Always true because the default allocator is stateless.
     */
    template <typename Lhs, typename Rhs>
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator==(const DefaultAllocator<Lhs> &lhs,
                                                               const DefaultAllocator<Rhs> &rhs) noexcept -> bool
    {
        (void)lhs;
        (void)rhs;
        return true;
    }

} // namespace cinder
