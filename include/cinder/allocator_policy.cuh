#pragma once

#include "cinder/concepts.cuh"

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
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
         * @brief 重绑定分配器（allocator rebinding）。Allocator rebinding.
         *
         * @tparam Other 目标值类型（target value type）。The target value type.
         */
        template <typename Other>
        struct rebind
        {
            /**
             * @brief 重绑定后的分配器类型（rebound allocator type）。The rebound allocator type.
             */
            using other = DefaultAllocator<Other>;
        };

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

    /**
     * @brief 分配器策略 traits（allocator policy traits）。Allocator policy traits.
     *
     * @tparam Allocator 分配器策略类型（allocator policy type）。The allocator policy type.
     *
     * @note 类型成员复用 `std::allocator_traits` 的标准推导；运行时操作优先调用 allocator 成员函数，
     *       否则使用 host/device 可编译的默认构造与销毁路径。Type members reuse standard deduction from
     *       `std::allocator_traits`; runtime operations prefer allocator member functions and otherwise use
     *       host/device-compilable construction and destruction paths.
     */
    template <typename Allocator>
    struct AllocatorTraits
    {
        /**
         * @brief 分配器类型（allocator type）。The allocator type.
         */
        using allocator_type = std::remove_cvref_t<Allocator>;

        /**
         * @brief 标准 allocator traits 类型（standard allocator traits type）。The standard allocator traits type.
         */
        using standard_traits = std::allocator_traits<allocator_type>;

        /**
         * @brief 值类型（value type）。The value type.
         */
        using value_type = typename standard_traits::value_type;

        /**
         * @brief 指针类型（pointer type）。The pointer type.
         */
        using pointer = typename standard_traits::pointer;

        /**
         * @brief 大小类型（size type）。The size type.
         */
        using size_type = typename standard_traits::size_type;

        /**
         * @brief 拷贝赋值传播标记（copy-assignment propagation marker）。Copy-assignment propagation marker.
         */
        using propagate_on_container_copy_assignment =
            typename standard_traits::propagate_on_container_copy_assignment;

        /**
         * @brief 移动赋值传播标记（move-assignment propagation marker）。Move-assignment propagation marker.
         */
        using propagate_on_container_move_assignment =
            typename standard_traits::propagate_on_container_move_assignment;

        /**
         * @brief 总是等价标记（always-equal marker）。Always-equal marker.
         */
        using is_always_equal = typename standard_traits::is_always_equal;

        /**
         * @brief 分配未初始化存储（allocate uninitialized storage）。Allocate uninitialized storage.
         *
         * @param allocator 分配器对象（allocator object）。The allocator object.
         * @param count 元素数量（element count）。The element count.
         * @return 分配器指针（allocator pointer）。The allocator pointer.
         */
        [[nodiscard]] CINDER_HOST_DEVICE static auto allocate(allocator_type &allocator,
                                                              size_type count) -> pointer
        {
            return allocator.allocate(count);
        }

        /**
         * @brief 释放存储（deallocate storage）。Deallocate storage.
         *
         * @param allocator 分配器对象（allocator object）。The allocator object.
         * @param storage 存储指针（storage pointer）。The storage pointer.
         * @param count 元素数量（element count）。The element count.
         */
        CINDER_HOST_DEVICE static void deallocate(allocator_type &allocator,
                                                  pointer storage,
                                                  size_type count) noexcept
        {
            allocator.deallocate(storage, count);
        }

        /**
         * @brief 构造对象（construct object）。Construct an object.
         *
         * @tparam Args 构造参数类型（constructor argument types）。The constructor argument types.
         * @param allocator 分配器对象（allocator object）。The allocator object.
         * @param storage 对象存储指针（object storage pointer）。The object storage pointer.
         * @param args 构造参数（constructor arguments）。The constructor arguments.
         */
        template <typename... Args>
        CINDER_HOST_DEVICE static void construct(allocator_type &allocator,
                                                 value_type *storage,
                                                 Args &&...args)
        {
            if constexpr (requires {
                              allocator.construct(storage, static_cast<Args &&>(args)...);
                          })
            {
                allocator.construct(storage, static_cast<Args &&>(args)...);
            }
            else
            {
                ::new (static_cast<void *>(storage)) value_type(static_cast<Args &&>(args)...);
            }
        }

        /**
         * @brief 销毁对象（destroy object）。Destroy an object.
         *
         * @param allocator 分配器对象（allocator object）。The allocator object.
         * @param storage 对象指针（object pointer）。The object pointer.
         */
        CINDER_HOST_DEVICE static void destroy(allocator_type &allocator,
                                               value_type *storage) noexcept
        {
            if constexpr (requires {
                              allocator.destroy(storage);
                          })
            {
                allocator.destroy(storage);
            }
            else
            {
                storage->~value_type();
            }
        }

        /**
         * @brief 返回最大可分配元素数（maximum allocatable element count）。Return max allocatable element count.
         *
         * @param allocator 分配器对象（allocator object）。The allocator object.
         * @return 最大元素数量（maximum element count）。The maximum element count.
         */
        [[nodiscard]] CINDER_HOST_DEVICE static auto max_size(const allocator_type &allocator) noexcept -> size_type
        {
            if constexpr (requires {
                              allocator.max_size();
                          })
            {
                return allocator.max_size();
            }
            else
            {
                return static_cast<size_type>(std::numeric_limits<size_type>::max() / sizeof(value_type));
            }
        }

        /**
         * @brief 选择拷贝构造用分配器（select allocator on copy construction）。Select allocator on copy construction.
         *
         * @param allocator 源分配器（source allocator）。The source allocator.
         * @return 拷贝构造使用的分配器（allocator for copy construction）。The allocator to use for copy construction.
         */
        [[nodiscard]] CINDER_HOST_DEVICE static auto select_on_container_copy_construction(
            const allocator_type &allocator) -> allocator_type
        {
            if constexpr (requires {
                              allocator.select_on_container_copy_construction();
                          })
            {
                return allocator.select_on_container_copy_construction();
            }
            else
            {
                return allocator;
            }
        }
    };

} // namespace cinder
