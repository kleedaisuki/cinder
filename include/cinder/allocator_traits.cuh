#pragma once

#include "cinder/concepts.cuh"

#include <limits>
#include <memory>
#include <type_traits>

namespace cinder
{
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
