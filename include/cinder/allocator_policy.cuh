#pragma once

#include "cinder/concepts.cuh"

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>

/**
 * @brief CUDA Runtime API 头可用性标记（CUDA Runtime API header availability marker）。Marks whether the CUDA Runtime API header is available.
 *
 * @note 该宏只表达当前编译单元能否包含 `cuda_runtime_api.h`；真正调用 CUDA allocator 时仍需要链接
 *       CUDA runtime。This macro only records whether this translation unit can include
 *       `cuda_runtime_api.h`; actually calling CUDA allocators still requires linking the CUDA runtime.
 */
#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#define CINDER_DETAIL_HAS_CUDA_RUNTIME_API 1
#endif
#endif

#ifndef CINDER_DETAIL_HAS_CUDA_RUNTIME_API
#define CINDER_DETAIL_HAS_CUDA_RUNTIME_API 0
#endif

namespace cinder
{
    /**
     * @brief CUDA 内存空间（CUDA memory space）。CUDA memory space selected by a CUDA allocator.
     */
    enum class CudaMemorySpace
    {
        /**
         * @brief 页锁定主机内存（page-locked host memory）。Page-locked host memory allocated by `cudaMallocHost`.
         */
        host,

        /**
         * @brief 设备内存（device memory）。Device memory allocated by `cudaMalloc`.
         */
        device,

        /**
         * @brief 统一内存（unified memory）。Unified/managed memory allocated by `cudaMallocManaged`.
         */
        managed
    };

    namespace detail
    {
        /**
         * @brief CUDA 分配器通用类型成员（common CUDA allocator type members）。Common type members for CUDA allocators.
         *
         * @tparam Value 被分配的值类型（allocated value type）。The allocated value type.
         */
        template <typename Value>
        class CudaAllocatorTypes
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
         * @brief 按 CUDA 内存空间分配字节存储（allocate byte storage by CUDA memory space）。Allocate byte storage for a CUDA memory space.
         *
         * @tparam MemorySpace CUDA 内存空间（CUDA memory space）。The CUDA memory space.
         * @param byte_count 字节数量（byte count）。The number of bytes to allocate.
         * @return 分配得到的字节存储指针（allocated byte-storage pointer）。The allocated byte-storage pointer.
         *
         * @note CUDA Runtime API 的分配函数是 host API；device 编译分支只保留防御性断言。CUDA Runtime
         *       allocation functions are host APIs; the device compilation branch only keeps a defensive
         *       assertion.
         */
        template <CudaMemorySpace MemorySpace>
        [[nodiscard]] CINDER_HOST_DEVICE auto cuda_allocate_bytes(std::size_t byte_count) -> void *
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            (void)byte_count;
            CINDER_ASSERT(false);
            return nullptr;
#elif CINDER_DETAIL_HAS_CUDA_RUNTIME_API
            void *memory{nullptr};
            cudaError_t status{};
            if constexpr (MemorySpace == CudaMemorySpace::host)
            {
                status = ::cudaMallocHost(&memory, byte_count);
            }
            else if constexpr (MemorySpace == CudaMemorySpace::device)
            {
                status = ::cudaMalloc(&memory, byte_count);
            }
            else
            {
                status = ::cudaMallocManaged(&memory, byte_count);
            }

            CINDER_ASSERT(status == cudaSuccess);
            if (status != cudaSuccess)
            {
                return nullptr;
            }
            return memory;
#else
            (void)byte_count;
            CINDER_ASSERT(false);
            return nullptr;
#endif
        }

        /**
         * @brief 按 CUDA 内存空间释放字节存储（deallocate byte storage by CUDA memory space）。Deallocate byte storage for a CUDA memory space.
         *
         * @tparam MemorySpace CUDA 内存空间（CUDA memory space）。The CUDA memory space.
         * @param storage 字节存储指针（byte-storage pointer）。The byte-storage pointer.
         *
         * @note `cudaMallocHost` 必须配对 `cudaFreeHost`；`cudaMalloc` 和 `cudaMallocManaged` 配对
         *       `cudaFree`。`cudaMallocHost` must be paired with `cudaFreeHost`; `cudaMalloc` and
         *       `cudaMallocManaged` are paired with `cudaFree`.
         */
        template <CudaMemorySpace MemorySpace>
        CINDER_HOST_DEVICE void cuda_deallocate_bytes(void *storage) noexcept
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            (void)storage;
            CINDER_ASSERT(false);
#elif CINDER_DETAIL_HAS_CUDA_RUNTIME_API
            cudaError_t status{};
            if constexpr (MemorySpace == CudaMemorySpace::host)
            {
                status = ::cudaFreeHost(storage);
            }
            else
            {
                status = ::cudaFree(storage);
            }

            CINDER_ASSERT(status == cudaSuccess);
            (void)status;
#else
            (void)storage;
            CINDER_ASSERT(false);
#endif
        }
    } // namespace detail

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

    /**
     * @brief CUDA Runtime API 分配器（CUDA Runtime API allocator）。Allocator backed by a selected CUDA Runtime API memory space.
     *
     * @tparam Value 被分配的值类型（allocated value type）。The allocated value type.
     * @tparam MemorySpace CUDA 内存空间（CUDA memory space）。The CUDA memory space.
     *
     * @note `CudaMemorySpace::host` 使用 `cudaMallocHost`/`cudaFreeHost`，申请 page-locked/pinned host
     *       memory，避免 pageable memory 传输时的临时 pinned staging copy。`CudaMemorySpace::host` uses
     *       `cudaMallocHost`/`cudaFreeHost` for page-locked/pinned host memory, avoiding the temporary
     *       pinned staging copy often needed for pageable-memory transfers.
     * @note `CudaMemorySpace::device` 使用 `cudaMalloc`/`cudaFree`。CUDA Runtime API 没有标准
     *       `cudaMallocDevice` 符号。`CudaMemorySpace::device` uses `cudaMalloc`/`cudaFree`; the CUDA
     *       Runtime API has no standard `cudaMallocDevice` symbol.
     * @note `CudaMemorySpace::managed` 使用 `cudaMallocManaged`/`cudaFree`，遵循 CUDA Unified Memory
     *       编程模型。`CudaMemorySpace::managed` uses `cudaMallocManaged`/`cudaFree` and follows the CUDA
     *       Unified Memory programming model.
     * @note 该 allocator 只描述存储驻留位置（storage residence）和分配/释放方式，不描述算术执行位置，也不
     *       隐式执行 host/device 数据迁移。This allocator only describes storage residence and
     *       allocation/deallocation; it does not describe where arithmetic executes and does not implicitly
     *       migrate data between host and device.
     */
    template <typename Value, CudaMemorySpace MemorySpace>
    class CudaAllocator : public detail::CudaAllocatorTypes<Value>
    {
        static_assert((MemorySpace == CudaMemorySpace::host) ||
                          (MemorySpace == CudaMemorySpace::device) ||
                          (MemorySpace == CudaMemorySpace::managed),
                      "Unsupported CUDA memory space.");

    private:
        using base_type = detail::CudaAllocatorTypes<Value>;

    public:
        /**
         * @brief 值类型（value type）。The allocated value type.
         */
        using value_type = typename base_type::value_type;

        /**
         * @brief 指针类型（pointer type）。The pointer type.
         */
        using pointer = typename base_type::pointer;

        /**
         * @brief 大小类型（size type）。The size type.
         */
        using size_type = typename base_type::size_type;

        /**
         * @brief 创建默认 CUDA 分配器（default CUDA allocator construction）。Create a default CUDA allocator.
         */
        constexpr CudaAllocator() noexcept = default;

        /**
         * @brief 从其他值类型的 CUDA 分配器创建（rebound CUDA allocator construction）。Create from a rebound CUDA allocator.
         *
         * @tparam Other 其他值类型（other value type）。The other value type.
         * @param other 其他分配器（other allocator）。The other allocator.
         */
        template <typename Other>
        CINDER_HOST_DEVICE constexpr CudaAllocator(const CudaAllocator<Other, MemorySpace> &other) noexcept
        {
            (void)other;
        }

        /**
         * @brief 分配 CUDA 存储（allocate CUDA storage）。Allocate CUDA storage.
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

            if (count > base_type::max_size())
            {
                CINDER_ASSERT(false);
                return nullptr;
            }

            const size_type byte_count = static_cast<size_type>(count * sizeof(value_type));
            return static_cast<pointer>(detail::cuda_allocate_bytes<MemorySpace>(byte_count));
        }

        /**
         * @brief 释放 CUDA 存储（deallocate CUDA storage）。Deallocate CUDA storage.
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

            detail::cuda_deallocate_bytes<MemorySpace>(storage);
        }
    };

    /**
     * @brief CUDA 页锁定主机内存分配器（CUDA page-locked host-memory allocator）。CUDA allocator backed by `cudaMallocHost`.
     *
     * @tparam Value 被分配的值类型（allocated value type）。The allocated value type.
     *
     * @note 该 allocator 返回 host 可直接解引用的 pinned memory（page-locked memory），适合作为 host/device
     *       传输 staging buffer。This allocator returns host-dereferenceable pinned memory, useful as a
     *       staging buffer for host/device transfers.
     * @note 使用该 allocator 的 `Tensor` 是 host-resident 容器；它不会自动代表 device-resident 张量。
     *       A `Tensor` using this allocator is a host-resident container; it does not automatically represent
     *       a device-resident tensor.
     */
    template <typename Value>
    using CudaHostAllocator = CudaAllocator<Value, CudaMemorySpace::host>;

    /**
     * @brief CUDA 设备内存分配器（CUDA device-memory allocator）。CUDA allocator backed by `cudaMalloc`.
     *
     * @tparam Value 被分配的值类型（allocated value type）。The allocated value type.
     *
     * @note 返回的 device pointer 通常不能由 host 直接解引用；需要由 kernel 或显式拷贝路径进行值构造与访问。
     *       The returned device pointer is generally not host-dereferenceable; value construction and access
     *       should happen through kernels or explicit copy paths.
     * @note 使用该 allocator 的 `Tensor` 是 device-resident 容器；host 侧对象可以持有其元数据和 device
     *       pointer，但跨 host/device 的值传输必须是显式操作。A `Tensor` using this allocator is a
     *       device-resident container; a host-side object may hold its metadata and device pointer, but value
     *       transfer across host/device must be explicit.
     */
    template <typename Value>
    using CudaDeviceAllocator = CudaAllocator<Value, CudaMemorySpace::device>;

    /**
     * @brief CUDA 统一内存分配器（CUDA unified-memory allocator）。CUDA allocator backed by `cudaMallocManaged`.
     *
     * @tparam Value 被分配的值类型（allocated value type）。The allocated value type.
     *
     * @note 统一内存（Unified Memory，也称 managed memory）在支持的系统上可由 CPU 和 GPU 访问，但访问顺序仍
     *       必须遵守 CUDA Unified Memory 编程模型。Unified Memory, also called managed memory, can be
     *       accessed by CPU and GPU on supported systems, but access ordering must still follow the CUDA
     *       Unified Memory programming model.
     * @note managed memory 改变可访问性（accessibility），不改变 `Tensor` 作为容器的职责边界；同步、预取和
     *       跨执行域使用仍应由调用方或显式 runtime 路径表达。Managed memory changes accessibility, not
     *       `Tensor`'s responsibility boundary as a container; synchronization, prefetching, and
     *       cross-execution-domain use should still be expressed by the caller or an explicit runtime path.
     */
    template <typename Value>
    using CudaManagedAllocator = CudaAllocator<Value, CudaMemorySpace::managed>;

    /**
     * @brief 比较 CUDA 分配器（CUDA allocator equality）。Compare CUDA allocators.
     *
     * @tparam Lhs 左值类型（left value type）。The left value type.
     * @tparam LhsMemorySpace 左 CUDA 内存空间（left CUDA memory space）。The left CUDA memory space.
     * @tparam Rhs 右值类型（right value type）。The right value type.
     * @tparam RhsMemorySpace 右 CUDA 内存空间（right CUDA memory space）。The right CUDA memory space.
     * @param lhs 左分配器（left allocator）。The left allocator.
     * @param rhs 右分配器（right allocator）。The right allocator.
     * @return 当两者使用同一 CUDA 内存空间时为 true。True when both allocators use the same CUDA memory space.
     */
    template <typename Lhs,
              CudaMemorySpace LhsMemorySpace,
              typename Rhs,
              CudaMemorySpace RhsMemorySpace>
    [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator==(
        const CudaAllocator<Lhs, LhsMemorySpace> &lhs,
        const CudaAllocator<Rhs, RhsMemorySpace> &rhs) noexcept -> bool
    {
        (void)lhs;
        (void)rhs;
        return LhsMemorySpace == RhsMemorySpace;
    }

} // namespace cinder

#undef CINDER_DETAIL_HAS_CUDA_RUNTIME_API
