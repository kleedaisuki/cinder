#include "cinder/tensor.cuh"

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

namespace
{
    /**
     * @brief 测试用数值类型（test numeric type）。A numeric type used by tests.
     */
    struct Scalar
    {
        /**
         * @brief 存储的整数值（stored integer value）。The stored integer value.
         */
        int value{};

        /**
         * @brief 默认创建零值（default zero construction）。Create a zero value by default.
         */
        constexpr Scalar() noexcept = default;

        /**
         * @brief 从整数创建值（integer construction）。Create a value from an integer.
         *
         * @param input 输入整数（input integer）。The input integer.
         */
        constexpr explicit Scalar(int input) noexcept
            : value{input}
        {
        }

        /**
         * @brief 比较两个值（equality comparison）。Compare two values.
         *
         * @param other 另一个值（other value）。The other value.
         * @return 若内部整数相等则为 true。True when stored integers are equal.
         */
        [[nodiscard]] constexpr auto operator==(const Scalar &other) const noexcept -> bool = default;
    };

    /**
     * @brief 加法运算（addition）。Add two scalar values.
     *
     * @param lhs 左操作数（left operand）。The left operand.
     * @param rhs 右操作数（right operand）。The right operand.
     * @return 两个值的和（sum of values）。The sum of the values.
     */
    [[nodiscard]] constexpr auto operator+(const Scalar &lhs, const Scalar &rhs) noexcept -> Scalar
    {
        return Scalar{lhs.value + rhs.value};
    }

    /**
     * @brief 减法运算（subtraction）。Subtract two scalar values.
     *
     * @param lhs 左操作数（left operand）。The left operand.
     * @param rhs 右操作数（right operand）。The right operand.
     * @return 两个值的差（difference of values）。The difference of the values.
     */
    [[nodiscard]] constexpr auto operator-(const Scalar &lhs, const Scalar &rhs) noexcept -> Scalar
    {
        return Scalar{lhs.value - rhs.value};
    }

    /**
     * @brief 乘法运算（multiplication）。Multiply two scalar values.
     *
     * @param lhs 左操作数（left operand）。The left operand.
     * @param rhs 右操作数（right operand）。The right operand.
     * @return 两个值的积（product of values）。The product of the values.
     */
    [[nodiscard]] constexpr auto operator*(const Scalar &lhs, const Scalar &rhs) noexcept -> Scalar
    {
        return Scalar{lhs.value * rhs.value};
    }

    /**
     * @brief 除法运算（division）。Divide two scalar values.
     *
     * @param lhs 左操作数（left operand）。The left operand.
     * @param rhs 右操作数（right operand）。The right operand.
     * @return 两个值的商（quotient of values）。The quotient of the values.
     */
    [[nodiscard]] constexpr auto operator/(const Scalar &lhs, const Scalar &rhs) noexcept -> Scalar
    {
        return Scalar{lhs.value / rhs.value};
    }

    /**
     * @brief 无显式存储大小的二维紧致布局（dense 2D layout without explicit storage size）。Dense 2D layout without explicit storage size.
     */
    class Dense2DLayout
    {
    public:
        /**
         * @brief 索引类型（index type）。The index type.
         */
        using index_type = std::size_t;

        /**
         * @brief 偏移类型（offset type）。The offset type.
         */
        using offset_type = std::size_t;

        /**
         * @brief 形状类型（shape type）。The shape type.
         */
        using shape_type = std::array<index_type, 2U>;

        /**
         * @brief 从形状创建布局（shape-based construction）。Create a layout from shape.
         *
         * @param shape 二维形状（two-dimensional shape）。The two-dimensional shape.
         */
        explicit constexpr Dense2DLayout(shape_type shape) noexcept
            : shape_{shape}
        {
        }

        /**
         * @brief 返回形状（shape access）。Return the shape.
         *
         * @return 形状常量引用（const reference to shape）。A const reference to the shape.
         */
        [[nodiscard]] constexpr auto shape() const noexcept -> const shape_type &
        {
            return shape_;
        }

        /**
         * @brief 映射二维索引到偏移（2D index to offset）。Map a two-dimensional index to offset.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 紧致行主序偏移（dense row-major offset）。The dense row-major offset.
         */
        [[nodiscard]] constexpr auto offset(cinder::IndexView<index_type> index) const noexcept
            -> offset_type
        {
            return (index[0] * shape_[1]) + index[1];
        }

    private:
        /**
         * @brief 二维形状（two-dimensional shape）。The two-dimensional shape.
         */
        shape_type shape_;
    };

    /**
     * @brief 分配统计信息（allocation statistics）。Allocation statistics.
     */
    struct AllocationStats
    {
        /**
         * @brief 分配次数（allocation count）。The allocation count.
         */
        std::size_t allocations{};

        /**
         * @brief 释放次数（deallocation count）。The deallocation count.
         */
        std::size_t deallocations{};

        /**
         * @brief 已分配元素总数（allocated element total）。The total number of allocated elements.
         */
        std::size_t allocated_elements{};

        /**
         * @brief 已释放元素总数（deallocated element total）。The total number of deallocated elements.
         */
        std::size_t deallocated_elements{};
    };

    /**
     * @brief 统计型分配器（counting allocator）。Allocator that counts allocate/deallocate calls.
     *
     * @tparam T 被分配对象类型（allocated object type）。The allocated object type.
     */
    template <typename T>
    class CountingAllocator
    {
    public:
        /**
         * @brief 分配器值类型（allocator value type）。The allocator value type.
         */
        using value_type = T;

        /**
         * @brief 创建带新统计对象的分配器（default counting allocator）。Create an allocator with new statistics.
         */
        CountingAllocator()
            : stats_{std::make_shared<AllocationStats>()}
        {
        }

        /**
         * @brief 创建共享统计对象的分配器（shared-stat allocator）。Create an allocator with shared statistics.
         *
         * @param stats 统计对象（statistics object）。The statistics object.
         */
        explicit CountingAllocator(std::shared_ptr<AllocationStats> stats) noexcept
            : stats_{std::move(stats)}
        {
        }

        /**
         * @brief 从其他值类型的统计分配器创建（rebound allocator construction）。Create from a rebound allocator.
         *
         * @tparam U 其他值类型（other value type）。The other value type.
         * @param other 其他分配器（other allocator）。The other allocator.
         */
        template <typename U>
        CountingAllocator(const CountingAllocator<U> &other) noexcept
            : stats_{other.stats_}
        {
        }

        /**
         * @brief 分配对象数组（allocate objects）。Allocate an object array.
         *
         * @param count 对象数量（object count）。The object count.
         * @return 分配得到的指针（allocated pointer）。The allocated pointer.
         */
        [[nodiscard]] auto allocate(std::size_t count) -> T *
        {
            ++stats_->allocations;
            stats_->allocated_elements += count;
            return std::allocator<T>{}.allocate(count);
        }

        /**
         * @brief 释放对象数组（deallocate objects）。Deallocate an object array.
         *
         * @param pointer 对象数组指针（object array pointer）。The object array pointer.
         * @param count 对象数量（object count）。The object count.
         */
        void deallocate(T *pointer, std::size_t count) noexcept
        {
            ++stats_->deallocations;
            stats_->deallocated_elements += count;
            std::allocator<T>{}.deallocate(pointer, count);
        }

        /**
         * @brief 比较统计对象身份（statistics identity comparison）。Compare statistics identity.
         *
         * @param lhs 左分配器（left allocator）。The left allocator.
         * @param rhs 右分配器（right allocator）。The right allocator.
         * @return 若共享同一统计对象则为 true。True when both allocators share the same statistics object.
         */
        [[nodiscard]] friend auto operator==(const CountingAllocator &lhs,
                                             const CountingAllocator &rhs) noexcept -> bool
        {
            return lhs.stats_ == rhs.stats_;
        }

    private:
        template <typename>
        friend class CountingAllocator;

        /**
         * @brief 共享统计对象（shared statistics object）。The shared statistics object.
         */
        std::shared_ptr<AllocationStats> stats_;
    };

    /**
     * @brief 固定 int 分配器（fixed int allocator）。Allocator fixed to int without rebind support.
     *
     * @note 该类型故意不是类模板，也不提供 `rebind`，用于验证 Tensor 不要求 allocator 自备
     *       rebinding。This type is intentionally not a class template and provides no `rebind`, verifying
     *       that Tensor does not require allocator rebinding.
     */
    class FixedIntAllocator
    {
    public:
        /**
         * @brief 分配器值类型（allocator value type）。The allocator value type.
         */
        using value_type = int;

        /**
         * @brief 分配对象数组（allocate objects）。Allocate an object array.
         *
         * @param count 对象数量（object count）。The object count.
         * @return 分配得到的指针（allocated pointer）。The allocated pointer.
         */
        [[nodiscard]] auto allocate(std::size_t count) -> int *
        {
            return std::allocator<int>{}.allocate(count);
        }

        /**
         * @brief 释放对象数组（deallocate objects）。Deallocate an object array.
         *
         * @param pointer 对象数组指针（object array pointer）。The object array pointer.
         * @param count 对象数量（object count）。The object count.
         */
        void deallocate(int *pointer, std::size_t count) noexcept
        {
            std::allocator<int>{}.deallocate(pointer, count);
        }
    };

    static_assert(cinder::ArithmeticLike<int>);
    static_assert(cinder::ArithmeticLike<double>);
    static_assert(cinder::ArithmeticLike<Scalar>);
    static_assert(cinder::LayoutLike<cinder::RowMajorLayout<2>>);
    static_assert(cinder::LayoutLike<Dense2DLayout>);
    static_assert(cinder::AllocatorLike<cinder::DefaultAllocator<int>>);
    static_assert(cinder::AllocatorLike<CountingAllocator<int>>);
    static_assert(cinder::AllocatorLike<FixedIntAllocator>);
    static_assert(cinder::TensorLike<cinder::Tensor<int, cinder::RowMajorLayout<2>>>);

    /**
     * @brief 测试二维整数张量（test two-dimensional integer tensor）。Test a two-dimensional integer tensor.
     */
    void test_integer_tensor()
    {
        using Layout = cinder::RowMajorLayout<2>;
        cinder::Tensor<int, Layout> tensor{Layout{2U, 3U}, {0, 1, 2, 3, 4, 5}};

        assert(tensor.size() == 6U);
        assert(tensor.rank() == 2U);
        assert(tensor(0, 0) == 0);
        assert(tensor(1, 2) == 5);

        tensor(1, 2) = 42;
        assert(tensor.at(1, 2) == 42);

        const int values[]{10, 11, 12, 13, 14, 15};
        cinder::Tensor<int, Layout> from_pointer{Layout{2U, 3U}, values, 6U};
        assert(from_pointer(1, 2) == 15);
    }

    /**
     * @brief 测试用户自定义算术类型（test user-defined arithmetic type）。Test a user-defined arithmetic type.
     */
    void test_custom_arithmetic_value()
    {
        using Layout = cinder::RowMajorLayout<1>;
        cinder::Tensor<Scalar, Layout> tensor{Layout{3U}, Scalar{7}};

        tensor(1) = Scalar{11};
        assert(tensor(0) + tensor(1) == Scalar{18});
        assert(tensor(1) * tensor(2) == Scalar{77});
    }

    /**
     * @brief 测试最小 LayoutLike 布局（test minimal LayoutLike layout）。Test a minimal LayoutLike layout.
     */
    void test_minimal_layout_without_storage_size()
    {
        cinder::Tensor<int, Dense2DLayout> tensor{
            Dense2DLayout{Dense2DLayout::shape_type{2U, 2U}},
            {1, 2, 3, 4}};

        assert(tensor.size() == 4U);
        assert(tensor(1, 0) == 3);
    }

    /**
     * @brief 测试 allocator traits 路径（test allocator-traits path）。Test allocator-traits allocation path.
     */
    void test_allocator_traits_path()
    {
        using Layout = cinder::RowMajorLayout<2>;
        using Allocator = CountingAllocator<int>;

        const auto stats = std::make_shared<AllocationStats>();
        {
            cinder::Tensor<int, Layout, Allocator> tensor{Layout{2U, 2U}, 3, Allocator{stats}};
            assert(tensor.size() == 4U);
            assert(tensor(1, 1) == 3);
            assert(stats->allocations == 1U);
            assert(stats->allocated_elements == 4U);
        }

        assert(stats->deallocations == 1U);
        assert(stats->deallocated_elements == 4U);

        cinder::Tensor<int, Layout, FixedIntAllocator> fixed_allocator_tensor{
            Layout{1U, 2U},
            {8, 9},
            FixedIntAllocator{}};
        assert(fixed_allocator_tensor(0, 1) == 9);
    }
} // namespace

/**
 * @brief 运行 Tensor 冒烟测试（run Tensor smoke tests）。Run Tensor smoke tests.
 *
 * @return 进程退出码（process exit code）。The process exit code.
 */
auto main() -> int
{
    test_integer_tensor();
    test_custom_arithmetic_value();
    test_minimal_layout_without_storage_size();
    test_allocator_traits_path();
    return 0;
}
