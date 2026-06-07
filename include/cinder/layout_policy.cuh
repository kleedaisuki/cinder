#pragma once

#include "cinder/concepts.cuh"

#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace cinder
{
    /**
     * @brief 默认布局阶数容量（default layout rank capacity）。Default rank capacity for dynamic-rank layouts.
     *
     * @note 布局的实际阶数（rank）仍在运行时确定；该值只限制无堆分配的小型存储容量。
     *       The actual rank remains runtime-selected; this value only bounds fixed-capacity storage.
     */
    inline constexpr std::size_t default_layout_rank_capacity = 8U;

    /**
     * @brief 固定容量布局向量（fixed-capacity layout vector）。A host/device-safe vector for layout metadata.
     *
     * @tparam Capacity 最大元素数量（maximum element count）。The maximum number of stored values.
     * @tparam Value 无符号值类型（unsigned value type）。The unsigned value type.
     *
     * @note 该类型用 `std::array` 存储数据并用运行时 `size()` 表达有效长度，避免把 layout 绑定到
     *       host-only heap 容器。This type stores data in `std::array` and keeps a runtime `size()`, avoiding
     *       a dependency on host-only heap containers for layout metadata.
     */
    template <std::size_t Capacity,
              std::unsigned_integral Value = std::size_t>
    class LayoutVector
    {
    public:
        /**
         * @brief 元素值类型（element value type）。The element value type.
         */
        using value_type = Value;

        /**
         * @brief 大小类型（size type）。The size type.
         */
        using size_type = std::size_t;

        /**
         * @brief 常量迭代器类型（const iterator type）。The const iterator type.
         */
        using const_iterator = const value_type *;

        /**
         * @brief 创建空布局向量（empty layout vector）。Create an empty layout vector.
         */
        CINDER_HOST_DEVICE constexpr LayoutVector() noexcept = default;

        /**
         * @brief 从索引视图创建布局向量（IndexView construction）。Create a layout vector from an IndexView.
         *
         * @param values 输入值视图（input value view）。The input values.
         */
        CINDER_HOST_DEVICE explicit constexpr LayoutVector(IndexView<value_type> values) noexcept
        {
            assign(values.data(), values.size());
        }

        /**
         * @brief 从初始化列表创建布局向量（initializer-list construction）。Create a layout vector from an initializer list.
         *
         * @param values 输入值列表（input value list）。The input values.
         */
        CINDER_HOST_DEVICE constexpr LayoutVector(std::initializer_list<value_type> values) noexcept
        {
            assign(values.begin(), values.size());
        }

        /**
         * @brief 从 `std::array` 创建布局向量（std::array construction）。Create a layout vector from std::array.
         *
         * @tparam Count 输入数组长度（input array length）。The input array length.
         * @param values 输入数组（input array）。The input array.
         */
        template <std::size_t Count>
        CINDER_HOST_DEVICE explicit constexpr LayoutVector(const std::array<value_type, Count> &values) noexcept
        {
            assign(values.data(), Count);
        }

        /**
         * @brief 从独立值创建布局向量（variadic construction）。Create a layout vector from separate values.
         *
         * @tparam Values 输入值类型（input value types）。The input value types.
         * @param values 输入值（input values）。The input values.
         */
        template <typename... Values>
            requires(sizeof...(Values) > 0U) &&
                    (sizeof...(Values) <= Capacity) &&
                    (std::convertible_to<Values, value_type> && ...)
        CINDER_HOST_DEVICE explicit constexpr LayoutVector(Values... values) noexcept
            : data_{static_cast<value_type>(values)...},
              size_{sizeof...(Values)}
        {
        }

        /**
         * @brief 重新赋值布局向量（assign layout vector）。Assign values to the layout vector.
         *
         * @param values 输入值首地址（input value pointer）。The pointer to the first input value.
         * @param value_count 输入值数量（input value count）。The number of input values.
         */
        CINDER_HOST_DEVICE constexpr void assign(const value_type *values, size_type value_count) noexcept
        {
            if (value_count > Capacity)
            {
                CINDER_ASSERT(false);
                size_ = size_type{0U};
                return;
            }
            if ((value_count != size_type{0U}) && (values == nullptr))
            {
                CINDER_ASSERT(false);
                size_ = size_type{0U};
                return;
            }

            for (size_type index{0U}; index < value_count; ++index)
            {
                data_[index] = values[index];
            }
            size_ = value_count;
        }

        /**
         * @brief 返回有效元素数量（size access）。Return the number of stored values.
         *
         * @return 有效元素数量（stored value count）。The number of stored values.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto size() const noexcept -> size_type
        {
            return size_;
        }

        /**
         * @brief 返回容量（capacity access）。Return the maximum capacity.
         *
         * @return 最大元素数量（maximum element count）。The maximum number of values.
         */
        [[nodiscard]] CINDER_HOST_DEVICE static constexpr auto capacity() noexcept -> size_type
        {
            return Capacity;
        }

        /**
         * @brief 判断是否为空（empty check）。Return whether no value is stored.
         *
         * @return 若没有有效元素则为 true。True when no value is stored.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto empty() const noexcept -> bool
        {
            return size_ == size_type{0U};
        }

        /**
         * @brief 读取指定位置的值（indexed value access）。Read a value at an index.
         *
         * @param index 元素位置（element index）。The element index.
         * @return 指定位置的值（value at the requested index）。The value at the requested index.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator[](size_type index) const noexcept -> value_type
        {
            if (index >= size_)
            {
                CINDER_ASSERT(false);
                return value_type{};
            }
            return data_[index];
        }

        /**
         * @brief 返回首元素迭代器（begin iterator）。Return the begin iterator.
         *
         * @return 首元素指针（pointer to first value）。Pointer to the first value.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto begin() const noexcept -> const_iterator
        {
            return data_.data();
        }

        /**
         * @brief 返回尾后迭代器（end iterator）。Return the end iterator.
         *
         * @return 尾后指针（one-past-the-end pointer）。Pointer one past the last value.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto end() const noexcept -> const_iterator
        {
            return data_.data() + size_;
        }

        /**
         * @brief 返回原始数据指针（raw data pointer）。Return the raw data pointer.
         *
         * @return 首元素指针（pointer to first stored slot）。Pointer to the first storage slot.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto data() const noexcept -> const value_type *
        {
            return data_.data();
        }

    private:
        /**
         * @brief 固定容量存储（fixed-capacity storage）。The fixed-capacity storage.
         */
        std::array<value_type, Capacity> data_{};

        /**
         * @brief 有效元素数量（stored value count）。The number of stored values.
         */
        size_type size_{0U};
    };

    namespace detail
    {
        /**
         * @brief 检查加法溢出并返回结果（checked unsigned addition）。Add unsigned values with overflow checking.
         *
         * @tparam Value 无符号值类型（unsigned value type）。The unsigned value type.
         * @param lhs 左操作数（left operand）。The left operand.
         * @param rhs 右操作数（right operand）。The right operand.
         * @return 加法结果（sum）。The sum.
         */
        template <std::unsigned_integral Value>
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto checked_add(Value lhs, Value rhs) noexcept -> Value
        {
            if (lhs > (std::numeric_limits<Value>::max() - rhs))
            {
                CINDER_ASSERT(false);
                return Value{0U};
            }
            return static_cast<Value>(lhs + rhs);
        }

        /**
         * @brief 检查乘法溢出并返回结果（checked unsigned multiplication）。Multiply unsigned values with overflow checking.
         *
         * @tparam Value 无符号值类型（unsigned value type）。The unsigned value type.
         * @param lhs 左操作数（left operand）。The left operand.
         * @param rhs 右操作数（right operand）。The right operand.
         * @return 乘法结果（product）。The product.
         */
        template <std::unsigned_integral Value>
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto checked_multiply(Value lhs, Value rhs) noexcept -> Value
        {
            if ((rhs != Value{0U}) && (lhs > (std::numeric_limits<Value>::max() / rhs)))
            {
                CINDER_ASSERT(false);
                return Value{0U};
            }
            return static_cast<Value>(lhs * rhs);
        }

        /**
         * @brief 上取整除法（ceiling division）。Divide and round up.
         *
         * @tparam Value 无符号值类型（unsigned value type）。The unsigned value type.
         * @param dividend 被除数（dividend）。The dividend.
         * @param divisor 除数（divisor）。The divisor.
         * @return 上取整商（rounded-up quotient）。The rounded-up quotient.
         */
        template <std::unsigned_integral Value>
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto ceil_div(Value dividend, Value divisor) noexcept -> Value
        {
            if (divisor == Value{0U})
            {
                CINDER_ASSERT(false);
                return Value{0U};
            }
            if (dividend == Value{0U})
            {
                return Value{0U};
            }
            return static_cast<Value>(((dividend - Value{1U}) / divisor) + Value{1U});
        }

        /**
         * @brief 返回形状长度（shape size helper）。Return shape size.
         *
         * @tparam Shape 形状范围类型（shape range type）。The shape range type.
         * @param shape 形状范围（shape range）。The shape range.
         * @return 形状长度（shape size）。The shape size.
         */
        template <typename Shape>
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto shape_size(const Shape &shape) noexcept -> std::size_t
        {
            static_assert(requires {
                              shape.size();
                          },
                          "Tensor layout shape must provide size() for host/device storage checks.");
            return static_cast<std::size_t>(shape.size());
        }

        /**
         * @brief 按轴读取形状长度（shape extent helper）。Read one shape extent by axis.
         *
         * @tparam Shape 形状范围类型（shape range type）。The shape range type.
         * @param shape 形状范围（shape range）。The shape range.
         * @param axis 轴编号（axis number）。The axis number.
         * @return 指定轴的长度（extent on the requested axis）。The extent on the requested axis.
         */
        template <typename Shape>
        [[nodiscard]] CINDER_HOST_DEVICE constexpr decltype(auto) shape_extent(const Shape &shape,
                                                                              std::size_t axis) noexcept
        {
            static_assert(requires {
                              shape[axis];
                          },
                          "Tensor layout shape must provide operator[] for host/device storage checks.");
            return shape[axis];
        }

        /**
         * @brief 带显式存储大小的布局（layout with explicit storage size）。Layout with explicit storage size.
         *
         * @tparam Layout 布局类型（layout type）。The layout type.
         */
        template <typename Layout>
        concept HasStorageSize =
            LayoutLike<Layout> &&
            requires(const std::remove_cvref_t<Layout> &layout) {
                { layout.storage_size() } -> std::convertible_to<layout_offset_t<Layout>>;
            };

        /**
         * @brief 从形状计算紧致存储大小（dense storage size）。Compute dense storage size from shape.
         *
         * @tparam Layout 布局类型（layout type）。The layout type.
         * @param layout 布局对象（layout object）。The layout object.
         * @return 紧致线性元素数量（dense linear element count）。The dense linear element count.
         */
        template <LayoutLike Layout>
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto dense_storage_size(const Layout &layout) noexcept
            -> layout_offset_t<Layout>
        {
            using offset_type = layout_offset_t<Layout>;

            offset_type total{1U};
            const auto &shape = layout.shape();
            const std::size_t rank = shape_size(shape);
            for (std::size_t axis{0U}; axis < rank; ++axis)
            {
                const auto extent = static_cast<offset_type>(shape_extent(shape, axis));
                if (extent == offset_type{0U})
                {
                    return offset_type{0U};
                }
                if (total > (std::numeric_limits<offset_type>::max() / extent))
                {
                    CINDER_ASSERT(false);
                    return offset_type{0U};
                }
                total = checked_multiply(total, extent);
            }

            return total;
        }

        /**
         * @brief 取得布局需要的线性存储大小（layout storage size）。Return required layout storage size.
         *
         * @tparam Layout 布局类型（layout type）。The layout type.
         * @param layout 布局对象（layout object）。The layout object.
         * @return 线性元素数量（linear element count）。The linear element count.
         *
         * @note 若布局提供 `storage_size()`，优先使用它；否则使用 `shape()` 的乘积作为紧致存储大小。
         *       If the layout provides `storage_size()`, it is used; otherwise the product of `shape()`
         *       is used as dense storage size.
         */
        template <LayoutLike Layout>
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto layout_storage_size(const Layout &layout) noexcept
            -> layout_offset_t<Layout>
        {
            if constexpr (HasStorageSize<Layout>)
            {
                return static_cast<layout_offset_t<Layout>>(layout.storage_size());
            }
            else
            {
                return dense_storage_size(layout);
            }
        }

        /**
         * @brief 判断多重索引是否位于形状内（shape bounds check）。Check whether a multi-index is inside shape.
         *
         * @tparam Layout 布局类型（layout type）。The layout type.
         * @param layout 布局对象（layout object）。The layout object.
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 若索引长度和各轴坐标均合法则为 true。True when rank and coordinates are in bounds.
         */
        template <LayoutLike Layout>
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto index_in_shape(
            const Layout &layout,
            IndexView<layout_index_t<Layout>> index) noexcept -> bool
        {
            const auto &shape = layout.shape();
            if (index.size() != shape_size(shape))
            {
                return false;
            }

            for (std::size_t axis{0U}; axis < index.size(); ++axis)
            {
                const auto extent = static_cast<layout_index_t<Layout>>(shape_extent(shape, axis));
                if (index[axis] >= extent)
                {
                    return false;
                }
            }

            return true;
        }
    } // namespace detail

    /**
     * @brief 行主序动态阶数布局（row-major dynamic-rank layout）。Row-major dynamic-rank layout.
     *
     * @tparam MaxRank 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
     * @tparam Index 自然数索引类型（natural-number index type）。The index type.
     * @tparam Offset 线性偏移类型（linear offset type）。The offset type.
     *
     * @note 该布局把最后一维作为连续维度（contiguous dimension），即 C/C++ 常见的 row-major 规则。
     *       实际阶数由构造时传入的 shape 长度确定。This layout makes the last axis contiguous, following
     *       the usual C/C++ row-major rule. The actual rank is determined by the runtime shape length.
     */
    template <std::size_t MaxRank = default_layout_rank_capacity,
              std::unsigned_integral Index = std::size_t,
              std::unsigned_integral Offset = std::size_t>
    class RowMajorLayout
    {
    public:
        /**
         * @brief 索引值类型（index value type）。The index value type.
         */
        using index_type = Index;

        /**
         * @brief 线性偏移类型（linear offset type）。The linear offset type.
         */
        using offset_type = Offset;

        /**
         * @brief 形状类型（shape type）。The shape type.
         */
        using shape_type = LayoutVector<MaxRank, index_type>;

        /**
         * @brief 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
         */
        static constexpr std::size_t max_rank_value = MaxRank;

        /**
         * @brief 创建标量形状布局（scalar-shape layout）。Create a scalar-shape layout.
         */
        CINDER_HOST_DEVICE constexpr RowMajorLayout() noexcept
        {
            rebuild_strides();
        }

        /**
         * @brief 从形状创建布局（shape-based layout construction）。Create a layout from shape.
         *
         * @param shape 每一维的长度（extent per axis）。The extent of each axis.
         */
        CINDER_HOST_DEVICE explicit constexpr RowMajorLayout(shape_type shape) noexcept
            : shape_{shape}
        {
            rebuild_strides();
        }

        /**
         * @brief 从形状视图创建布局（shape-view construction）。Create a layout from a shape view.
         *
         * @param shape 每一维的长度视图（extent view per axis）。The extent view for each axis.
         */
        CINDER_HOST_DEVICE explicit constexpr RowMajorLayout(IndexView<index_type> shape) noexcept
            : RowMajorLayout{shape_type{shape}}
        {
        }

        /**
         * @brief 从各轴长度创建布局（axis extents construction）。Create a layout from axis extents.
         *
         * @tparam Extents 轴长度参数类型（axis extent argument types）。The axis extent argument types.
         * @param extents 各轴长度（axis extents）。The axis extents.
         */
        template <typename... Extents>
            requires(sizeof...(Extents) > 0U) &&
                    (sizeof...(Extents) <= MaxRank) &&
                    (std::convertible_to<Extents, index_type> && ...)
        CINDER_HOST_DEVICE explicit constexpr RowMajorLayout(Extents... extents) noexcept
            : RowMajorLayout{shape_type{static_cast<index_type>(extents)...}}
        {
        }

        /**
         * @brief 返回形状（shape access）。Return the shape.
         *
         * @return 形状数组的常量引用（const reference to shape array）。A const reference to the shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto shape() const noexcept -> const shape_type &
        {
            return shape_;
        }

        /**
         * @brief 返回阶数（rank access）。Return the rank.
         *
         * @return 运行时阶数（runtime rank）。The runtime rank.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto rank() const noexcept -> std::size_t
        {
            return shape_.size();
        }

        /**
         * @brief 将多重索引映射到线性偏移（multi-index to linear offset）。Map a multi-index to offset.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 行主序线性偏移（row-major linear offset）。The row-major linear offset.
         *
         * @note 该函数不做边界检查（bounds checking），调用者必须保证索引长度和各轴坐标合法。
         *       This function performs no bounds checking; callers must ensure rank and coordinates are valid.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto offset(IndexView<index_type> index) const noexcept
            -> offset_type
        {
            if (index.size() != shape_.size())
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type linear_offset{0U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                const offset_type coordinate = static_cast<offset_type>(index[axis]);
                const offset_type contribution = detail::checked_multiply(coordinate, strides_[axis]);
                linear_offset = detail::checked_add(linear_offset, contribution);
            }
            return linear_offset;
        }

    private:
        /**
         * @brief 重建步幅（stride rebuild）。Rebuild strides.
         */
        CINDER_HOST_DEVICE constexpr void rebuild_strides() noexcept
        {
            offset_type stride{1U};
            for (std::size_t reversed_axis{shape_.size()}; reversed_axis > 0U; --reversed_axis)
            {
                const std::size_t axis = reversed_axis - 1U;
                strides_[axis] = stride;
                stride = detail::checked_multiply(stride, static_cast<offset_type>(shape_[axis]));
            }
        }

        /**
         * @brief 每一维的长度（extent per axis）。The extent of each axis.
         */
        shape_type shape_{};

        /**
         * @brief 每一维的行主序步幅（row-major stride per axis）。The row-major stride of each axis.
         */
        std::array<offset_type, MaxRank> strides_{};
    };

    /**
     * @brief 列主序动态阶数布局（column-major dynamic-rank layout）。Column-major dynamic-rank layout.
     *
     * @tparam MaxRank 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
     * @tparam Index 自然数索引类型（natural-number index type）。The index type.
     * @tparam Offset 线性偏移类型（linear offset type）。The offset type.
     *
     * @note 该布局把第一维作为连续维度（contiguous dimension），符合 Fortran/BLAS 常见 column-major
     *       规则。This layout makes the first axis contiguous, following the common Fortran/BLAS
     *       column-major rule.
     */
    template <std::size_t MaxRank = default_layout_rank_capacity,
              std::unsigned_integral Index = std::size_t,
              std::unsigned_integral Offset = std::size_t>
    class ColumnMajorLayout
    {
    public:
        /**
         * @brief 索引值类型（index value type）。The index value type.
         */
        using index_type = Index;

        /**
         * @brief 线性偏移类型（linear offset type）。The linear offset type.
         */
        using offset_type = Offset;

        /**
         * @brief 形状类型（shape type）。The shape type.
         */
        using shape_type = LayoutVector<MaxRank, index_type>;

        /**
         * @brief 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
         */
        static constexpr std::size_t max_rank_value = MaxRank;

        /**
         * @brief 创建标量形状布局（scalar-shape layout）。Create a scalar-shape layout.
         */
        CINDER_HOST_DEVICE constexpr ColumnMajorLayout() noexcept
        {
            rebuild_strides();
        }

        /**
         * @brief 从形状创建布局（shape-based layout construction）。Create a layout from shape.
         *
         * @param shape 每一维的长度（extent per axis）。The extent of each axis.
         */
        CINDER_HOST_DEVICE explicit constexpr ColumnMajorLayout(shape_type shape) noexcept
            : shape_{shape}
        {
            rebuild_strides();
        }

        /**
         * @brief 从形状视图创建布局（shape-view construction）。Create a layout from a shape view.
         *
         * @param shape 每一维的长度视图（extent view per axis）。The extent view for each axis.
         */
        CINDER_HOST_DEVICE explicit constexpr ColumnMajorLayout(IndexView<index_type> shape) noexcept
            : ColumnMajorLayout{shape_type{shape}}
        {
        }

        /**
         * @brief 从各轴长度创建布局（axis extents construction）。Create a layout from axis extents.
         *
         * @tparam Extents 轴长度参数类型（axis extent argument types）。The axis extent argument types.
         * @param extents 各轴长度（axis extents）。The axis extents.
         */
        template <typename... Extents>
            requires(sizeof...(Extents) > 0U) &&
                    (sizeof...(Extents) <= MaxRank) &&
                    (std::convertible_to<Extents, index_type> && ...)
        CINDER_HOST_DEVICE explicit constexpr ColumnMajorLayout(Extents... extents) noexcept
            : ColumnMajorLayout{shape_type{static_cast<index_type>(extents)...}}
        {
        }

        /**
         * @brief 返回形状（shape access）。Return the shape.
         *
         * @return 形状向量的常量引用（const reference to shape vector）。A const reference to the shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto shape() const noexcept -> const shape_type &
        {
            return shape_;
        }

        /**
         * @brief 返回阶数（rank access）。Return the rank.
         *
         * @return 运行时阶数（runtime rank）。The runtime rank.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto rank() const noexcept -> std::size_t
        {
            return shape_.size();
        }

        /**
         * @brief 将多重索引映射到线性偏移（multi-index to linear offset）。Map a multi-index to offset.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 列主序线性偏移（column-major linear offset）。The column-major linear offset.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto offset(IndexView<index_type> index) const noexcept
            -> offset_type
        {
            if (index.size() != shape_.size())
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type linear_offset{0U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                const offset_type coordinate = static_cast<offset_type>(index[axis]);
                const offset_type contribution = detail::checked_multiply(coordinate, strides_[axis]);
                linear_offset = detail::checked_add(linear_offset, contribution);
            }
            return linear_offset;
        }

    private:
        /**
         * @brief 重建步幅（stride rebuild）。Rebuild strides.
         */
        CINDER_HOST_DEVICE constexpr void rebuild_strides() noexcept
        {
            offset_type stride{1U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                strides_[axis] = stride;
                stride = detail::checked_multiply(stride, static_cast<offset_type>(shape_[axis]));
            }
        }

        /**
         * @brief 每一维的长度（extent per axis）。The extent of each axis.
         */
        shape_type shape_{};

        /**
         * @brief 每一维的列主序步幅（column-major stride per axis）。The column-major stride of each axis.
         */
        std::array<offset_type, MaxRank> strides_{};
    };

    /**
     * @brief 显式步幅动态阶数布局（explicit-stride dynamic-rank layout）。Dynamic-rank layout with explicit strides.
     *
     * @tparam MaxRank 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
     * @tparam Index 自然数索引类型（natural-number index type）。The index type.
     * @tparam Offset 线性偏移类型（linear offset type）。The offset type.
     *
     * @note 该布局允许非紧致存储（non-contiguous storage）和别名（aliasing）。存储大小按可达最大偏移
     *       加一计算。This layout permits non-contiguous storage and aliasing. Storage size is computed as
     *       one plus the maximum reachable offset.
     */
    template <std::size_t MaxRank = default_layout_rank_capacity,
              std::unsigned_integral Index = std::size_t,
              std::unsigned_integral Offset = std::size_t>
    class StridedLayout
    {
    public:
        /**
         * @brief 索引值类型（index value type）。The index value type.
         */
        using index_type = Index;

        /**
         * @brief 线性偏移类型（linear offset type）。The linear offset type.
         */
        using offset_type = Offset;

        /**
         * @brief 形状类型（shape type）。The shape type.
         */
        using shape_type = LayoutVector<MaxRank, index_type>;

        /**
         * @brief 步幅类型（stride type）。The stride type.
         */
        using stride_type = LayoutVector<MaxRank, offset_type>;

        /**
         * @brief 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
         */
        static constexpr std::size_t max_rank_value = MaxRank;

        /**
         * @brief 创建标量布局（scalar layout）。Create a scalar layout.
         */
        CINDER_HOST_DEVICE constexpr StridedLayout() noexcept = default;

        /**
         * @brief 从形状和步幅创建布局（shape-and-stride construction）。Create a layout from shape and strides.
         *
         * @param shape 每一维的长度（extent per axis）。The extent of each axis.
         * @param strides 每一维的线性步幅（linear stride per axis）。The linear stride of each axis.
         */
        CINDER_HOST_DEVICE constexpr StridedLayout(shape_type shape, stride_type strides) noexcept
            : shape_{shape},
              strides_{strides}
        {
            validate_rank_match();
        }

        /**
         * @brief 从形状和步幅视图创建布局（view construction）。Create a layout from shape and stride views.
         *
         * @param shape 每一维的长度视图（extent view per axis）。The extent view for each axis.
         * @param strides 每一维的线性步幅视图（linear stride view per axis）。The stride view for each axis.
         */
        CINDER_HOST_DEVICE constexpr StridedLayout(IndexView<index_type> shape,
                                                   IndexView<offset_type> strides) noexcept
            : StridedLayout{shape_type{shape}, stride_type{strides}}
        {
        }

        /**
         * @brief 返回形状（shape access）。Return the shape.
         *
         * @return 形状向量的常量引用（const reference to shape vector）。A const reference to the shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto shape() const noexcept -> const shape_type &
        {
            return shape_;
        }

        /**
         * @brief 返回步幅（stride access）。Return the strides.
         *
         * @return 步幅向量的常量引用（const reference to stride vector）。A const reference to the strides.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto strides() const noexcept -> const stride_type &
        {
            return strides_;
        }

        /**
         * @brief 返回阶数（rank access）。Return the rank.
         *
         * @return 运行时阶数（runtime rank）。The runtime rank.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto rank() const noexcept -> std::size_t
        {
            return shape_.size();
        }

        /**
         * @brief 将多重索引映射到线性偏移（multi-index to linear offset）。Map a multi-index to offset.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 显式步幅线性偏移（strided linear offset）。The strided linear offset.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto offset(IndexView<index_type> index) const noexcept
            -> offset_type
        {
            if ((index.size() != shape_.size()) || (strides_.size() != shape_.size()))
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type linear_offset{0U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                const offset_type coordinate = static_cast<offset_type>(index[axis]);
                const offset_type contribution = detail::checked_multiply(coordinate, strides_[axis]);
                linear_offset = detail::checked_add(linear_offset, contribution);
            }
            return linear_offset;
        }

        /**
         * @brief 返回需要的线性存储大小（storage size）。Return required linear storage size.
         *
         * @return 可覆盖所有合法索引偏移的元素数量（element count covering all valid offsets）。The element count covering all valid offsets.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto storage_size() const noexcept -> offset_type
        {
            if (strides_.size() != shape_.size())
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type maximum_offset{0U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                const index_type extent = shape_[axis];
                if (extent == index_type{0U})
                {
                    return offset_type{0U};
                }

                const offset_type last_coordinate = static_cast<offset_type>(extent - index_type{1U});
                const offset_type contribution = detail::checked_multiply(last_coordinate, strides_[axis]);
                maximum_offset = detail::checked_add(maximum_offset, contribution);
            }

            return detail::checked_add(maximum_offset, offset_type{1U});
        }

    private:
        /**
         * @brief 校验形状和步幅阶数一致（rank match validation）。Validate that shape and strides have the same rank.
         */
        CINDER_HOST_DEVICE constexpr void validate_rank_match() const noexcept
        {
            CINDER_ASSERT(shape_.size() == strides_.size());
        }

        /**
         * @brief 每一维的长度（extent per axis）。The extent of each axis.
         */
        shape_type shape_{};

        /**
         * @brief 每一维的显式步幅（explicit stride per axis）。The explicit stride of each axis.
         */
        stride_type strides_{};
    };

    /**
     * @brief 填充行主序动态阶数布局（padded row-major dynamic-rank layout）。Dynamic-rank row-major layout with padded extents.
     *
     * @tparam MaxRank 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
     * @tparam Index 自然数索引类型（natural-number index type）。The index type.
     * @tparam Offset 线性偏移类型（linear offset type）。The offset type.
     *
     * @note `shape()` 返回逻辑形状（logical shape），`padded_shape()` 返回参与地址计算和存储大小计算的物理形状
     *       （physical shape）。`shape()` returns the logical shape; `padded_shape()` returns the physical shape
     *       used for address and storage-size calculation.
     */
    template <std::size_t MaxRank = default_layout_rank_capacity,
              std::unsigned_integral Index = std::size_t,
              std::unsigned_integral Offset = std::size_t>
    class PaddedLayout
    {
    public:
        /**
         * @brief 索引值类型（index value type）。The index value type.
         */
        using index_type = Index;

        /**
         * @brief 线性偏移类型（linear offset type）。The linear offset type.
         */
        using offset_type = Offset;

        /**
         * @brief 形状类型（shape type）。The shape type.
         */
        using shape_type = LayoutVector<MaxRank, index_type>;

        /**
         * @brief 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
         */
        static constexpr std::size_t max_rank_value = MaxRank;

        /**
         * @brief 创建标量布局（scalar layout）。Create a scalar layout.
         */
        CINDER_HOST_DEVICE constexpr PaddedLayout() noexcept
        {
            rebuild_strides();
        }

        /**
         * @brief 从逻辑形状创建无额外填充布局（dense shape construction）。Create an unpadded layout from logical shape.
         *
         * @param shape 每一维的逻辑长度（logical extent per axis）。The logical extent of each axis.
         */
        CINDER_HOST_DEVICE explicit constexpr PaddedLayout(shape_type shape) noexcept
            : PaddedLayout{shape, shape}
        {
        }

        /**
         * @brief 从逻辑形状和物理形状创建布局（logical-and-physical construction）。Create a layout from logical and physical shapes.
         *
         * @param shape 每一维的逻辑长度（logical extent per axis）。The logical extent of each axis.
         * @param padded_shape 每一维的物理长度（physical extent per axis）。The physical extent of each axis.
         */
        CINDER_HOST_DEVICE constexpr PaddedLayout(shape_type shape, shape_type padded_shape) noexcept
            : shape_{shape},
              padded_shape_{padded_shape}
        {
            validate_padded_shape();
            rebuild_strides();
        }

        /**
         * @brief 从逻辑和物理形状视图创建布局（view construction）。Create a layout from logical and physical shape views.
         *
         * @param shape 每一维的逻辑长度视图（logical extent view per axis）。The logical extent view.
         * @param padded_shape 每一维的物理长度视图（physical extent view per axis）。The physical extent view.
         */
        CINDER_HOST_DEVICE constexpr PaddedLayout(IndexView<index_type> shape,
                                                  IndexView<index_type> padded_shape) noexcept
            : PaddedLayout{shape_type{shape}, shape_type{padded_shape}}
        {
        }

        /**
         * @brief 返回逻辑形状（logical shape access）。Return the logical shape.
         *
         * @return 逻辑形状常量引用（const reference to logical shape）。A const reference to the logical shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto shape() const noexcept -> const shape_type &
        {
            return shape_;
        }

        /**
         * @brief 返回物理形状（physical shape access）。Return the physical padded shape.
         *
         * @return 物理形状常量引用（const reference to physical shape）。A const reference to the physical shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto padded_shape() const noexcept -> const shape_type &
        {
            return padded_shape_;
        }

        /**
         * @brief 返回阶数（rank access）。Return the rank.
         *
         * @return 运行时阶数（runtime rank）。The runtime rank.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto rank() const noexcept -> std::size_t
        {
            return shape_.size();
        }

        /**
         * @brief 将多重索引映射到线性偏移（multi-index to linear offset）。Map a multi-index to offset.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 带填充行主序线性偏移（padded row-major linear offset）。The padded row-major linear offset.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto offset(IndexView<index_type> index) const noexcept
            -> offset_type
        {
            if ((index.size() != shape_.size()) || (padded_shape_.size() != shape_.size()))
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type linear_offset{0U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                const offset_type coordinate = static_cast<offset_type>(index[axis]);
                const offset_type contribution = detail::checked_multiply(coordinate, strides_[axis]);
                linear_offset = detail::checked_add(linear_offset, contribution);
            }
            return linear_offset;
        }

        /**
         * @brief 返回需要的线性存储大小（storage size）。Return required linear storage size.
         *
         * @return 物理形状覆盖的元素数量（element count covered by the physical shape）。The element count covered by the physical shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto storage_size() const noexcept -> offset_type
        {
            if (padded_shape_.size() != shape_.size())
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type total{1U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                if (shape_[axis] == index_type{0U})
                {
                    return offset_type{0U};
                }
                total = detail::checked_multiply(total, static_cast<offset_type>(padded_shape_[axis]));
            }
            return total;
        }

    private:
        /**
         * @brief 校验物理形状可覆盖逻辑形状（padded shape validation）。Validate that physical shape covers logical shape.
         */
        CINDER_HOST_DEVICE constexpr void validate_padded_shape() const noexcept
        {
            CINDER_ASSERT(shape_.size() == padded_shape_.size());
            const std::size_t rank = shape_.size() < padded_shape_.size() ? shape_.size() : padded_shape_.size();
            for (std::size_t axis{0U}; axis < rank; ++axis)
            {
                CINDER_ASSERT(padded_shape_[axis] >= shape_[axis]);
            }
        }

        /**
         * @brief 重建物理步幅（physical stride rebuild）。Rebuild physical strides.
         */
        CINDER_HOST_DEVICE constexpr void rebuild_strides() noexcept
        {
            offset_type stride{1U};
            for (std::size_t reversed_axis{padded_shape_.size()}; reversed_axis > 0U; --reversed_axis)
            {
                const std::size_t axis = reversed_axis - 1U;
                strides_[axis] = stride;
                stride = detail::checked_multiply(stride, static_cast<offset_type>(padded_shape_[axis]));
            }
        }

        /**
         * @brief 每一维的逻辑长度（logical extent per axis）。The logical extent of each axis.
         */
        shape_type shape_{};

        /**
         * @brief 每一维的物理长度（physical extent per axis）。The physical extent of each axis.
         */
        shape_type padded_shape_{};

        /**
         * @brief 每一维的物理行主序步幅（physical row-major stride per axis）。The physical row-major stride of each axis.
         */
        std::array<offset_type, MaxRank> strides_{};
    };

    /**
     * @brief 分块/平铺动态阶数布局（blocked/tiled dynamic-rank layout）。Dynamic-rank blocked/tiled layout.
     *
     * @tparam MaxRank 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
     * @tparam Index 自然数索引类型（natural-number index type）。The index type.
     * @tparam Offset 线性偏移类型（linear offset type）。The offset type.
     *
     * @note 存储顺序为 tile-major：先按行主序排列 tile 坐标，再按行主序排列 tile 内坐标。Storage is
     *       tile-major: tile coordinates are row-major, followed by row-major coordinates inside each tile.
     */
    template <std::size_t MaxRank = default_layout_rank_capacity,
              std::unsigned_integral Index = std::size_t,
              std::unsigned_integral Offset = std::size_t>
    class BlockedLayout
    {
    public:
        /**
         * @brief 索引值类型（index value type）。The index value type.
         */
        using index_type = Index;

        /**
         * @brief 线性偏移类型（linear offset type）。The linear offset type.
         */
        using offset_type = Offset;

        /**
         * @brief 形状类型（shape type）。The shape type.
         */
        using shape_type = LayoutVector<MaxRank, index_type>;

        /**
         * @brief 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
         */
        static constexpr std::size_t max_rank_value = MaxRank;

        /**
         * @brief 创建标量布局（scalar layout）。Create a scalar layout.
         */
        CINDER_HOST_DEVICE constexpr BlockedLayout() noexcept = default;

        /**
         * @brief 从形状和 tile 形状创建布局（shape-and-tile construction）。Create a layout from shape and tile shape.
         *
         * @param shape 每一维的逻辑长度（logical extent per axis）。The logical extent of each axis.
         * @param tile_shape 每一维的 tile 长度（tile extent per axis）。The tile extent of each axis.
         */
        CINDER_HOST_DEVICE constexpr BlockedLayout(shape_type shape, shape_type tile_shape) noexcept
            : shape_{shape},
              tile_shape_{tile_shape}
        {
            validate_tile_shape();
        }

        /**
         * @brief 从形状和 tile 视图创建布局（view construction）。Create a layout from shape and tile views.
         *
         * @param shape 每一维的逻辑长度视图（logical extent view per axis）。The logical extent view.
         * @param tile_shape 每一维的 tile 长度视图（tile extent view per axis）。The tile extent view.
         */
        CINDER_HOST_DEVICE constexpr BlockedLayout(IndexView<index_type> shape,
                                                   IndexView<index_type> tile_shape) noexcept
            : BlockedLayout{shape_type{shape}, shape_type{tile_shape}}
        {
        }

        /**
         * @brief 返回逻辑形状（logical shape access）。Return the logical shape.
         *
         * @return 逻辑形状常量引用（const reference to logical shape）。A const reference to the logical shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto shape() const noexcept -> const shape_type &
        {
            return shape_;
        }

        /**
         * @brief 返回 tile 形状（tile shape access）。Return the tile shape.
         *
         * @return tile 形状常量引用（const reference to tile shape）。A const reference to the tile shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto tile_shape() const noexcept -> const shape_type &
        {
            return tile_shape_;
        }

        /**
         * @brief 返回阶数（rank access）。Return the rank.
         *
         * @return 运行时阶数（runtime rank）。The runtime rank.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto rank() const noexcept -> std::size_t
        {
            return shape_.size();
        }

        /**
         * @brief 将多重索引映射到线性偏移（multi-index to linear offset）。Map a multi-index to offset.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 分块线性偏移（blocked linear offset）。The blocked linear offset.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto offset(IndexView<index_type> index) const noexcept
            -> offset_type
        {
            if ((index.size() != shape_.size()) || (tile_shape_.size() != shape_.size()))
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type tile_linear{0U};
            offset_type intra_tile_linear{0U};
            offset_type tile_volume{1U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                const offset_type extent = static_cast<offset_type>(shape_[axis]);
                const offset_type tile_extent = static_cast<offset_type>(tile_shape_[axis]);
                const offset_type coordinate = static_cast<offset_type>(index[axis]);
                const offset_type tile_grid_extent = detail::ceil_div(extent, tile_extent);
                const offset_type tile_coordinate = static_cast<offset_type>(coordinate / tile_extent);
                const offset_type intra_tile_coordinate = static_cast<offset_type>(coordinate % tile_extent);

                tile_linear = detail::checked_add(
                    detail::checked_multiply(tile_linear, tile_grid_extent),
                    tile_coordinate);
                intra_tile_linear = detail::checked_add(
                    detail::checked_multiply(intra_tile_linear, tile_extent),
                    intra_tile_coordinate);
                tile_volume = detail::checked_multiply(tile_volume, tile_extent);
            }

            return detail::checked_add(
                detail::checked_multiply(tile_linear, tile_volume),
                intra_tile_linear);
        }

        /**
         * @brief 返回需要的线性存储大小（storage size）。Return required linear storage size.
         *
         * @return 覆盖完整 tile 网格的元素数量（element count covering the full tile grid）。The element count covering the full tile grid.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto storage_size() const noexcept -> offset_type
        {
            if (tile_shape_.size() != shape_.size())
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type total{1U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                const index_type extent = shape_[axis];
                const index_type tile_extent = tile_shape_[axis];
                if (extent == index_type{0U})
                {
                    return offset_type{0U};
                }
                if (tile_extent == index_type{0U})
                {
                    CINDER_ASSERT(false);
                    return offset_type{0U};
                }

                const offset_type tile_grid_extent = detail::ceil_div(
                    static_cast<offset_type>(extent),
                    static_cast<offset_type>(tile_extent));
                const offset_type physical_extent = detail::checked_multiply(
                    tile_grid_extent,
                    static_cast<offset_type>(tile_extent));
                total = detail::checked_multiply(total, physical_extent);
            }
            return total;
        }

    private:
        /**
         * @brief 校验 tile 形状合法（tile shape validation）。Validate the tile shape.
         */
        CINDER_HOST_DEVICE constexpr void validate_tile_shape() const noexcept
        {
            CINDER_ASSERT(shape_.size() == tile_shape_.size());
            const std::size_t rank = shape_.size() < tile_shape_.size() ? shape_.size() : tile_shape_.size();
            for (std::size_t axis{0U}; axis < rank; ++axis)
            {
                CINDER_ASSERT(tile_shape_[axis] != index_type{0U});
            }
        }

        /**
         * @brief 每一维的逻辑长度（logical extent per axis）。The logical extent of each axis.
         */
        shape_type shape_{};

        /**
         * @brief 每一维的 tile 长度（tile extent per axis）。The tile extent of each axis.
         */
        shape_type tile_shape_{};
    };

    /**
     * @brief 平铺布局别名（tiled layout alias）。Alias for the blocked/tiled layout implementation.
     *
     * @tparam MaxRank 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
     * @tparam Index 自然数索引类型（natural-number index type）。The index type.
     * @tparam Offset 线性偏移类型（linear offset type）。The offset type.
     */
    template <std::size_t MaxRank = default_layout_rank_capacity,
              std::unsigned_integral Index = std::size_t,
              std::unsigned_integral Offset = std::size_t>
    using TiledLayout = BlockedLayout<MaxRank, Index, Offset>;

    /**
     * @brief 广播动态阶数布局（broadcast dynamic-rank layout）。Dynamic-rank broadcast layout.
     *
     * @tparam MaxRank 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
     * @tparam Index 自然数索引类型（natural-number index type）。The index type.
     * @tparam Offset 线性偏移类型（linear offset type）。The offset type.
     *
     * @note `shape()` 是逻辑输出形状，`source_shape()` 是实际存储形状。源轴长度为 1 时，该轴所有逻辑坐标
     *       都映射到源坐标 0。`shape()` is the logical output shape, while `source_shape()` is the physical
     *       source shape. When a source axis has extent 1, every logical coordinate on that axis maps to
     *       source coordinate 0.
     */
    template <std::size_t MaxRank = default_layout_rank_capacity,
              std::unsigned_integral Index = std::size_t,
              std::unsigned_integral Offset = std::size_t>
    class BroadcastLayout
    {
    public:
        /**
         * @brief 索引值类型（index value type）。The index value type.
         */
        using index_type = Index;

        /**
         * @brief 线性偏移类型（linear offset type）。The linear offset type.
         */
        using offset_type = Offset;

        /**
         * @brief 形状类型（shape type）。The shape type.
         */
        using shape_type = LayoutVector<MaxRank, index_type>;

        /**
         * @brief 最大阶数容量（maximum rank capacity）。The maximum rank capacity.
         */
        static constexpr std::size_t max_rank_value = MaxRank;

        /**
         * @brief 创建标量广播布局（scalar broadcast layout）。Create a scalar broadcast layout.
         */
        CINDER_HOST_DEVICE constexpr BroadcastLayout() noexcept
        {
            rebuild_source_strides();
        }

        /**
         * @brief 从形状创建无广播布局（identity broadcast construction）。Create an identity broadcast layout.
         *
         * @param shape 每一维的逻辑长度（logical extent per axis）。The logical extent of each axis.
         */
        CINDER_HOST_DEVICE explicit constexpr BroadcastLayout(shape_type shape) noexcept
            : BroadcastLayout{shape, shape}
        {
        }

        /**
         * @brief 从逻辑形状和源形状创建广播布局（logical-and-source construction）。Create a broadcast layout.
         *
         * @param shape 每一维的逻辑长度（logical extent per axis）。The logical extent of each axis.
         * @param source_shape 每一维的源长度（source extent per axis）。The source extent of each axis.
         */
        CINDER_HOST_DEVICE constexpr BroadcastLayout(shape_type shape, shape_type source_shape) noexcept
            : shape_{shape},
              source_shape_{source_shape}
        {
            validate_source_shape();
            rebuild_source_strides();
        }

        /**
         * @brief 从逻辑和源形状视图创建广播布局（view construction）。Create a broadcast layout from shape views.
         *
         * @param shape 每一维的逻辑长度视图（logical extent view per axis）。The logical extent view.
         * @param source_shape 每一维的源长度视图（source extent view per axis）。The source extent view.
         */
        CINDER_HOST_DEVICE constexpr BroadcastLayout(IndexView<index_type> shape,
                                                     IndexView<index_type> source_shape) noexcept
            : BroadcastLayout{shape_type{shape}, shape_type{source_shape}}
        {
        }

        /**
         * @brief 返回逻辑形状（logical shape access）。Return the logical shape.
         *
         * @return 逻辑形状常量引用（const reference to logical shape）。A const reference to the logical shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto shape() const noexcept -> const shape_type &
        {
            return shape_;
        }

        /**
         * @brief 返回源形状（source shape access）。Return the source shape.
         *
         * @return 源形状常量引用（const reference to source shape）。A const reference to the source shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto source_shape() const noexcept -> const shape_type &
        {
            return source_shape_;
        }

        /**
         * @brief 返回阶数（rank access）。Return the rank.
         *
         * @return 运行时阶数（runtime rank）。The runtime rank.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto rank() const noexcept -> std::size_t
        {
            return shape_.size();
        }

        /**
         * @brief 将多重索引映射到线性偏移（multi-index to linear offset）。Map a multi-index to offset.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 广播源存储偏移（broadcast source storage offset）。The offset in source storage.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto offset(IndexView<index_type> index) const noexcept
            -> offset_type
        {
            if ((index.size() != shape_.size()) || (source_shape_.size() != shape_.size()))
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type linear_offset{0U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                const index_type source_extent = source_shape_[axis];
                const index_type source_coordinate = source_extent == index_type{1U}
                                                         ? index_type{0U}
                                                         : index[axis];
                const offset_type contribution = detail::checked_multiply(
                    static_cast<offset_type>(source_coordinate),
                    source_strides_[axis]);
                linear_offset = detail::checked_add(linear_offset, contribution);
            }
            return linear_offset;
        }

        /**
         * @brief 返回需要的线性存储大小（storage size）。Return required linear storage size.
         *
         * @return 源形状需要的元素数量（element count required by source shape）。The element count required by the source shape.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto storage_size() const noexcept -> offset_type
        {
            if (source_shape_.size() != shape_.size())
            {
                CINDER_ASSERT(false);
                return offset_type{0U};
            }

            offset_type total{1U};
            for (std::size_t axis{0U}; axis < shape_.size(); ++axis)
            {
                if (shape_[axis] == index_type{0U})
                {
                    return offset_type{0U};
                }
                total = detail::checked_multiply(total, static_cast<offset_type>(source_shape_[axis]));
            }
            return total;
        }

    private:
        /**
         * @brief 校验源形状可广播到逻辑形状（source shape validation）。Validate that source shape broadcasts to logical shape.
         */
        CINDER_HOST_DEVICE constexpr void validate_source_shape() const noexcept
        {
            CINDER_ASSERT(shape_.size() == source_shape_.size());
            const std::size_t rank = shape_.size() < source_shape_.size() ? shape_.size() : source_shape_.size();
            for (std::size_t axis{0U}; axis < rank; ++axis)
            {
                const index_type logical_extent = shape_[axis];
                const index_type source_extent = source_shape_[axis];
                if (logical_extent != index_type{0U})
                {
                    CINDER_ASSERT((source_extent == index_type{1U}) || (source_extent == logical_extent));
                }
            }
        }

        /**
         * @brief 重建源行主序步幅（source stride rebuild）。Rebuild row-major strides for source storage.
         */
        CINDER_HOST_DEVICE constexpr void rebuild_source_strides() noexcept
        {
            offset_type stride{1U};
            for (std::size_t reversed_axis{source_shape_.size()}; reversed_axis > 0U; --reversed_axis)
            {
                const std::size_t axis = reversed_axis - 1U;
                source_strides_[axis] = stride;
                stride = detail::checked_multiply(stride, static_cast<offset_type>(source_shape_[axis]));
            }
        }

        /**
         * @brief 每一维的逻辑长度（logical extent per axis）。The logical extent of each axis.
         */
        shape_type shape_{};

        /**
         * @brief 每一维的源长度（source extent per axis）。The source extent of each axis.
         */
        shape_type source_shape_{};

        /**
         * @brief 源存储的行主序步幅（source row-major stride per axis）。The row-major stride of source storage.
         */
        std::array<offset_type, MaxRank> source_strides_{};
    };

} // namespace cinder
