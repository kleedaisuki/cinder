#pragma once

#include "cinder/concepts.cuh"

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <type_traits>

namespace cinder
{
    namespace detail
    {
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
        [[nodiscard]] auto dense_storage_size(const Layout &layout) -> layout_offset_t<Layout>
        {
            using offset_type = layout_offset_t<Layout>;

            offset_type total{1U};
            for (const auto extent_ref : layout.shape())
            {
                const auto extent = static_cast<offset_type>(extent_ref);
                if (extent == offset_type{0U})
                {
                    return offset_type{0U};
                }
                if (total > (std::numeric_limits<offset_type>::max() / extent))
                {
                    throw std::length_error{"cinder::Tensor layout shape overflows offset_type"};
                }
                total = static_cast<offset_type>(total * extent);
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
        [[nodiscard]] auto layout_storage_size(const Layout &layout) -> layout_offset_t<Layout>
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
        [[nodiscard]] auto index_in_shape(const Layout &layout,
                                          IndexView<layout_index_t<Layout>> index) -> bool
        {
            if (index.size() != std::ranges::size(layout.shape()))
            {
                return false;
            }

            std::size_t axis{0U};
            for (const auto extent_ref : layout.shape())
            {
                const auto extent = static_cast<layout_index_t<Layout>>(extent_ref);
                if (index[axis] >= extent)
                {
                    return false;
                }
                ++axis;
            }

            return true;
        }
    } // namespace detail

    /**
     * @brief 行主序静态阶数布局（row-major static-rank layout）。Row-major static-rank layout.
     *
     * @tparam Rank 阶数/维数（rank/dimension）。The tensor rank.
     * @tparam Index 自然数索引类型（natural-number index type）。The index type.
     * @tparam Offset 线性偏移类型（linear offset type）。The offset type.
     *
     * @note 该布局把最后一维作为连续维度（contiguous dimension），即 C/C++ 常见的 row-major 规则。
     *       This layout makes the last axis contiguous, following the usual C/C++ row-major rule.
     */
    template <std::size_t Rank,
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
        using shape_type = std::array<index_type, Rank>;

        /**
         * @brief 静态阶数值（static rank value）。The static rank value.
         */
        static constexpr std::size_t rank_value = Rank;

        /**
         * @brief 创建零形状布局（zero-shape layout）。Create a zero-shape layout.
         *
         * @note 对于正阶张量，默认形状含零长度轴。For positive-rank tensors, the default shape contains
         *       zero-length axes.
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
         * @brief 从各轴长度创建布局（axis extents construction）。Create a layout from axis extents.
         *
         * @tparam Extents 轴长度参数类型（axis extent argument types）。The axis extent argument types.
         * @param extents 各轴长度（axis extents）。The axis extents.
         */
        template <typename... Extents>
            requires((sizeof...(Extents) == Rank) &&
                     (std::convertible_to<Extents, index_type> && ...))
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
         * @return 静态阶数（static rank）。The static rank.
         */
        [[nodiscard]] CINDER_HOST_DEVICE static constexpr auto rank() noexcept -> std::size_t
        {
            return Rank;
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
            offset_type linear_offset{0U};
            for (std::size_t axis{0U}; axis < Rank; ++axis)
            {
                linear_offset = static_cast<offset_type>(
                    linear_offset + (static_cast<offset_type>(index[axis]) * strides_[axis]));
            }
            return linear_offset;
        }

    private:
        /**
         * @brief 重建步幅（stride rebuild）。Rebuild strides.
         */
        CINDER_HOST_DEVICE constexpr void rebuild_strides() noexcept
        {
            if constexpr (Rank > 0U)
            {
                offset_type stride{1U};
                for (std::size_t reversed_axis{Rank}; reversed_axis > 0U; --reversed_axis)
                {
                    const std::size_t axis = reversed_axis - 1U;
                    strides_[axis] = stride;
                    stride = static_cast<offset_type>(stride * static_cast<offset_type>(shape_[axis]));
                }
            }
        }

        /**
         * @brief 每一维的长度（extent per axis）。The extent of each axis.
         */
        shape_type shape_{};

        /**
         * @brief 每一维的行主序步幅（row-major stride per axis）。The row-major stride of each axis.
         */
        std::array<offset_type, Rank> strides_{};
    };

} // namespace cinder
