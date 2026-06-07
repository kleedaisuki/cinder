#pragma once

#include "cinder/allocator_policy.cuh"
#include "cinder/concepts.cuh"
#include "cinder/layout_policy.cuh"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cinder
{
    /**
     * @brief 拥有线性存储的策略化张量（policy-based owning tensor）。Policy-based owning tensor with linear storage.
     *
     * @tparam Value 运算值类型（arithmetic value type）。The arithmetic value type.
     * @tparam Layout 布局策略类型（layout policy type）。The layout policy type.
     * @tparam Allocator 分配器策略类型（allocator policy type）。The allocator policy type.
     *
     * @note `Layout` 负责把多重索引（multi-index）映射为线性偏移（linear offset）；`Allocator` 通过
     *       `std::allocator_traits` 申请、构造、销毁和释放线性内存。`Layout` maps multi-indices to
     *       linear offsets; `Allocator` is mediated by `std::allocator_traits` for allocation,
     *       construction, destruction, and deallocation.
     */
    template <typename Value,
              typename Layout,
              typename Allocator = DefaultAllocator<Value>>
        requires ArithmeticLike<Value> &&
                 LayoutLike<Layout> &&
                 AllocatorLike<Allocator> &&
                 std::same_as<std::remove_cvref_t<Value>,
                              typename std::remove_cvref_t<Allocator>::value_type>
    class Tensor
    {
    public:
        /**
         * @brief 值类型（value type）。The stored value type.
         */
        using value_type = std::remove_cvref_t<Value>;

        /**
         * @brief 布局类型（layout type）。The layout type.
         */
        using layout_type = std::remove_cvref_t<Layout>;

        /**
         * @brief 分配器类型（allocator type）。The allocator type.
         */
        using allocator_type = std::remove_cvref_t<Allocator>;

        /**
         * @brief 分配器 traits 类型（allocator traits type）。The allocator traits type.
         */
        using allocator_traits = std::allocator_traits<allocator_type>;

        /**
         * @brief 分配器指针类型（allocator pointer type）。The allocator pointer type.
         */
        using pointer = typename allocator_traits::pointer;

        /**
         * @brief 引用类型（reference type）。The reference type.
         */
        using reference = value_type &;

        /**
         * @brief 常量引用类型（const reference type）。The const reference type.
         */
        using const_reference = const value_type &;

        /**
         * @brief 自然数索引类型（natural-number index type）。The natural-number index type.
         */
        using index_type = typename layout_type::index_type;

        /**
         * @brief 线性偏移类型（linear offset type）。The linear offset type.
         */
        using offset_type = typename layout_type::offset_type;

        /**
         * @brief 线性存储大小类型（linear storage size type）。The linear storage size type.
         */
        using size_type = typename allocator_traits::size_type;

        /**
         * @brief 创建默认张量（default tensor）。Create a default tensor.
         *
         * @note 仅当布局和分配器都可默认构造（default constructible）时可用。This overload is available
         *       only when both layout and allocator are default constructible.
         */
        Tensor()
            requires(std::default_initializable<layout_type> &&
                     std::default_initializable<allocator_type>)
            : Tensor{layout_type{}, allocator_type{}}
        {
        }

        /**
         * @brief 按布局创建值初始化张量（value-initialized tensor）。Create a value-initialized tensor.
         *
         * @param layout 布局对象（layout object）。The layout object.
         * @param allocator 分配器对象（allocator object）。The allocator object.
         */
        explicit Tensor(layout_type layout,
                        const allocator_type &allocator = allocator_type{})
            : layout_{std::move(layout)},
              allocator_{allocator}
        {
            initialize_default(checked_storage_size(layout_, allocator_));
        }

        /**
         * @brief 按布局和值创建填充张量（filled tensor）。Create a tensor filled with one value.
         *
         * @param layout 布局对象（layout object）。The layout object.
         * @param value 填充值（fill value）。The fill value.
         * @param allocator 分配器对象（allocator object）。The allocator object.
         */
        Tensor(layout_type layout,
               const value_type &value,
               const allocator_type &allocator = allocator_type{})
            : layout_{std::move(layout)},
              allocator_{allocator}
        {
            initialize_fill(checked_storage_size(layout_, allocator_), value);
        }

        /**
         * @brief 从初始化列表创建张量（initializer-list tensor construction）。Create a tensor from an initializer list.
         *
         * @param layout 布局对象（layout object）。The layout object.
         * @param values 初始值列表（initializer list）。The initializer list.
         * @param allocator 分配器对象（allocator object）。The allocator object.
         */
        Tensor(layout_type layout,
               std::initializer_list<value_type> values,
               const allocator_type &allocator = allocator_type{})
            : layout_{std::move(layout)},
              allocator_{allocator}
        {
            const size_type element_count = checked_storage_size(layout_, allocator_);
            if (values.size() != static_cast<std::size_t>(element_count))
            {
                throw std::length_error{"cinder::Tensor initializer count does not match layout storage size"};
            }
            initialize_range(element_count, values);
        }

        /**
         * @brief 从输入范围创建张量（range-based tensor construction）。Create a tensor from an input range.
         *
         * @tparam Range 输入范围类型（input range type）。The input range type.
         * @param layout 布局对象（layout object）。The layout object.
         * @param values 初始值范围（initial value range）。The initial value range.
         * @param allocator 分配器对象（allocator object）。The allocator object.
         */
        template <std::ranges::input_range Range>
            requires(!std::same_as<std::remove_cvref_t<Range>, Tensor>) &&
                        std::convertible_to<std::ranges::range_reference_t<Range>, value_type>
        Tensor(layout_type layout,
               Range &&values,
               const allocator_type &allocator = allocator_type{})
            : layout_{std::move(layout)},
              allocator_{allocator}
        {
            initialize_range(checked_storage_size(layout_, allocator_), std::forward<Range>(values));
        }

        /**
         * @brief 拷贝构造张量（copy construction）。Copy-construct a tensor.
         *
         * @param other 源张量（source tensor）。The source tensor.
         */
        Tensor(const Tensor &other)
            : layout_{other.layout_},
              allocator_{allocator_traits::select_on_container_copy_construction(other.allocator_)}
        {
            initialize_copy(other.size_, other.raw_data_unchecked());
        }

        /**
         * @brief 移动构造张量（move construction）。Move-construct a tensor.
         *
         * @param other 源张量（source tensor）。The source tensor.
         */
        Tensor(Tensor &&other) noexcept(std::is_nothrow_move_constructible_v<layout_type> &&
                                        std::is_nothrow_move_constructible_v<allocator_type>)
            : layout_{std::move(other.layout_)},
              allocator_{std::move(other.allocator_)},
              data_{std::exchange(other.data_, pointer{})},
              size_{std::exchange(other.size_, size_type{0U})}
        {
        }

        /**
         * @brief 销毁张量（destruction）。Destroy the tensor.
         */
        ~Tensor()
        {
            release();
        }

        /**
         * @brief 拷贝赋值张量（copy assignment）。Copy-assign a tensor.
         *
         * @param other 源张量（source tensor）。The source tensor.
         * @return 当前张量引用（reference to this tensor）。A reference to this tensor.
         */
        auto operator=(const Tensor &other) -> Tensor &
        {
            if (this == &other)
            {
                return *this;
            }

            if constexpr (allocator_traits::propagate_on_container_copy_assignment::value)
            {
                Tensor temporary{other.layout_, other.size_, other.raw_data_unchecked(), other.allocator_};
                release();
                allocator_ = other.allocator_;
                adopt_storage_from(temporary);
            }
            else
            {
                Tensor temporary{other.layout_, other.size_, other.raw_data_unchecked(), allocator_};
                release();
                adopt_storage_from(temporary);
            }

            return *this;
        }

        /**
         * @brief 移动赋值张量（move assignment）。Move-assign a tensor.
         *
         * @param other 源张量（source tensor）。The source tensor.
         * @return 当前张量引用（reference to this tensor）。A reference to this tensor.
         */
        auto operator=(Tensor &&other) -> Tensor &
        {
            if (this == &other)
            {
                return *this;
            }

            if constexpr (allocator_traits::propagate_on_container_move_assignment::value)
            {
                release();
                allocator_ = std::move(other.allocator_);
                adopt_storage_from(other);
            }
            else if constexpr (allocator_traits::is_always_equal::value)
            {
                release();
                adopt_storage_from(other);
            }
            else
            {
                Tensor temporary{other.layout_, allocator_};
                move_assign_values_to(temporary, other);
                release();
                adopt_storage_from(temporary);
            }

            return *this;
        }

        /**
         * @brief 返回形状（shape access）。Return the tensor shape.
         *
         * @return 布局暴露的形状范围（shape range exposed by layout）。The shape range exposed by layout.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr decltype(auto) shape() const noexcept
        {
            return layout_.shape();
        }

        /**
         * @brief 返回布局（layout access）。Return the layout.
         *
         * @return 布局常量引用（const reference to layout）。A const reference to the layout.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto layout() const noexcept -> const layout_type &
        {
            return layout_;
        }

        /**
         * @brief 返回分配器（allocator access）。Return the allocator.
         *
         * @return 分配器常量引用（const reference to allocator）。A const reference to the allocator.
         */
        [[nodiscard]] auto allocator() const noexcept -> const allocator_type &
        {
            return allocator_;
        }

        /**
         * @brief 返回线性元素数量（linear element count）。Return the linear element count.
         *
         * @return 已分配并构造的元素数量（allocated and constructed element count）。The number of allocated and constructed elements.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto size() const noexcept -> size_type
        {
            return size_;
        }

        /**
         * @brief 判断张量是否为空（empty check）。Return whether the tensor has no linear elements.
         *
         * @return 若线性元素数量为零则为 true。True when the linear element count is zero.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto empty() const noexcept -> bool
        {
            return size_ == size_type{0U};
        }

        /**
         * @brief 返回阶数（rank access）。Return the tensor rank.
         *
         * @return 形状范围大小（shape range size）。The size of the shape range.
         */
        [[nodiscard]] auto rank() const noexcept -> std::size_t
        {
            return std::ranges::size(shape());
        }

        /**
         * @brief 返回原始数据指针（raw data pointer）。Return the raw data pointer.
         *
         * @return 首元素指针，空张量返回 nullptr。Pointer to the first element, or nullptr for an empty tensor.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto data() noexcept -> value_type *
        {
            return raw_data_unchecked();
        }

        /**
         * @brief 返回原始数据指针（const raw data pointer）。Return the const raw data pointer.
         *
         * @return 首元素常量指针，空张量返回 nullptr。Const pointer to the first element, or nullptr for an empty tensor.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto data() const noexcept -> const value_type *
        {
            return raw_data_unchecked();
        }

        /**
         * @brief 返回线性存储起点（begin pointer）。Return begin pointer of linear storage.
         *
         * @return 首元素指针（pointer to first element）。Pointer to the first element.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto begin() noexcept -> value_type *
        {
            return data();
        }

        /**
         * @brief 返回线性存储起点（const begin pointer）。Return const begin pointer of linear storage.
         *
         * @return 首元素常量指针（const pointer to first element）。Const pointer to the first element.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto begin() const noexcept -> const value_type *
        {
            return data();
        }

        /**
         * @brief 返回线性存储终点（end pointer）。Return end pointer of linear storage.
         *
         * @return 尾后元素指针（one-past-the-end pointer）。Pointer one past the last element.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto end() noexcept -> value_type *
        {
            value_type *const current = data();
            return current == nullptr ? nullptr : current + size_;
        }

        /**
         * @brief 返回线性存储终点（const end pointer）。Return const end pointer of linear storage.
         *
         * @return 尾后元素常量指针（const one-past-the-end pointer）。Const pointer one past the last element.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto end() const noexcept -> const value_type *
        {
            const value_type *const current = data();
            return current == nullptr ? nullptr : current + size_;
        }

        /**
         * @brief 按多重索引访问元素（multi-index element access）。Access an element by multi-index.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 元素引用（element reference）。A reference to the element.
         *
         * @note 该函数不做边界检查（bounds checking），用于性能敏感路径。This function performs no bounds
         *       checking and is intended for performance-sensitive paths.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator()(IndexView<index_type> index) noexcept
            -> reference
        {
            return raw_data_unchecked()[static_cast<size_type>(layout_.offset(index))];
        }

        /**
         * @brief 按多重索引访问元素（const multi-index element access）。Access an element by multi-index.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 元素常量引用（const element reference）。A const reference to the element.
         *
         * @note 该函数不做边界检查（bounds checking），用于性能敏感路径。This function performs no bounds
         *       checking and is intended for performance-sensitive paths.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto operator()(IndexView<index_type> index) const noexcept
            -> const_reference
        {
            return raw_data_unchecked()[static_cast<size_type>(layout_.offset(index))];
        }

        /**
         * @brief 按独立坐标访问元素（variadic index access）。Access an element by separate coordinates.
         *
         * @tparam Indices 坐标参数类型（coordinate argument types）。The coordinate argument types.
         * @param indices 各轴坐标（axis coordinates）。The axis coordinates.
         * @return 元素引用（element reference）。A reference to the element.
         */
        template <typename... Indices>
            requires(sizeof...(Indices) > 0U) &&
                    (std::integral<std::remove_cvref_t<Indices>> && ...)
        [[nodiscard]] constexpr auto operator()(Indices... indices) noexcept -> reference
        {
            const std::array<index_type, sizeof...(Indices)> multi_index{
                static_cast<index_type>(indices)...};
            return (*this)(IndexView<index_type>{multi_index.data(), multi_index.size()});
        }

        /**
         * @brief 按独立坐标访问元素（const variadic index access）。Access an element by separate coordinates.
         *
         * @tparam Indices 坐标参数类型（coordinate argument types）。The coordinate argument types.
         * @param indices 各轴坐标（axis coordinates）。The axis coordinates.
         * @return 元素常量引用（const element reference）。A const reference to the element.
         */
        template <typename... Indices>
            requires(sizeof...(Indices) > 0U) &&
                    (std::integral<std::remove_cvref_t<Indices>> && ...)
        [[nodiscard]] constexpr auto operator()(Indices... indices) const noexcept -> const_reference
        {
            const std::array<index_type, sizeof...(Indices)> multi_index{
                static_cast<index_type>(indices)...};
            return (*this)(IndexView<index_type>{multi_index.data(), multi_index.size()});
        }

        /**
         * @brief 带检查访问元素（checked element access）。Access an element with bounds checks.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 元素引用（element reference）。A reference to the element.
         */
        [[nodiscard]] auto at(IndexView<index_type> index) -> reference
        {
            check_index(index);
            return (*this)(index);
        }

        /**
         * @brief 带检查访问元素（const checked element access）。Access an element with bounds checks.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         * @return 元素常量引用（const element reference）。A const reference to the element.
         */
        [[nodiscard]] auto at(IndexView<index_type> index) const -> const_reference
        {
            check_index(index);
            return (*this)(index);
        }

        /**
         * @brief 按独立坐标带检查访问元素（checked variadic index access）。Access by separate coordinates with checks.
         *
         * @tparam Indices 坐标参数类型（coordinate argument types）。The coordinate argument types.
         * @param indices 各轴坐标（axis coordinates）。The axis coordinates.
         * @return 元素引用（element reference）。A reference to the element.
         */
        template <typename... Indices>
            requires(sizeof...(Indices) > 0U) &&
                    (std::integral<std::remove_cvref_t<Indices>> && ...)
        [[nodiscard]] auto at(Indices... indices) -> reference
        {
            const std::array<index_type, sizeof...(Indices)> multi_index{
                static_cast<index_type>(indices)...};
            return at(IndexView<index_type>{multi_index.data(), multi_index.size()});
        }

        /**
         * @brief 按独立坐标带检查访问元素（const checked variadic index access）。Access by separate coordinates with checks.
         *
         * @tparam Indices 坐标参数类型（coordinate argument types）。The coordinate argument types.
         * @param indices 各轴坐标（axis coordinates）。The axis coordinates.
         * @return 元素常量引用（const element reference）。A const reference to the element.
         */
        template <typename... Indices>
            requires(sizeof...(Indices) > 0U) &&
                    (std::integral<std::remove_cvref_t<Indices>> && ...)
        [[nodiscard]] auto at(Indices... indices) const -> const_reference
        {
            const std::array<index_type, sizeof...(Indices)> multi_index{
                static_cast<index_type>(indices)...};
            return at(IndexView<index_type>{multi_index.data(), multi_index.size()});
        }

        /**
         * @brief 用同一值填充张量（fill tensor）。Fill the tensor with one value.
         *
         * @param value 填充值（fill value）。The fill value.
         */
        void fill(const value_type &value)
        {
            if (empty())
            {
                return;
            }
            std::fill_n(data(), size_, value);
        }

    private:
        /**
         * @brief 私有拷贝构造辅助（private copy helper construction）。Private construction helper for copying.
         *
         * @param layout 布局对象（layout object）。The layout object.
         * @param element_count 元素数量（element count）。The element count.
         * @param source 源数据指针（source data pointer）。The source data pointer.
         * @param allocator 分配器对象（allocator object）。The allocator object.
         */
        Tensor(layout_type layout,
               size_type element_count,
               const value_type *source,
               const allocator_type &allocator)
            : layout_{std::move(layout)},
              allocator_{allocator}
        {
            const size_type expected_count = checked_storage_size(layout_, allocator_);
            if (element_count != expected_count)
            {
                throw std::length_error{"cinder::Tensor source count does not match layout storage size"};
            }
            initialize_copy(element_count, source);
        }

        /**
         * @brief 将分配器指针转为原始指针（allocator pointer to raw pointer）。Convert allocator pointer to raw pointer.
         *
         * @param current 分配器指针（allocator pointer）。The allocator pointer.
         * @return 原始指针（raw pointer）。The raw pointer.
         */
        [[nodiscard]] CINDER_HOST_DEVICE static constexpr auto raw_pointer(pointer current) noexcept
            -> value_type *
        {
            if constexpr (std::is_pointer_v<pointer>)
            {
                return current;
            }
            else
            {
                return std::to_address(current);
            }
        }

        /**
         * @brief 返回原始数据指针且不检查空值（unchecked raw data）。Return raw data without additional checks.
         *
         * @return 原始数据指针（raw data pointer）。The raw data pointer.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto raw_data_unchecked() noexcept -> value_type *
        {
            if (size_ == size_type{0U})
            {
                return nullptr;
            }
            return raw_pointer(data_);
        }

        /**
         * @brief 返回原始数据常量指针且不检查空值（unchecked const raw data）。Return const raw data without additional checks.
         *
         * @return 原始数据常量指针（const raw data pointer）。The const raw data pointer.
         */
        [[nodiscard]] CINDER_HOST_DEVICE constexpr auto raw_data_unchecked() const noexcept -> const value_type *
        {
            if (size_ == size_type{0U})
            {
                return nullptr;
            }
            return raw_pointer(data_);
        }

        /**
         * @brief 检查布局存储大小是否可分配（checked layout storage size）。Check allocatable layout storage size.
         *
         * @param layout 布局对象（layout object）。The layout object.
         * @param allocator 分配器对象（allocator object）。The allocator object.
         * @return 可用于分配器的元素数量（allocator-compatible element count）。The allocator-compatible element count.
         */
        [[nodiscard]] static auto checked_storage_size(const layout_type &layout,
                                                       const allocator_type &allocator) -> size_type
        {
            const auto layout_count = detail::layout_storage_size(layout);
            const auto element_count = static_cast<size_type>(layout_count);
            if (static_cast<offset_type>(element_count) != layout_count)
            {
                throw std::length_error{"cinder::Tensor layout storage size cannot fit allocator size_type"};
            }
            if (element_count > allocator_traits::max_size(allocator))
            {
                throw std::length_error{"cinder::Tensor layout storage size exceeds allocator max_size"};
            }
            return element_count;
        }

        /**
         * @brief 分配未初始化存储（allocate uninitialized storage）。Allocate uninitialized storage.
         *
         * @param element_count 元素数量（element count）。The element count.
         * @return 分配器指针（allocator pointer）。The allocator pointer.
         */
        [[nodiscard]] auto allocate_storage(size_type element_count) -> pointer
        {
            if (element_count == size_type{0U})
            {
                return pointer{};
            }
            return allocator_traits::allocate(allocator_, element_count);
        }

        /**
         * @brief 销毁已构造元素（destroy constructed elements）。Destroy constructed elements.
         *
         * @param storage 存储指针（storage pointer）。The storage pointer.
         * @param constructed_count 已构造数量（constructed count）。The number of constructed elements.
         */
        void destroy_constructed(pointer storage, size_type constructed_count) noexcept
        {
            if (constructed_count == size_type{0U})
            {
                return;
            }

            value_type *const raw_storage = raw_pointer(storage);
            for (size_type index{0U}; index < constructed_count; ++index)
            {
                allocator_traits::destroy(allocator_, raw_storage + index);
            }
        }

        /**
         * @brief 释放存储（deallocate storage）。Deallocate storage.
         *
         * @param storage 存储指针（storage pointer）。The storage pointer.
         * @param element_count 元素数量（element count）。The element count.
         */
        void deallocate_storage(pointer storage, size_type element_count) noexcept
        {
            if (element_count != size_type{0U})
            {
                allocator_traits::deallocate(allocator_, storage, element_count);
            }
        }

        /**
         * @brief 销毁并释放当前存储（release current storage）。Destroy and deallocate current storage.
         */
        void release() noexcept
        {
            destroy_constructed(data_, size_);
            deallocate_storage(data_, size_);
            data_ = pointer{};
            size_ = size_type{0U};
        }

        /**
         * @brief 值初始化线性存储（value-initialize storage）。Value-initialize linear storage.
         *
         * @param element_count 元素数量（element count）。The element count.
         */
        void initialize_default(size_type element_count)
        {
            pointer storage = allocate_storage(element_count);
            size_type constructed_count{0U};
            try
            {
                value_type *const raw_storage = element_count == size_type{0U} ? nullptr : raw_pointer(storage);
                for (; constructed_count < element_count; ++constructed_count)
                {
                    allocator_traits::construct(allocator_, raw_storage + constructed_count);
                }
            }
            catch (...)
            {
                destroy_constructed(storage, constructed_count);
                deallocate_storage(storage, element_count);
                throw;
            }

            data_ = storage;
            size_ = element_count;
        }

        /**
         * @brief 用同一值初始化线性存储（fill-initialize storage）。Initialize linear storage with one value.
         *
         * @param element_count 元素数量（element count）。The element count.
         * @param value 填充值（fill value）。The fill value.
         */
        void initialize_fill(size_type element_count, const value_type &value)
        {
            pointer storage = allocate_storage(element_count);
            size_type constructed_count{0U};
            try
            {
                value_type *const raw_storage = element_count == size_type{0U} ? nullptr : raw_pointer(storage);
                for (; constructed_count < element_count; ++constructed_count)
                {
                    allocator_traits::construct(allocator_, raw_storage + constructed_count, value);
                }
            }
            catch (...)
            {
                destroy_constructed(storage, constructed_count);
                deallocate_storage(storage, element_count);
                throw;
            }

            data_ = storage;
            size_ = element_count;
        }

        /**
         * @brief 从原始数组初始化线性存储（copy-initialize storage）。Initialize linear storage from raw source.
         *
         * @param element_count 元素数量（element count）。The element count.
         * @param source 源数据指针（source data pointer）。The source data pointer.
         */
        void initialize_copy(size_type element_count, const value_type *source)
        {
            pointer storage = allocate_storage(element_count);
            size_type constructed_count{0U};
            try
            {
                value_type *const raw_storage = element_count == size_type{0U} ? nullptr : raw_pointer(storage);
                for (; constructed_count < element_count; ++constructed_count)
                {
                    allocator_traits::construct(
                        allocator_,
                        raw_storage + constructed_count,
                        source[constructed_count]);
                }
            }
            catch (...)
            {
                destroy_constructed(storage, constructed_count);
                deallocate_storage(storage, element_count);
                throw;
            }

            data_ = storage;
            size_ = element_count;
        }

        /**
         * @brief 从范围初始化线性存储（range-initialize storage）。Initialize linear storage from a range.
         *
         * @tparam Range 输入范围类型（input range type）。The input range type.
         * @param element_count 元素数量（element count）。The element count.
         * @param values 初始值范围（initial value range）。The initial value range.
         */
        template <std::ranges::input_range Range>
        void initialize_range(size_type element_count, Range &&values)
        {
            pointer storage = allocate_storage(element_count);
            size_type constructed_count{0U};
            auto iterator = std::ranges::begin(values);
            auto sentinel = std::ranges::end(values);

            try
            {
                value_type *const raw_storage = element_count == size_type{0U} ? nullptr : raw_pointer(storage);
                for (; constructed_count < element_count && iterator != sentinel;
                     ++constructed_count, ++iterator)
                {
                    allocator_traits::construct(
                        allocator_,
                        raw_storage + constructed_count,
                        static_cast<value_type>(*iterator));
                }

                if (constructed_count != element_count || iterator != sentinel)
                {
                    throw std::length_error{"cinder::Tensor range count does not match layout storage size"};
                }
            }
            catch (...)
            {
                destroy_constructed(storage, constructed_count);
                deallocate_storage(storage, element_count);
                throw;
            }

            data_ = storage;
            size_ = element_count;
        }

        /**
         * @brief 取得并接管另一个张量的存储（adopt storage）。Adopt another tensor's storage.
         *
         * @param other 源张量（source tensor）。The source tensor.
         */
        void adopt_storage_from(Tensor &other) noexcept(std::is_nothrow_move_assignable_v<layout_type>)
        {
            layout_ = std::move(other.layout_);
            data_ = std::exchange(other.data_, pointer{});
            size_ = std::exchange(other.size_, size_type{0U});
        }

        /**
         * @brief 将源张量的值移动赋给目标张量（move values to target）。Move-assign source values to target.
         *
         * @param target 目标张量（target tensor）。The target tensor.
         * @param source 源张量（source tensor）。The source tensor.
         */
        static void move_assign_values_to(Tensor &target, Tensor &source)
        {
            for (size_type index{0U}; index < target.size_; ++index)
            {
                target.raw_data_unchecked()[index] = std::move(source.raw_data_unchecked()[index]);
            }
        }

        /**
         * @brief 检查多重索引并确认线性偏移可访问（checked index validation）。Validate multi-index and linear offset.
         *
         * @param index 多重索引视图（multi-index view）。The multi-index view.
         */
        void check_index(IndexView<index_type> index) const
        {
            if (!detail::index_in_shape(layout_, index))
            {
                throw std::out_of_range{"cinder::Tensor index is outside shape"};
            }

            const offset_type linear_offset = layout_.offset(index);
            const auto element_offset = static_cast<size_type>(linear_offset);
            if (static_cast<offset_type>(element_offset) != linear_offset || element_offset >= size_)
            {
                throw std::out_of_range{"cinder::Tensor layout offset is outside storage"};
            }
        }

        /**
         * @brief 布局策略对象（layout policy object）。The layout policy object.
         */
        [[no_unique_address]] layout_type layout_{};

        /**
         * @brief 分配器策略对象（allocator policy object）。The allocator policy object.
         */
        [[no_unique_address]] allocator_type allocator_{};

        /**
         * @brief 分配器返回的线性存储指针（allocator storage pointer）。The linear storage pointer returned by allocator.
         */
        pointer data_{};

        /**
         * @brief 已构造线性元素数量（constructed linear element count）。The constructed linear element count.
         */
        size_type size_{0U};
    };

} // namespace cinder
