#pragma once

#include "cinder/mapping.cuh"
#include "cinder/shape.cuh"
#include "cinder/tensor_view.cuh"

#include <cstddef>
#include <vector>

namespace cinder
{

  /**
   * @brief packed CUDA 张量存储头。Packed CUDA tensor storage header.
   *
   * @note 设备内存布局为 header、extents、aligned data；kernel 可从同一 allocation 自解析元信息。
   *       The device memory layout is header, extents, and aligned data; kernels can parse metadata from the same allocation.
   */
  struct TensorStorageHeader final
  {
    /**
     * @brief 张量秩（rank）。The tensor rank.
     */
    std::size_t rank{};

    /**
     * @brief dense 张量元素总数。The dense tensor element count.
     */
    std::size_t element_count{};
  };

  /**
   * @brief 可变 packed Tensor device allocation 描述符。Mutable packed Tensor device allocation descriptor.
   *
   * @note 该描述符只保存 device allocation 起点与 data 偏移，适合按值传入 CUDA kernel。
   *       This descriptor only stores the device allocation base and data offset, so it is suitable for passing to CUDA kernels by value.
   */
  struct TensorDeviceBuffer final
  {
    /**
     * @brief 元素值类型。Element value type.
     */
    using value_type = float;

    /**
     * @brief 大小类型。Size type.
     */
    using size_type = std::size_t;

    /**
     * @brief 可变 TensorView 类型。Mutable TensorView type.
     */
    using view_type = TensorView<value_type, DenseRowMajorMapping>;

    /**
     * @brief 只读 TensorView 类型。Read-only TensorView type.
     */
    using const_view_type = TensorView<const value_type, DenseRowMajorMapping>;

    /**
     * @brief packed device allocation 起点。Base address of the packed device allocation.
     */
    unsigned char *storage{};

    /**
     * @brief data 区域相对 allocation 起点的字节偏移。Byte offset of the data region from the allocation base.
     */
    size_type data_offset{};

    /**
     * @brief 返回可变存储头指针。Return the mutable storage header pointer.
     *
     * @return 可变 TensorStorageHeader 指针。Mutable TensorStorageHeader pointer.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto header() const noexcept -> TensorStorageHeader *
    {
      return reinterpret_cast<TensorStorageHeader *>(storage);
    }

    /**
     * @brief 返回可变 extent 元数据指针。Return the mutable extent metadata pointer.
     *
     * @return 可变 extent 指针。Mutable extent pointer.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto extents() const noexcept -> size_type *
    {
      return reinterpret_cast<size_type *>(storage + sizeof(TensorStorageHeader));
    }

    /**
     * @brief 返回可变 data 指针。Return the mutable data pointer.
     *
     * @return 可变元素指针。Mutable element pointer.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto data() const noexcept -> value_type *
    {
      return reinterpret_cast<value_type *>(storage + data_offset);
    }

    /**
     * @brief 从 packed 元信息构建 Shape。Build a Shape from packed metadata.
     *
     * @return 指向 device extents 的 Shape。Shape pointing at device extents.
     *
     * @note 在 host 上调用该函数会解引用 device pointer；预期用途是在 kernel 内调用。
     *       Calling this on host dereferences a device pointer; the intended use is inside kernels.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto shape() const noexcept -> Shape<size_type>
    {
      return Shape<size_type>(header()->rank, extents());
    }

    /**
     * @brief 在 kernel 内从 packed allocation 构建可变 TensorView。Build a mutable TensorView from the packed allocation inside a kernel.
     *
     * @return 可变 TensorView。Mutable TensorView.
     *
     * @note 在 host 上调用该函数会读取 device 元信息；host 侧请使用 Tensor::view()。
     *       Calling this on host reads device metadata; use Tensor::view() on the host side.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto view() const noexcept -> view_type
    {
      return view_type(data(), shape());
    }

    /**
     * @brief 在 kernel 内从 packed allocation 构建只读 TensorView。Build a read-only TensorView from the packed allocation inside a kernel.
     *
     * @return 只读 TensorView。Read-only TensorView.
     *
     * @note 在 host 上调用该函数会读取 device 元信息；host 侧请使用 Tensor::view()。
     *       Calling this on host reads device metadata; use Tensor::view() on the host side.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto const_view() const noexcept -> const_view_type
    {
      return const_view_type(data(), shape());
    }
  };

  /**
   * @brief 只读 packed Tensor device allocation 描述符。Read-only packed Tensor device allocation descriptor.
   *
   * @note 该描述符用于把输入 Tensor 按值传入 CUDA kernel，并在 kernel 内自解析 TensorView。
   *       This descriptor passes input tensors into CUDA kernels by value and lets kernels parse TensorView metadata internally.
   */
  struct ConstTensorDeviceBuffer final
  {
    /**
     * @brief 元素值类型。Element value type.
     */
    using value_type = float;

    /**
     * @brief 大小类型。Size type.
     */
    using size_type = std::size_t;

    /**
     * @brief 只读 TensorView 类型。Read-only TensorView type.
     */
    using view_type = TensorView<const value_type, DenseRowMajorMapping>;

    /**
     * @brief packed device allocation 起点。Base address of the packed device allocation.
     */
    const unsigned char *storage{};

    /**
     * @brief data 区域相对 allocation 起点的字节偏移。Byte offset of the data region from the allocation base.
     */
    size_type data_offset{};

    /**
     * @brief 返回只读存储头指针。Return the read-only storage header pointer.
     *
     * @return 只读 TensorStorageHeader 指针。Read-only TensorStorageHeader pointer.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto header() const noexcept -> const TensorStorageHeader *
    {
      return reinterpret_cast<const TensorStorageHeader *>(storage);
    }

    /**
     * @brief 返回只读 extent 元数据指针。Return the read-only extent metadata pointer.
     *
     * @return 只读 extent 指针。Read-only extent pointer.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto extents() const noexcept -> const size_type *
    {
      return reinterpret_cast<const size_type *>(storage + sizeof(TensorStorageHeader));
    }

    /**
     * @brief 返回只读 data 指针。Return the read-only data pointer.
     *
     * @return 只读元素指针。Read-only element pointer.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto data() const noexcept -> const value_type *
    {
      return reinterpret_cast<const value_type *>(storage + data_offset);
    }

    /**
     * @brief 从 packed 元信息构建 Shape。Build a Shape from packed metadata.
     *
     * @return 指向 device extents 的 Shape。Shape pointing at device extents.
     *
     * @note 在 host 上调用该函数会解引用 device pointer；预期用途是在 kernel 内调用。
     *       Calling this on host dereferences a device pointer; the intended use is inside kernels.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto shape() const noexcept -> Shape<size_type>
    {
      return Shape<size_type>(header()->rank, extents());
    }

    /**
     * @brief 在 kernel 内从 packed allocation 构建只读 TensorView。Build a read-only TensorView from the packed allocation inside a kernel.
     *
     * @return 只读 TensorView。Read-only TensorView.
     *
     * @note 在 host 上调用该函数会读取 device 元信息；host 侧请使用 Tensor::view()。
     *       Calling this on host reads device metadata; use Tensor::view() on the host side.
     */
    [[nodiscard]] CINDER_HOST_DEVICE auto view() const noexcept -> view_type
    {
      return view_type(data(), shape());
    }
  };

  /**
   * @brief host-owned CUDA dense Tensor，管理 packed device memory 并套 TensorView。
   *        Host-owned CUDA dense Tensor that manages packed device memory and wraps TensorView.
   *
   * @note 当前 Tensor 使用 float32 元素和 dense row-major mapping；shape 元信息与数据位于同一个 device allocation。
   *       The current Tensor uses float32 elements and dense row-major mapping; shape metadata and data live in one device allocation.
   */
  class Tensor final
  {
  public:
    /**
     * @brief 元素值类型。Element value type.
     */
    using value_type = float;

    /**
     * @brief 大小与索引类型。Size and index type.
     */
    using size_type = std::size_t;

    /**
     * @brief dense row-major mapping 类型。Dense row-major mapping type.
     */
    using mapping_type = DenseRowMajorMapping;

    /**
     * @brief 可变 TensorView 类型。Mutable TensorView type.
     */
    using view_type = TensorView<value_type, mapping_type>;

    /**
     * @brief 只读 TensorView 类型。Read-only TensorView type.
     */
    using const_view_type = TensorView<const value_type, mapping_type>;

    /**
     * @brief 构造空 Tensor。Construct an empty Tensor.
     */
    Tensor() noexcept = default;

    /**
     * @brief 构造零初始化 Tensor。Construct a zero-initialized Tensor.
     *
     * @param extents 每个轴的 extent。Per-axis extents.
     */
    explicit Tensor(std::vector<size_type> extents);

    /**
     * @brief 从 host 数据构造 Tensor。Construct a Tensor from host data.
     *
     * @param extents 每个轴的 extent。Per-axis extents.
     * @param values host 侧 dense row-major 数据。Host-side dense row-major data.
     */
    Tensor(std::vector<size_type> extents, const std::vector<value_type> &values);

    /**
     * @brief 释放 owned device memory。Release owned device memory.
     */
    ~Tensor();

    /**
     * @brief 深拷贝构造 Tensor。Deep-copy construct a Tensor.
     *
     * @param other 源 Tensor。Source Tensor.
     */
    Tensor(const Tensor &other);

    /**
     * @brief 深拷贝赋值 Tensor。Deep-copy assign a Tensor.
     *
     * @param other 源 Tensor。Source Tensor.
     * @return 当前 Tensor 引用。Reference to this Tensor.
     */
    auto operator=(const Tensor &other) -> Tensor &;

    /**
     * @brief 移动构造 Tensor。Move-construct a Tensor.
     *
     * @param other 源 Tensor。Source Tensor.
     */
    Tensor(Tensor &&other) noexcept;

    /**
     * @brief 移动赋值 Tensor。Move-assign a Tensor.
     *
     * @param other 源 Tensor。Source Tensor.
     * @return 当前 Tensor 引用。Reference to this Tensor.
     */
    auto operator=(Tensor &&other) noexcept -> Tensor &;

    /**
     * @brief 返回 host 侧 shape 元信息。Return host-side shape metadata.
     *
     * @return 每个轴的 extent。Per-axis extents.
     */
    [[nodiscard]] auto shape() const noexcept -> const std::vector<size_type> &;

    /**
     * @brief 返回张量秩。Return the tensor rank.
     *
     * @return rank 值。Rank value.
     */
    [[nodiscard]] auto rank() const noexcept -> size_type;

    /**
     * @brief 返回 dense 元素总数。Return the dense element count.
     *
     * @return 元素总数。Element count.
     */
    [[nodiscard]] auto size() const noexcept -> size_type;

    /**
     * @brief 判断 Tensor 是否未分配 device storage。Check whether the Tensor has no device storage.
     *
     * @return 未分配时为 true。True when no device storage is allocated.
     */
    [[nodiscard]] auto empty() const noexcept -> bool;

    /**
     * @brief 返回可变 packed device buffer 描述符。Return a mutable packed device buffer descriptor.
     *
     * @return 可变 device buffer 描述符。Mutable device buffer descriptor.
     */
    [[nodiscard]] auto device_buffer() noexcept -> TensorDeviceBuffer;

    /**
     * @brief 返回只读 packed device buffer 描述符。Return a read-only packed device buffer descriptor.
     *
     * @return 只读 device buffer 描述符。Read-only device buffer descriptor.
     */
    [[nodiscard]] auto device_buffer() const noexcept -> ConstTensorDeviceBuffer;

    /**
     * @brief 返回可变 device TensorView。Return a mutable device TensorView.
     *
     * @return 指向 device 数据和 device extents 的 TensorView。TensorView pointing at device data and device extents.
     *
     * @note 该 view 可直接按值传入 kernel；host 侧不要解引用 view 中的 device 指针。
     *       This view can be passed into kernels by value; do not dereference its device pointers on host.
     */
    [[nodiscard]] auto view() noexcept -> view_type;

    /**
     * @brief 返回只读 device TensorView。Return a read-only device TensorView.
     *
     * @return 指向 device 数据和 device extents 的只读 TensorView。Read-only TensorView pointing at device data and device extents.
     *
     * @note 该 view 可直接按值传入 kernel；host 侧不要解引用 view 中的 device 指针。
     *       This view can be passed into kernels by value; do not dereference its device pointers on host.
     */
    [[nodiscard]] auto view() const noexcept -> const_view_type;

    /**
     * @brief 把 data 区域拷回 host vector。Copy the data region back to a host vector.
     *
     * @return host 侧 dense row-major 数据。Host-side dense row-major data.
     */
    [[nodiscard]] auto to_vector() const -> std::vector<value_type>;

    /**
     * @brief 计算当前 Tensor 与另一个 Tensor 的张量积。Compute the tensor product of this Tensor and another Tensor.
     *
     * @param rhs 右侧 Tensor。Right-hand side Tensor.
     * @return 张量积结果 Tensor。Tensor product result Tensor.
     *
     * @note 输出 shape 为左右输入 shape 的拼接；输出 data 由单个 CUDA kernel 写入。
     *       The output shape is the concatenation of the input shapes; output data is written by one CUDA kernel.
     */
    [[nodiscard]] auto tensor_product(const Tensor &rhs) const -> Tensor;

  private:
    /**
     * @brief 允许加法运算符访问私有二元运算入口。Allow the addition operator to access the private binary operation entry point.
     */
    friend auto operator+(const Tensor &lhs, const Tensor &rhs) -> Tensor;

    /**
     * @brief 允许减法运算符访问私有二元运算入口。Allow the subtraction operator to access the private binary operation entry point.
     */
    friend auto operator-(const Tensor &lhs, const Tensor &rhs) -> Tensor;

    /**
     * @brief 允许乘法运算符访问私有二元运算入口。Allow the multiplication operator to access the private binary operation entry point.
     */
    friend auto operator*(const Tensor &lhs, const Tensor &rhs) -> Tensor;

    /**
     * @brief 允许除法运算符访问私有二元运算入口。Allow the division operator to access the private binary operation entry point.
     */
    friend auto operator/(const Tensor &lhs, const Tensor &rhs) -> Tensor;

    /**
     * @brief 允许张量积函数构造未初始化输出 Tensor。Allow tensor_product to construct an uninitialized output Tensor.
     */
    friend auto tensor_product(const Tensor &lhs, const Tensor &rhs) -> Tensor;

    /**
     * @brief 不初始化 device storage 的构造标签。Construction tag for uninitialized device storage.
     */
    struct UninitializedTag final
    {
    };

    /**
     * @brief 构造未初始化 packed device storage。Construct uninitialized packed device storage.
     *
     * @param extents 每个轴的 extent。Per-axis extents.
     * @param tag 未初始化标签。Uninitialized tag.
     */
    Tensor(std::vector<size_type> extents, UninitializedTag tag);

    /**
     * @brief 执行逐元素二元运算。Run an elementwise binary operation.
     *
     * @param lhs 左侧 Tensor。Left-hand side Tensor.
     * @param rhs 右侧 Tensor。Right-hand side Tensor.
     * @param operation 二元运算码。Binary operation code.
     * @return 运算结果 Tensor。Result Tensor.
     */
    [[nodiscard]] static auto binary(const Tensor &lhs, const Tensor &rhs, unsigned int operation) -> Tensor;

    /**
     * @brief 返回 device extent 元数据指针。Return the device extent metadata pointer.
     *
     * @return device extent 指针。Device extent pointer.
     */
    [[nodiscard]] auto device_extents() const noexcept -> const size_type *;

    /**
     * @brief 返回可变 device data 指针。Return the mutable device data pointer.
     *
     * @return 可变 device data 指针。Mutable device data pointer.
     */
    [[nodiscard]] auto device_data() noexcept -> value_type *;

    /**
     * @brief 返回只读 device data 指针。Return the read-only device data pointer.
     *
     * @return 只读 device data 指针。Read-only device data pointer.
     */
    [[nodiscard]] auto device_data() const noexcept -> const value_type *;

    /**
     * @brief 根据 host extents 计算 packed storage layout。Compute the packed storage layout from host extents.
     */
    auto set_layout() -> void;

    /**
     * @brief 分配 packed device storage。Allocate packed device storage.
     */
    auto allocate_device_storage() -> void;

    /**
     * @brief 打包 host 元信息与可选数据并一次性拷贝到 device。Pack host metadata and optional data, then copy to device once.
     *
     * @param values 可选 host 数据指针；为空时 data 区域填零。Optional host data pointer; when null, the data region is zero-filled.
     */
    auto copy_host_storage(const std::vector<value_type> *values) -> void;

    /**
     * @brief 释放 device storage。Release device storage.
     */
    auto release() noexcept -> void;

    /**
     * @brief 与另一个 Tensor 交换 owned 状态。Swap owned state with another Tensor.
     *
     * @param other 另一个 Tensor。The other Tensor.
     */
    auto swap(Tensor &other) noexcept -> void;

    /**
     * @brief host 侧 extent 元信息。Host-side extent metadata.
     */
    std::vector<size_type> extents_{};

    /**
     * @brief packed device allocation 起点。Base address of the packed device allocation.
     */
    unsigned char *device_storage_{};

    /**
     * @brief dense 元素总数。Dense element count.
     */
    size_type element_count_{};

    /**
     * @brief packed device allocation 字节数。Packed device allocation size in bytes.
     */
    size_type storage_bytes_{};

    /**
     * @brief data 区域相对 allocation 起点的字节偏移。Byte offset of the data region from the allocation base.
     */
    size_type data_offset_{};
  };

  /**
   * @brief 逐元素加法运算符。Elementwise addition operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 加法结果 Tensor。Addition result Tensor.
   */
  [[nodiscard]] auto operator+(const Tensor &lhs, const Tensor &rhs) -> Tensor;

  /**
   * @brief 逐元素减法运算符。Elementwise subtraction operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 减法结果 Tensor。Subtraction result Tensor.
   */
  [[nodiscard]] auto operator-(const Tensor &lhs, const Tensor &rhs) -> Tensor;

  /**
   * @brief 逐元素乘法运算符。Elementwise multiplication operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 乘法结果 Tensor。Multiplication result Tensor.
   */
  [[nodiscard]] auto operator*(const Tensor &lhs, const Tensor &rhs) -> Tensor;

  /**
   * @brief 逐元素除法运算符。Elementwise division operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 除法结果 Tensor。Division result Tensor.
   */
  [[nodiscard]] auto operator/(const Tensor &lhs, const Tensor &rhs) -> Tensor;

  /**
   * @brief 计算两个 Tensor 的张量积。Compute the tensor product of two Tensors.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 张量积结果 Tensor。Tensor product result Tensor.
   *
   * @note 对 dense row-major storage，输出 shape 为 lhs.shape() + rhs.shape()，
   *       且 output.linear(i * rhs.size() + j) = lhs.linear(i) * rhs.linear(j)。
   *       For dense row-major storage, the output shape is lhs.shape() + rhs.shape(),
   *       and output.linear(i * rhs.size() + j) = lhs.linear(i) * rhs.linear(j).
   */
  [[nodiscard]] auto tensor_product(const Tensor &lhs, const Tensor &rhs) -> Tensor;

} // namespace cinder
