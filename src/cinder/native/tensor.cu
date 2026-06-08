#include "cinder/tensor.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace
{

  /**
   * @brief Tensor CUDA kernel 的线程块大小。Thread block size for Tensor CUDA kernels.
   */
  constexpr unsigned int k_tensor_threads_per_block = 256U;

  /**
   * @brief 把 size_t 值相加并检查溢出。Add size_t values with overflow checking.
   *
   * @param lhs 左操作数。Left operand.
   * @param rhs 右操作数。Right operand.
   * @param context 错误上下文。Error context.
   * @return 相加结果。Sum result.
   */
  [[nodiscard]] auto checked_add(std::size_t lhs, std::size_t rhs, std::string_view context) -> std::size_t
  {
    /**
     * @brief size_t 最大值。Maximum size_t value.
     */
    constexpr auto limit = std::numeric_limits<std::size_t>::max();

    if (lhs > (limit - rhs))
    {
      throw std::overflow_error(std::string(context) + " would overflow");
    }

    return lhs + rhs;
  }

  /**
   * @brief 把 size_t 值相乘并检查溢出。Multiply size_t values with overflow checking.
   *
   * @param lhs 左操作数。Left operand.
   * @param rhs 右操作数。Right operand.
   * @param context 错误上下文。Error context.
   * @return 相乘结果。Product result.
   */
  [[nodiscard]] auto checked_multiply(std::size_t lhs, std::size_t rhs, std::string_view context) -> std::size_t
  {
    /**
     * @brief size_t 最大值。Maximum size_t value.
     */
    constexpr auto limit = std::numeric_limits<std::size_t>::max();

    if ((rhs != 0U) && (lhs > (limit / rhs)))
    {
      throw std::overflow_error(std::string(context) + " would overflow");
    }

    return lhs * rhs;
  }

  /**
   * @brief 向上对齐 size_t 值并检查溢出。Align a size_t value upward with overflow checking.
   *
   * @param value 待对齐值。Value to align.
   * @param alignment 对齐字节数。Alignment in bytes.
   * @return 对齐后的值。Aligned value.
   */
  [[nodiscard]] auto align_up(std::size_t value, std::size_t alignment) -> std::size_t
  {
    /**
     * @brief 当前余数。Current remainder.
     */
    const auto remainder = value % alignment;

    if (remainder == 0U)
    {
      return value;
    }

    /**
     * @brief 对齐所需 padding。Padding required for alignment.
     */
    const auto padding = alignment - remainder;

    return checked_add(value, padding, "aligned tensor storage offset");
  }

  /**
   * @brief 计算 dense shape 的元素总数。Compute the element count for a dense shape.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   * @return 元素总数。Element count.
   */
  [[nodiscard]] auto checked_element_count(const std::vector<std::size_t> &extents) -> std::size_t
  {
    /**
     * @brief 累积元素总数。Accumulated element count.
     */
    std::size_t count = 1U;

    for (const auto extent : extents)
    {
      count = checked_multiply(count, extent, "tensor element count");
    }

    return count;
  }

  /**
   * @brief 计算 data 区域字节偏移。Compute the data region byte offset.
   *
   * @param rank 张量秩。Tensor rank.
   * @return data 区域字节偏移。Data region byte offset.
   */
  [[nodiscard]] auto data_offset_for_rank(std::size_t rank) -> std::size_t
  {
    /**
     * @brief extent 元数据字节数。Extent metadata size in bytes.
     */
    const auto extent_bytes = checked_multiply(rank, sizeof(std::size_t), "tensor extent metadata size");

    /**
     * @brief data 前面的元信息字节数。Metadata byte size before data.
     */
    const auto metadata_bytes = checked_add(sizeof(cinder::TensorStorageHeader), extent_bytes, "tensor metadata size");

    return align_up(metadata_bytes, alignof(cinder::Tensor::value_type));
  }

  /**
   * @brief 计算 packed storage 总字节数。Compute the total packed storage size in bytes.
   *
   * @param rank 张量秩。Tensor rank.
   * @param element_count 元素总数。Element count.
   * @return packed storage 总字节数。Total packed storage size in bytes.
   */
  [[nodiscard]] auto storage_bytes_for(std::size_t rank, std::size_t element_count) -> std::size_t
  {
    /**
     * @brief data 区域字节偏移。Data region byte offset.
     */
    const auto data_offset = data_offset_for_rank(rank);

    /**
     * @brief data 区域字节数。Data region size in bytes.
     */
    const auto data_bytes = checked_multiply(element_count, sizeof(cinder::Tensor::value_type), "tensor data size");

    return checked_add(data_offset, data_bytes, "tensor storage size");
  }

  /**
   * @brief 检查 CUDA runtime 调用结果。Check a CUDA runtime call result.
   *
   * @param status CUDA runtime 状态码。CUDA runtime status code.
   * @param action 正在执行的动作描述。Description of the action being performed.
   */
  auto check_cuda(cudaError_t status, std::string_view action) -> void
  {
    if (status == cudaSuccess)
    {
      return;
    }

    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
  }

  /**
   * @brief 构造 packed host staging buffer。Build a packed host staging buffer.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   * @param element_count 元素总数。Element count.
   * @param data_offset data 区域字节偏移。Data region byte offset.
   * @param storage_bytes packed storage 总字节数。Total packed storage size in bytes.
   * @param values 可选 host 数据；为空时 data 区域保持零初始化。Optional host data; when null, the data region stays zero-initialized.
   * @return packed host staging bytes。Packed host staging bytes.
   */
  [[nodiscard]] auto make_host_storage(const std::vector<std::size_t> &extents,
                                       std::size_t element_count,
                                       std::size_t data_offset,
                                       std::size_t storage_bytes,
                                       const std::vector<cinder::Tensor::value_type> *values)
      -> std::vector<unsigned char>
  {
    /**
     * @brief host staging buffer。Host staging buffer.
     */
    std::vector<unsigned char> storage(storage_bytes, 0U);

    /**
     * @brief packed storage header。Packed storage header.
     */
    const cinder::TensorStorageHeader header{
        extents.size(),
        element_count,
    };

    std::memcpy(storage.data(), &header, sizeof(header));

    if (!extents.empty())
    {
      /**
       * @brief extent 元数据字节数。Extent metadata byte size.
       */
      const auto extent_bytes = checked_multiply(extents.size(), sizeof(std::size_t), "tensor extent staging size");

      std::memcpy(storage.data() + sizeof(cinder::TensorStorageHeader), extents.data(), extent_bytes);
    }

    if ((values != nullptr) && !values->empty())
    {
      /**
       * @brief data 区域字节数。Data region byte size.
       */
      const auto data_bytes = checked_multiply(values->size(), sizeof(cinder::Tensor::value_type), "tensor data staging size");

      std::memcpy(storage.data() + data_offset, values->data(), data_bytes);
    }

    return storage;
  }

  /**
   * @brief 在 device 上应用二元运算。Apply a binary operation on device.
   *
   * @param lhs 左侧值。Left-hand side value.
   * @param rhs 右侧值。Right-hand side value.
   * @param operation 运算码。Operation code.
   * @return 运算结果。Operation result.
   */
  [[nodiscard]] __device__ auto apply_binary_operation(cinder::Tensor::value_type lhs,
                                                       cinder::Tensor::value_type rhs,
                                                       cinder::TensorBinaryOperation operation) noexcept
      -> cinder::Tensor::value_type
  {
    switch (operation)
    {
    case cinder::TensorBinaryOperation::add:
      return lhs + rhs;
    case cinder::TensorBinaryOperation::subtract:
      return lhs - rhs;
    case cinder::TensorBinaryOperation::multiply:
      return lhs * rhs;
    case cinder::TensorBinaryOperation::divide:
      return lhs / rhs;
    }

    return lhs;
  }

  /**
   * @brief 逐元素 Tensor 二元运算 kernel。Elementwise Tensor binary operation kernel.
   *
   * @param lhs 左侧只读 packed Tensor buffer。Left-hand side read-only packed Tensor buffer.
   * @param rhs 右侧只读 packed Tensor buffer。Right-hand side read-only packed Tensor buffer.
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param operation 运算码。Operation code.
   *
   * @note 第 0 个线程写 output header/extents；所有数据线程通过输入元信息构造 TensorView 并写 output data。
   *       Thread 0 writes the output header/extents; all data threads construct TensorView from input metadata and write output data.
   */
  __global__ void tensor_binary_kernel(cinder::ConstTensorDeviceBuffer lhs,
                                       cinder::ConstTensorDeviceBuffer rhs,
                                       cinder::TensorDeviceBuffer output,
                                       cinder::TensorBinaryOperation operation)
  {
    /**
     * @brief 左侧 packed storage header。Left-hand side packed storage header.
     */
    const auto *lhs_header = lhs.header();

    /**
     * @brief 当前线程线性编号。Current thread linear index.
     */
    const auto thread_index = (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) +
                              static_cast<std::size_t>(threadIdx.x);

    if (thread_index == 0U)
    {
      /**
       * @brief 输出 packed storage header。Output packed storage header.
       */
      auto *output_header = output.header();

      output_header->rank = lhs_header->rank;
      output_header->element_count = lhs_header->element_count;

      for (std::size_t axis = 0U; axis < lhs_header->rank; ++axis)
      {
        output.extents()[axis] = lhs.extents()[axis];
      }
    }

    if (thread_index >= lhs_header->element_count)
    {
      return;
    }

    /**
     * @brief 左侧只读 TensorView。Left-hand side read-only TensorView.
     */
    const auto lhs_view = lhs.view();

    /**
     * @brief 右侧只读 TensorView。Right-hand side read-only TensorView.
     */
    const auto rhs_view = rhs.view();

    /**
     * @brief 输出 TensorView，复用输入 shape 避免读取尚未同步的输出 header。Output TensorView reusing input shape to avoid reading the not-yet-synchronized output header.
     */
    cinder::TensorDeviceBuffer::view_type output_view(output.data(), lhs.shape());

    output_view.linear(thread_index) =
        apply_binary_operation(lhs_view.linear(thread_index), rhs_view.linear(thread_index), operation);
  }

  /**
   * @brief 启动 Tensor 二元运算 kernel。Launch the Tensor binary operation kernel.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param output 输出 Tensor。Output Tensor.
   * @param operation 运算码。Operation code.
   */
  auto launch_tensor_binary_kernel(const cinder::Tensor &lhs,
                                   const cinder::Tensor &rhs,
                                   cinder::Tensor &output,
                                   cinder::TensorBinaryOperation operation) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(lhs.size(), 1U);

    /**
     * @brief kernel block 数量。Kernel block count.
     */
    const auto block_count =
        (work_items + static_cast<std::size_t>(k_tensor_threads_per_block) - 1U) /
        static_cast<std::size_t>(k_tensor_threads_per_block);

    if (block_count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    {
      throw std::length_error("Tensor is too large for a 1D CUDA launch");
    }

    /**
     * @brief CUDA grid 维度。CUDA grid dimensions.
     */
    const dim3 grid(static_cast<unsigned int>(block_count));

    /**
     * @brief CUDA block 维度。CUDA block dimensions.
     */
    const dim3 block(k_tensor_threads_per_block);

    tensor_binary_kernel<<<grid, block>>>(lhs.device_buffer(), rhs.device_buffer(), output.device_buffer(), operation);
    check_cuda(cudaGetLastError(), "launch Tensor binary operation kernel");
  }

} // namespace

namespace cinder
{

  /**
   * @brief 构造零初始化 Tensor。Construct a zero-initialized Tensor.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   */
  Tensor::Tensor(std::vector<size_type> extents)
      : extents_(std::move(extents))
  {
    set_layout();
    allocate_device_storage();
    copy_host_storage(nullptr);
  }

  /**
   * @brief 从 host 数据构造 Tensor。Construct a Tensor from host data.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   * @param values host 侧 dense row-major 数据。Host-side dense row-major data.
   */
  Tensor::Tensor(std::vector<size_type> extents, const std::vector<value_type> &values)
      : extents_(std::move(extents))
  {
    set_layout();

    if (values.size() != element_count_)
    {
      throw std::invalid_argument("Tensor values size does not match shape element count");
    }

    allocate_device_storage();
    copy_host_storage(&values);
  }

  /**
   * @brief 释放 owned device memory。Release owned device memory.
   */
  Tensor::~Tensor()
  {
    release();
  }

  /**
   * @brief 深拷贝构造 Tensor。Deep-copy construct a Tensor.
   *
   * @param other 源 Tensor。Source Tensor.
   */
  Tensor::Tensor(const Tensor &other)
      : extents_(other.extents_),
        element_count_(other.element_count_),
        storage_bytes_(other.storage_bytes_),
        data_offset_(other.data_offset_)
  {
    if (other.device_storage_ == nullptr)
    {
      return;
    }

    allocate_device_storage();
    check_cuda(cudaMemcpy(device_storage_, other.device_storage_, storage_bytes_, cudaMemcpyDeviceToDevice),
               "copy Tensor device storage");
  }

  /**
   * @brief 深拷贝赋值 Tensor。Deep-copy assign a Tensor.
   *
   * @param other 源 Tensor。Source Tensor.
   * @return 当前 Tensor 引用。Reference to this Tensor.
   */
  auto Tensor::operator=(const Tensor &other) -> Tensor &
  {
    if (this == &other)
    {
      return *this;
    }

    /**
     * @brief 深拷贝临时对象。Deep-copy temporary object.
     */
    Tensor temporary(other);

    swap(temporary);

    return *this;
  }

  /**
   * @brief 移动构造 Tensor。Move-construct a Tensor.
   *
   * @param other 源 Tensor。Source Tensor.
   */
  Tensor::Tensor(Tensor &&other) noexcept
  {
    swap(other);
  }

  /**
   * @brief 移动赋值 Tensor。Move-assign a Tensor.
   *
   * @param other 源 Tensor。Source Tensor.
   * @return 当前 Tensor 引用。Reference to this Tensor.
   */
  auto Tensor::operator=(Tensor &&other) noexcept -> Tensor &
  {
    if (this == &other)
    {
      return *this;
    }

    release();
    swap(other);

    return *this;
  }

  /**
   * @brief 返回 host 侧 shape 元信息。Return host-side shape metadata.
   *
   * @return 每个轴的 extent。Per-axis extents.
   */
  auto Tensor::shape() const noexcept -> const std::vector<size_type> &
  {
    return extents_;
  }

  /**
   * @brief 返回张量秩。Return the tensor rank.
   *
   * @return rank 值。Rank value.
   */
  auto Tensor::rank() const noexcept -> size_type
  {
    return extents_.size();
  }

  /**
   * @brief 返回 dense 元素总数。Return the dense element count.
   *
   * @return 元素总数。Element count.
   */
  auto Tensor::size() const noexcept -> size_type
  {
    return element_count_;
  }

  /**
   * @brief 判断 Tensor 是否未分配 device storage。Check whether the Tensor has no device storage.
   *
   * @return 未分配时为 true。True when no device storage is allocated.
   */
  auto Tensor::empty() const noexcept -> bool
  {
    return device_storage_ == nullptr;
  }

  /**
   * @brief 返回可变 packed device buffer 描述符。Return a mutable packed device buffer descriptor.
   *
   * @return 可变 device buffer 描述符。Mutable device buffer descriptor.
   */
  auto Tensor::device_buffer() noexcept -> TensorDeviceBuffer
  {
    return TensorDeviceBuffer{
        device_storage_,
        data_offset_,
    };
  }

  /**
   * @brief 返回只读 packed device buffer 描述符。Return a read-only packed device buffer descriptor.
   *
   * @return 只读 device buffer 描述符。Read-only device buffer descriptor.
   */
  auto Tensor::device_buffer() const noexcept -> ConstTensorDeviceBuffer
  {
    return ConstTensorDeviceBuffer{
        device_storage_,
        data_offset_,
    };
  }

  /**
   * @brief 返回可变 device TensorView。Return a mutable device TensorView.
   *
   * @return 指向 device 数据和 device extents 的 TensorView。TensorView pointing at device data and device extents.
   */
  auto Tensor::view() noexcept -> view_type
  {
    return view_type(device_data(), Shape<size_type>(rank(), device_extents()));
  }

  /**
   * @brief 返回只读 device TensorView。Return a read-only device TensorView.
   *
   * @return 指向 device 数据和 device extents 的只读 TensorView。Read-only TensorView pointing at device data and device extents.
   */
  auto Tensor::view() const noexcept -> const_view_type
  {
    return const_view_type(device_data(), Shape<size_type>(rank(), device_extents()));
  }

  /**
   * @brief 把 data 区域拷回 host vector。Copy the data region back to a host vector.
   *
   * @return host 侧 dense row-major 数据。Host-side dense row-major data.
   */
  auto Tensor::to_vector() const -> std::vector<value_type>
  {
    /**
     * @brief host 输出数据。Host output data.
     */
    std::vector<value_type> values(element_count_);

    if (values.empty())
    {
      return values;
    }

    check_cuda(cudaMemcpy(values.data(), device_data(), values.size() * sizeof(value_type), cudaMemcpyDeviceToHost),
               "copy Tensor data to host");

    return values;
  }

  /**
   * @brief 执行逐元素二元运算。Run an elementwise binary operation.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param operation 二元运算码。Binary operation code.
   * @return 运算结果 Tensor。Result Tensor.
   */
  auto Tensor::binary(const Tensor &lhs, const Tensor &rhs, TensorBinaryOperation operation) -> Tensor
  {
    if (lhs.empty() || rhs.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }

    if (lhs.shape() != rhs.shape())
    {
      throw std::invalid_argument("Tensor shapes must match for elementwise operations");
    }

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(lhs.extents_, UninitializedTag{});

    launch_tensor_binary_kernel(lhs, rhs, output, operation);

    return output;
  }

  /**
   * @brief 构造未初始化 packed device storage。Construct uninitialized packed device storage.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   * @param tag 未初始化标签。Uninitialized tag.
   */
  Tensor::Tensor(std::vector<size_type> extents, UninitializedTag tag)
      : extents_(std::move(extents))
  {
    static_cast<void>(tag);
    set_layout();
    allocate_device_storage();
  }

  /**
   * @brief 返回 device extent 元数据指针。Return the device extent metadata pointer.
   *
   * @return device extent 指针。Device extent pointer.
   */
  auto Tensor::device_extents() const noexcept -> const size_type *
  {
    if (device_storage_ == nullptr)
    {
      return nullptr;
    }

    return reinterpret_cast<const size_type *>(device_storage_ + sizeof(TensorStorageHeader));
  }

  /**
   * @brief 返回可变 device data 指针。Return the mutable device data pointer.
   *
   * @return 可变 device data 指针。Mutable device data pointer.
   */
  auto Tensor::device_data() noexcept -> value_type *
  {
    if (device_storage_ == nullptr)
    {
      return nullptr;
    }

    return reinterpret_cast<value_type *>(device_storage_ + data_offset_);
  }

  /**
   * @brief 返回只读 device data 指针。Return the read-only device data pointer.
   *
   * @return 只读 device data 指针。Read-only device data pointer.
   */
  auto Tensor::device_data() const noexcept -> const value_type *
  {
    if (device_storage_ == nullptr)
    {
      return nullptr;
    }

    return reinterpret_cast<const value_type *>(device_storage_ + data_offset_);
  }

  /**
   * @brief 根据 host extents 计算 packed storage layout。Compute the packed storage layout from host extents.
   */
  auto Tensor::set_layout() -> void
  {
    element_count_ = checked_element_count(extents_);
    data_offset_ = data_offset_for_rank(extents_.size());
    storage_bytes_ = storage_bytes_for(extents_.size(), element_count_);
  }

  /**
   * @brief 分配 packed device storage。Allocate packed device storage.
   */
  auto Tensor::allocate_device_storage() -> void
  {
    if (storage_bytes_ == 0U)
    {
      return;
    }

    /**
     * @brief cudaMalloc 返回的泛型 allocation 指针。Generic allocation pointer returned by cudaMalloc.
     */
    void *allocation = nullptr;

    check_cuda(cudaMalloc(&allocation, storage_bytes_), "allocate Tensor device storage");
    device_storage_ = static_cast<unsigned char *>(allocation);
  }

  /**
   * @brief 打包 host 元信息与可选数据并一次性拷贝到 device。Pack host metadata and optional data, then copy to device once.
   *
   * @param values 可选 host 数据指针；为空时 data 区域填零。Optional host data pointer; when null, the data region is zero-filled.
   */
  auto Tensor::copy_host_storage(const std::vector<value_type> *values) -> void
  {
    if (device_storage_ == nullptr)
    {
      return;
    }

    /**
     * @brief packed host staging storage。Packed host staging storage.
     */
    const auto storage = make_host_storage(extents_, element_count_, data_offset_, storage_bytes_, values);

    check_cuda(cudaMemcpy(device_storage_, storage.data(), storage.size(), cudaMemcpyHostToDevice),
               "copy Tensor packed storage to device");
  }

  /**
   * @brief 释放 device storage。Release device storage.
   */
  auto Tensor::release() noexcept -> void
  {
    if (device_storage_ != nullptr)
    {
      /**
       * @brief cudaFree 状态；析构路径不抛异常。cudaFree status; destructor path does not throw.
       */
      const auto status = cudaFree(device_storage_);

      static_cast<void>(status);
    }

    device_storage_ = nullptr;
    element_count_ = 0U;
    storage_bytes_ = 0U;
    data_offset_ = 0U;
    extents_.clear();
  }

  /**
   * @brief 与另一个 Tensor 交换 owned 状态。Swap owned state with another Tensor.
   *
   * @param other 另一个 Tensor。The other Tensor.
   */
  auto Tensor::swap(Tensor &other) noexcept -> void
  {
    using std::swap;

    swap(extents_, other.extents_);
    swap(device_storage_, other.device_storage_);
    swap(element_count_, other.element_count_);
    swap(storage_bytes_, other.storage_bytes_);
    swap(data_offset_, other.data_offset_);
  }

  /**
   * @brief 逐元素加法。Elementwise addition.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 加法结果 Tensor。Addition result Tensor.
   */
  auto add(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    return Tensor::binary(lhs, rhs, TensorBinaryOperation::add);
  }

  /**
   * @brief 逐元素减法。Elementwise subtraction.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 减法结果 Tensor。Subtraction result Tensor.
   */
  auto subtract(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    return Tensor::binary(lhs, rhs, TensorBinaryOperation::subtract);
  }

  /**
   * @brief 逐元素乘法。Elementwise multiplication.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 乘法结果 Tensor。Multiplication result Tensor.
   */
  auto multiply(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    return Tensor::binary(lhs, rhs, TensorBinaryOperation::multiply);
  }

  /**
   * @brief 逐元素除法。Elementwise division.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 除法结果 Tensor。Division result Tensor.
   */
  auto divide(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    return Tensor::binary(lhs, rhs, TensorBinaryOperation::divide);
  }

  /**
   * @brief 逐元素加法运算符。Elementwise addition operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 加法结果 Tensor。Addition result Tensor.
   */
  auto operator+(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    return add(lhs, rhs);
  }

  /**
   * @brief 逐元素减法运算符。Elementwise subtraction operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 减法结果 Tensor。Subtraction result Tensor.
   */
  auto operator-(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    return subtract(lhs, rhs);
  }

  /**
   * @brief 逐元素乘法运算符。Elementwise multiplication operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 乘法结果 Tensor。Multiplication result Tensor.
   */
  auto operator*(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    return multiply(lhs, rhs);
  }

  /**
   * @brief 逐元素除法运算符。Elementwise division operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 除法结果 Tensor。Division result Tensor.
   */
  auto operator/(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    return divide(lhs, rhs);
  }

} // namespace cinder
