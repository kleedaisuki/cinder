#include "cinder/tensor.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

  /**
   * @brief Tensor 标量运算 kernel 的线程块大小。Thread block size for Tensor scalar-operation kernels.
   */
  constexpr unsigned int k_tensor_scalar_threads_per_block = 256U;

  /**
   * @brief Tensor reduction kernel 的线程块大小。Thread block size for Tensor reduction kernels.
   */
  constexpr unsigned int k_tensor_reduction_threads_per_block = 256U;

  /**
   * @brief reduction 第一阶段每个线程的目标元素数。Target element count per thread in the first reduction pass.
   */
  constexpr std::size_t k_tensor_reduction_items_per_thread = 4U;

  /**
   * @brief reduction 第一阶段最多启动的 block 数。Maximum block count for the first reduction pass.
   */
  constexpr std::size_t k_tensor_reduction_max_blocks = 4096U;

  /**
   * @brief Tensor 标量二元运算码。Tensor-scalar binary operation code.
   */
  enum class TensorScalarOperation : unsigned int
  {
    /**
     * @brief 加法。Addition.
     */
    plus,

    /**
     * @brief 减法。Subtraction.
     */
    minus,

    /**
     * @brief 乘法。Multiplication.
     */
    times,

    /**
     * @brief 除法。Division.
     */
    quotient
  };

  /**
   * @brief Tensor reduction 运算码。Tensor reduction operation code.
   */
  enum class TensorReductionOperation : unsigned int
  {
    /**
     * @brief 内积。Inner product.
     */
    inner,

    /**
     * @brief L2 范数。L2 norm.
     */
    l2_norm
  };

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
   * @brief 在 device 上应用 Tensor-标量二元运算。Apply a Tensor-scalar binary operation on device.
   *
   * @param tensor_value Tensor 元素值。Tensor element value.
   * @param scalar_value 标量值。Scalar value.
   * @param operation 运算码。Operation code.
   * @param scalar_on_left 标量是否位于左操作数。Whether the scalar is the left-hand operand.
   * @return 运算结果。Operation result.
   */
  [[nodiscard]] __device__ auto apply_scalar_operation(cinder::Tensor::value_type tensor_value,
                                                       cinder::Tensor::value_type scalar_value,
                                                       TensorScalarOperation operation,
                                                       bool scalar_on_left) noexcept
      -> cinder::Tensor::value_type
  {
    /**
     * @brief 左操作数。Left-hand side operand.
     */
    const auto lhs = scalar_on_left ? scalar_value : tensor_value;

    /**
     * @brief 右操作数。Right-hand side operand.
     */
    const auto rhs = scalar_on_left ? tensor_value : scalar_value;

    switch (operation)
    {
    case TensorScalarOperation::plus:
      return lhs + rhs;
    case TensorScalarOperation::minus:
      return lhs - rhs;
    case TensorScalarOperation::times:
      return lhs * rhs;
    case TensorScalarOperation::quotient:
      return lhs / rhs;
    }

    return tensor_value;
  }

  /**
   * @brief 逐元素 Tensor-标量二元运算 kernel。Elementwise Tensor-scalar binary operation kernel.
   *
   * @param input 输入只读 packed Tensor buffer。Input read-only packed Tensor buffer.
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param scalar_value 标量值。Scalar value.
   * @param operation 运算码。Operation code.
   * @param scalar_on_left 标量是否位于左操作数。Whether the scalar is the left-hand operand.
   *
   * @note 第 0 个线程写 output metadata；标量按值传入 kernel，不需要为标量构造 device Tensor。
   *       Thread 0 writes output metadata; the scalar is passed by value to the kernel, so no device Tensor is built for it.
   */
  __global__ void tensor_scalar_kernel(cinder::ConstTensorDeviceBuffer input,
                                       cinder::TensorDeviceBuffer output,
                                       cinder::Tensor::value_type scalar_value,
                                       TensorScalarOperation operation,
                                       bool scalar_on_left)
  {
    /**
     * @brief 输入 packed storage header。Input packed storage header.
     */
    const auto *input_header = input.header();

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

      output_header->rank = input_header->rank;
      output_header->element_count = input_header->element_count;

      for (std::size_t axis = 0U; axis < input_header->rank; ++axis)
      {
        output.extents()[axis] = input.extents()[axis];
      }
    }

    if (thread_index >= input_header->element_count)
    {
      return;
    }

    output.data()[thread_index] =
        apply_scalar_operation(input.data()[thread_index], scalar_value, operation, scalar_on_left);
  }

  /**
   * @brief 计算 reduction 的单个输入项。Compute one input term for a reduction.
   *
   * @param lhs 左侧只读 packed Tensor buffer。Left-hand side read-only packed Tensor buffer.
   * @param rhs 右侧只读 packed Tensor buffer；L2 范数中不使用。Right-hand side read-only packed Tensor buffer; unused for L2 norm.
   * @param index 输入线性索引。Input linear index.
   * @param operation reduction 运算码。Reduction operation code.
   * @return 当前索引贡献的部分和。Partial sum contribution for this index.
   */
  [[nodiscard]] __device__ auto reduction_term(cinder::ConstTensorDeviceBuffer lhs,
                                               cinder::ConstTensorDeviceBuffer rhs,
                                               std::size_t index,
                                               TensorReductionOperation operation) noexcept
      -> cinder::Tensor::value_type
  {
    switch (operation)
    {
    case TensorReductionOperation::inner:
      return lhs.data()[index] * rhs.data()[index];
    case TensorReductionOperation::l2_norm:
    {
      /**
       * @brief 输入元素值。Input element value.
       */
      const auto value = lhs.data()[index];

      return value * value;
    }
    }

    return 0.0F;
  }

  /**
   * @brief 完成 reduction 输出值转换。Finish the reduction output value conversion.
   *
   * @param sum reduction 的求和结果。Reduction sum result.
   * @param operation reduction 运算码。Reduction operation code.
   * @return 最终输出值。Final output value.
   */
  [[nodiscard]] __device__ auto finish_reduction_value(cinder::Tensor::value_type sum,
                                                       TensorReductionOperation operation) noexcept
      -> cinder::Tensor::value_type
  {
    switch (operation)
    {
    case TensorReductionOperation::inner:
      return sum;
    case TensorReductionOperation::l2_norm:
      return sqrtf(sum);
    }

    return sum;
  }

  /**
   * @brief 对一个 block 内的 shared memory 求和。Sum shared memory values inside one block.
   *
   * @param shared_values block 内 shared memory 数组。Shared memory array inside the block.
   * @return block 内求和结果。Block sum result.
   */
  [[nodiscard]] __device__ auto reduce_block_sum(cinder::Tensor::value_type *shared_values) noexcept
      -> cinder::Tensor::value_type
  {
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U)
    {
      if (threadIdx.x < stride)
      {
        shared_values[threadIdx.x] += shared_values[threadIdx.x + stride];
      }

      __syncthreads();
    }

    return shared_values[0];
  }

  /**
   * @brief 单 block 直接写 rank-0 reduction 输出的 kernel。Single-block kernel that writes a rank-0 reduction output directly.
   *
   * @param lhs 左侧只读 packed Tensor buffer。Left-hand side read-only packed Tensor buffer.
   * @param rhs 右侧只读 packed Tensor buffer；L2 范数中不使用。Right-hand side read-only packed Tensor buffer; unused for L2 norm.
   * @param output 输出 rank-0 Tensor buffer。Output rank-0 Tensor buffer.
   * @param operation reduction 运算码。Reduction operation code.
   */
  __global__ void tensor_reduction_direct_kernel(cinder::ConstTensorDeviceBuffer lhs,
                                                 cinder::ConstTensorDeviceBuffer rhs,
                                                 cinder::TensorDeviceBuffer output,
                                                 TensorReductionOperation operation)
  {
    /**
     * @brief 输入 packed storage header。Input packed storage header.
     */
    const auto *lhs_header = lhs.header();

    /**
     * @brief 当前线程的私有部分和。Thread-local partial sum.
     */
    cinder::Tensor::value_type sum = 0.0F;

    for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < lhs_header->element_count;
         index += static_cast<std::size_t>(blockDim.x))
    {
      sum += reduction_term(lhs, rhs, index, operation);
    }

    /**
     * @brief block 内 shared 部分和。Shared partial sums inside the block.
     */
    __shared__ cinder::Tensor::value_type shared_sums[k_tensor_reduction_threads_per_block];

    shared_sums[threadIdx.x] = sum;

    /**
     * @brief block 总和。Block sum.
     */
    const auto block_sum = reduce_block_sum(shared_sums);

    if (threadIdx.x == 0U)
    {
      /**
       * @brief 输出 packed storage header。Output packed storage header.
       */
      auto *output_header = output.header();

      output_header->rank = 0U;
      output_header->element_count = 1U;
      output.data()[0] = finish_reduction_value(block_sum, operation);
    }
  }

  /**
   * @brief reduction 第一阶段 kernel，写每个 block 的部分和。First-pass reduction kernel that writes one partial sum per block.
   *
   * @param lhs 左侧只读 packed Tensor buffer。Left-hand side read-only packed Tensor buffer.
   * @param rhs 右侧只读 packed Tensor buffer；L2 范数中不使用。Right-hand side read-only packed Tensor buffer; unused for L2 norm.
   * @param partials 输出 partial-sum Tensor buffer。Output partial-sum Tensor buffer.
   * @param operation reduction 运算码。Reduction operation code.
   */
  __global__ void tensor_reduction_partial_kernel(cinder::ConstTensorDeviceBuffer lhs,
                                                  cinder::ConstTensorDeviceBuffer rhs,
                                                  cinder::TensorDeviceBuffer partials,
                                                  TensorReductionOperation operation)
  {
    /**
     * @brief 输入 packed storage header。Input packed storage header.
     */
    const auto *lhs_header = lhs.header();

    /**
     * @brief 当前线程全局线性编号。Current global thread linear index.
     */
    auto index = (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)) +
                 static_cast<std::size_t>(threadIdx.x);

    /**
     * @brief grid-stride loop 的步长。Stride for the grid-stride loop.
     */
    const auto stride = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    if (index == 0U)
    {
      /**
       * @brief partial-sum packed storage header。Partial-sum packed storage header.
       */
      auto *partials_header = partials.header();

      partials_header->rank = 1U;
      partials_header->element_count = static_cast<std::size_t>(gridDim.x);
      partials.extents()[0] = static_cast<std::size_t>(gridDim.x);
    }

    /**
     * @brief 当前线程的私有部分和。Thread-local partial sum.
     */
    cinder::Tensor::value_type sum = 0.0F;

    for (; index < lhs_header->element_count; index += stride)
    {
      sum += reduction_term(lhs, rhs, index, operation);
    }

    /**
     * @brief block 内 shared 部分和。Shared partial sums inside the block.
     */
    __shared__ cinder::Tensor::value_type shared_sums[k_tensor_reduction_threads_per_block];

    shared_sums[threadIdx.x] = sum;

    /**
     * @brief block 总和。Block sum.
     */
    const auto block_sum = reduce_block_sum(shared_sums);

    if (threadIdx.x == 0U)
    {
      partials.data()[blockIdx.x] = block_sum;
    }
  }

  /**
   * @brief reduction 末阶段 kernel，把 partial sums 写成 rank-0 输出。Final-pass reduction kernel that writes partial sums into a rank-0 output.
   *
   * @param partials 输入 partial-sum Tensor buffer。Input partial-sum Tensor buffer.
   * @param output 输出 rank-0 Tensor buffer。Output rank-0 Tensor buffer.
   * @param operation reduction 运算码。Reduction operation code.
   */
  __global__ void tensor_reduction_final_kernel(cinder::ConstTensorDeviceBuffer partials,
                                                cinder::TensorDeviceBuffer output,
                                                TensorReductionOperation operation)
  {
    /**
     * @brief partial-sum packed storage header。Partial-sum packed storage header.
     */
    const auto *partials_header = partials.header();

    /**
     * @brief 当前线程的私有部分和。Thread-local partial sum.
     */
    cinder::Tensor::value_type sum = 0.0F;

    for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < partials_header->element_count;
         index += static_cast<std::size_t>(blockDim.x))
    {
      sum += partials.data()[index];
    }

    /**
     * @brief block 内 shared 部分和。Shared partial sums inside the block.
     */
    __shared__ cinder::Tensor::value_type shared_sums[k_tensor_reduction_threads_per_block];

    shared_sums[threadIdx.x] = sum;

    /**
     * @brief block 总和。Block sum.
     */
    const auto block_sum = reduce_block_sum(shared_sums);

    if (threadIdx.x == 0U)
    {
      /**
       * @brief 输出 packed storage header。Output packed storage header.
       */
      auto *output_header = output.header();

      output_header->rank = 0U;
      output_header->element_count = 1U;
      output.data()[0] = finish_reduction_value(block_sum, operation);
    }
  }

  /**
   * @brief 启动 Tensor-标量二元运算 kernel。Launch a Tensor-scalar binary operation kernel.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param output 输出 Tensor。Output Tensor.
   * @param scalar_value 标量值。Scalar value.
   * @param operation 标量运算码。Scalar operation code.
   * @param scalar_on_left 标量是否位于左操作数。Whether the scalar is the left-hand operand.
   */
  auto launch_tensor_scalar_kernel(const cinder::Tensor &input,
                                   cinder::Tensor &output,
                                   cinder::Tensor::value_type scalar_value,
                                   TensorScalarOperation operation,
                                   bool scalar_on_left) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(input.size(), 1U);

    /**
     * @brief kernel block 数量。Kernel block count.
     */
    const auto block_count =
        (work_items + static_cast<std::size_t>(k_tensor_scalar_threads_per_block) - 1U) /
        static_cast<std::size_t>(k_tensor_scalar_threads_per_block);

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
    const dim3 block(k_tensor_scalar_threads_per_block);

    tensor_scalar_kernel<<<grid, block>>>(
        input.device_buffer(), output.device_buffer(), scalar_value, operation, scalar_on_left);
    check_cuda(cudaGetLastError(), "launch Tensor scalar operation kernel");
  }

  /**
   * @brief 计算 reduction 第一阶段 block 数。Compute the first-pass block count for a reduction.
   *
   * @param element_count 输入元素总数。Input element count.
   * @return 第一阶段 block 数。First-pass block count.
   */
  [[nodiscard]] auto reduction_block_count(std::size_t element_count) -> std::size_t
  {
    if (element_count == 0U)
    {
      return 1U;
    }

    /**
     * @brief 每个 block 目标处理元素数。Target element count per block.
     */
    constexpr auto elements_per_block =
        static_cast<std::size_t>(k_tensor_reduction_threads_per_block) * k_tensor_reduction_items_per_thread;

    /**
     * @brief 向上取整后的 block 数。Rounded-up block count.
     */
    const auto rounded_blocks = (element_count / elements_per_block) +
                                ((element_count % elements_per_block) == 0U ? 0U : 1U);

    return std::min<std::size_t>(std::max<std::size_t>(rounded_blocks, 1U), k_tensor_reduction_max_blocks);
  }

  /**
   * @brief 启动单 block direct reduction kernel。Launch the single-block direct reduction kernel.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor buffer；L2 范数中可为空。Right-hand side Tensor buffer; may be empty for L2 norm.
   * @param output 输出 rank-0 Tensor。Output rank-0 Tensor.
   * @param operation reduction 运算码。Reduction operation code.
   */
  auto launch_direct_reduction_kernel(const cinder::Tensor &lhs,
                                      cinder::ConstTensorDeviceBuffer rhs,
                                      cinder::Tensor &output,
                                      TensorReductionOperation operation) -> void
  {
    /**
     * @brief CUDA grid 维度。CUDA grid dimensions.
     */
    const dim3 grid(1U);

    /**
     * @brief CUDA block 维度。CUDA block dimensions.
     */
    const dim3 block(k_tensor_reduction_threads_per_block);

    tensor_reduction_direct_kernel<<<grid, block>>>(lhs.device_buffer(), rhs, output.device_buffer(), operation);
    check_cuda(cudaGetLastError(), "launch Tensor direct reduction kernel");
  }

  /**
   * @brief 启动两阶段 reduction kernels。Launch the two-pass reduction kernels.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor buffer；L2 范数中可为空。Right-hand side Tensor buffer; may be empty for L2 norm.
   * @param partials partial-sum Tensor。Partial-sum Tensor.
   * @param output 输出 rank-0 Tensor。Output rank-0 Tensor.
   * @param operation reduction 运算码。Reduction operation code.
   */
  auto launch_two_pass_reduction_kernels(const cinder::Tensor &lhs,
                                         cinder::ConstTensorDeviceBuffer rhs,
                                         cinder::Tensor &partials,
                                         cinder::Tensor &output,
                                         TensorReductionOperation operation) -> void
  {
    if (partials.size() > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    {
      throw std::length_error("Tensor reduction partial count is too large for a 1D CUDA launch");
    }

    /**
     * @brief 第一阶段 CUDA grid 维度。First-pass CUDA grid dimensions.
     */
    const dim3 partial_grid(static_cast<unsigned int>(partials.size()));

    /**
     * @brief CUDA block 维度。CUDA block dimensions.
     */
    const dim3 block(k_tensor_reduction_threads_per_block);

    tensor_reduction_partial_kernel<<<partial_grid, block>>>(
        lhs.device_buffer(), rhs, partials.device_buffer(), operation);
    check_cuda(cudaGetLastError(), "launch Tensor partial reduction kernel");

    /**
     * @brief 末阶段 CUDA grid 维度。Final-pass CUDA grid dimensions.
     */
    const dim3 final_grid(1U);

    tensor_reduction_final_kernel<<<final_grid, block>>>(
        static_cast<const cinder::Tensor &>(partials).device_buffer(), output.device_buffer(), operation);
    check_cuda(cudaGetLastError(), "launch Tensor final reduction kernel");
  }

  /**
   * @brief 校验 Tensor-标量运算输入。Validate Tensor-scalar operation input.
   *
   * @param input 输入 Tensor。Input Tensor.
   */
  auto validate_scalar_input(const cinder::Tensor &input) -> void
  {
    if (input.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }
  }

  /**
   * @brief 校验内积输入。Validate inner-product inputs.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   */
  auto validate_inner_inputs(const cinder::Tensor &lhs, const cinder::Tensor &rhs) -> void
  {
    if (lhs.empty() || rhs.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }

    if (lhs.shape() != rhs.shape())
    {
      throw std::invalid_argument("Tensor shapes must match for inner product");
    }
  }

  /**
   * @brief 校验范数输入。Validate norm input.
   *
   * @param input 输入 Tensor。Input Tensor.
   */
  auto validate_norm_input(const cinder::Tensor &input) -> void
  {
    if (input.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }
  }

} // namespace

namespace cinder
{

  /**
   * @brief 计算当前 Tensor 与另一个 Tensor 的内积。Compute the inner product of this Tensor and another Tensor.
   *
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return rank-0 Tensor 形式的内积结果。Inner-product result as a rank-0 Tensor.
   */
  auto Tensor::inner(const Tensor &rhs) const -> Tensor
  {
    return cinder::inner(*this, rhs);
  }

  /**
   * @brief 计算当前 Tensor 与另一个 Tensor 的点积别名。Compute the dot-product alias of this Tensor and another Tensor.
   *
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return rank-0 Tensor 形式的点积结果。Dot-product result as a rank-0 Tensor.
   */
  auto Tensor::dot(const Tensor &rhs) const -> Tensor
  {
    return cinder::inner(*this, rhs);
  }

  /**
   * @brief 计算当前 Tensor 的 L2 范数。Compute the L2 norm of this Tensor.
   *
   * @return rank-0 Tensor 形式的 L2 范数。L2-norm result as a rank-0 Tensor.
   */
  auto Tensor::norm() const -> Tensor
  {
    return cinder::norm(*this);
  }

  /**
   * @brief 执行逐元素 Tensor-标量二元运算。Run an elementwise Tensor-scalar binary operation.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param scalar_value 标量值。Scalar value.
   * @param operation 二元运算码。Binary operation code.
   * @param scalar_on_left 标量是否位于左操作数。Whether the scalar is the left-hand operand.
   * @return 运算结果 Tensor。Result Tensor.
   */
  auto Tensor::scalar(const Tensor &input, value_type scalar_value, unsigned int operation, bool scalar_on_left) -> Tensor
  {
    validate_scalar_input(input);

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(input.extents_, UninitializedTag{});

    launch_tensor_scalar_kernel(
        input, output, scalar_value, static_cast<TensorScalarOperation>(operation), scalar_on_left);

    return output;
  }

  /**
   * @brief Tensor 与标量逐元素加法运算符。Elementwise Tensor-plus-scalar operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧标量。Right-hand side scalar.
   * @return 加法结果 Tensor。Addition result Tensor.
   */
  auto operator+(const Tensor &lhs, Tensor::value_type rhs) -> Tensor
  {
    return Tensor::scalar(lhs, rhs, static_cast<unsigned int>(TensorScalarOperation::plus), false);
  }

  /**
   * @brief 标量与 Tensor 逐元素加法运算符。Elementwise scalar-plus-Tensor operator.
   *
   * @param lhs 左侧标量。Left-hand side scalar.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 加法结果 Tensor。Addition result Tensor.
   */
  auto operator+(Tensor::value_type lhs, const Tensor &rhs) -> Tensor
  {
    return Tensor::scalar(rhs, lhs, static_cast<unsigned int>(TensorScalarOperation::plus), true);
  }

  /**
   * @brief Tensor 与标量逐元素减法运算符。Elementwise Tensor-minus-scalar operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧标量。Right-hand side scalar.
   * @return 减法结果 Tensor。Subtraction result Tensor.
   */
  auto operator-(const Tensor &lhs, Tensor::value_type rhs) -> Tensor
  {
    return Tensor::scalar(lhs, rhs, static_cast<unsigned int>(TensorScalarOperation::minus), false);
  }

  /**
   * @brief 标量与 Tensor 逐元素减法运算符。Elementwise scalar-minus-Tensor operator.
   *
   * @param lhs 左侧标量。Left-hand side scalar.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 减法结果 Tensor。Subtraction result Tensor.
   */
  auto operator-(Tensor::value_type lhs, const Tensor &rhs) -> Tensor
  {
    return Tensor::scalar(rhs, lhs, static_cast<unsigned int>(TensorScalarOperation::minus), true);
  }

  /**
   * @brief Tensor 与标量逐元素乘法运算符。Elementwise Tensor-times-scalar operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧标量。Right-hand side scalar.
   * @return 乘法结果 Tensor。Multiplication result Tensor.
   */
  auto operator*(const Tensor &lhs, Tensor::value_type rhs) -> Tensor
  {
    return Tensor::scalar(lhs, rhs, static_cast<unsigned int>(TensorScalarOperation::times), false);
  }

  /**
   * @brief 标量与 Tensor 逐元素乘法运算符。Elementwise scalar-times-Tensor operator.
   *
   * @param lhs 左侧标量。Left-hand side scalar.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 乘法结果 Tensor。Multiplication result Tensor.
   */
  auto operator*(Tensor::value_type lhs, const Tensor &rhs) -> Tensor
  {
    return Tensor::scalar(rhs, lhs, static_cast<unsigned int>(TensorScalarOperation::times), true);
  }

  /**
   * @brief Tensor 与标量逐元素除法运算符。Elementwise Tensor-divided-by-scalar operator.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧标量。Right-hand side scalar.
   * @return 除法结果 Tensor。Division result Tensor.
   */
  auto operator/(const Tensor &lhs, Tensor::value_type rhs) -> Tensor
  {
    return Tensor::scalar(lhs, rhs, static_cast<unsigned int>(TensorScalarOperation::quotient), false);
  }

  /**
   * @brief 标量与 Tensor 逐元素除法运算符。Elementwise scalar-divided-by-Tensor operator.
   *
   * @param lhs 左侧标量。Left-hand side scalar.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 除法结果 Tensor。Division result Tensor.
   */
  auto operator/(Tensor::value_type lhs, const Tensor &rhs) -> Tensor
  {
    return Tensor::scalar(rhs, lhs, static_cast<unsigned int>(TensorScalarOperation::quotient), true);
  }

  /**
   * @brief 计算两个 Tensor 的内积。Compute the inner product of two Tensors.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return rank-0 Tensor 形式的内积结果。Inner-product result as a rank-0 Tensor.
   */
  auto inner(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    validate_inner_inputs(lhs, rhs);

    /**
     * @brief 未初始化 rank-0 输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized rank-0 output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(std::vector<Tensor::size_type>{}, Tensor::UninitializedTag{});

    /**
     * @brief reduction 第一阶段 block 数。First-pass reduction block count.
     */
    const auto block_count = reduction_block_count(lhs.size());

    if (block_count == 1U)
    {
      launch_direct_reduction_kernel(lhs, rhs.device_buffer(), output, TensorReductionOperation::inner);
      return output;
    }

    /**
     * @brief partial-sum Tensor；kernel 会写 metadata 和 data。Partial-sum Tensor; the kernel writes metadata and data.
     */
    Tensor partials(std::vector<Tensor::size_type>{block_count}, Tensor::UninitializedTag{});

    launch_two_pass_reduction_kernels(lhs, rhs.device_buffer(), partials, output, TensorReductionOperation::inner);

    return output;
  }

  /**
   * @brief 计算 Tensor 的 L2 范数。Compute the L2 norm of a Tensor.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @return rank-0 Tensor 形式的 L2 范数。L2-norm result as a rank-0 Tensor.
   */
  auto norm(const Tensor &input) -> Tensor
  {
    validate_norm_input(input);

    /**
     * @brief 未初始化 rank-0 输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized rank-0 output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(std::vector<Tensor::size_type>{}, Tensor::UninitializedTag{});

    /**
     * @brief reduction 第一阶段 block 数。First-pass reduction block count.
     */
    const auto block_count = reduction_block_count(input.size());

    if (block_count == 1U)
    {
      launch_direct_reduction_kernel(input, ConstTensorDeviceBuffer{}, output, TensorReductionOperation::l2_norm);
      return output;
    }

    /**
     * @brief partial-sum Tensor；kernel 会写 metadata 和 data。Partial-sum Tensor; the kernel writes metadata and data.
     */
    Tensor partials(std::vector<Tensor::size_type>{block_count}, Tensor::UninitializedTag{});

    launch_two_pass_reduction_kernels(input, ConstTensorDeviceBuffer{}, partials, output, TensorReductionOperation::l2_norm);

    return output;
  }

} // namespace cinder
