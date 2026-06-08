#include "cinder/tensor.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

  /**
   * @brief Tensor 转置 kernel 的线程块大小。Thread block size for the Tensor transpose kernel.
   */
  constexpr unsigned int k_tensor_transpose_threads_per_block = 256U;

  /**
   * @brief Tensor 转置支持的最大 rank。Maximum supported rank for Tensor transpose.
   */
  constexpr std::size_t k_tensor_transpose_max_rank = 32U;

  /**
   * @brief Tensor 转置 kernel 参数计划。Kernel-parameter plan for Tensor transpose.
   *
   * @note 该结构按值传入 kernel，避免为轴置换和 stride 元数据做额外 device allocation 或 H2D copy。
   *       This structure is passed to the kernel by value, avoiding extra device allocation or H2D copies for axis permutation and stride metadata.
   */
  struct TensorTransposePlan final
  {
    /**
     * @brief 输出张量秩。Output tensor rank.
     */
    std::size_t output_rank{};

    /**
     * @brief 输出 dense 元素总数。Dense output element count.
     */
    std::size_t output_element_count{};

    /**
     * @brief 输出每个轴的 extent。Per-axis output extents.
     */
    std::size_t output_extents[k_tensor_transpose_max_rank]{};

    /**
     * @brief 输出轴对应的输入 row-major stride。Input row-major stride for each output axis.
     */
    std::size_t input_strides[k_tensor_transpose_max_rank]{};
  };

  /**
   * @brief host 侧转置准备结果。Host-side transpose preparation result.
   */
  struct TensorTransposePreparation final
  {
    /**
     * @brief 可按值传入 kernel 的转置计划。Transpose plan that can be passed to the kernel.
     */
    TensorTransposePlan plan{};

    /**
     * @brief 输出 Tensor 的 host 侧 shape。Host-side shape for the output Tensor.
     */
    std::vector<cinder::Tensor::size_type> output_extents{};
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
   * @brief 计算 dense row-major stride。Compute dense row-major strides.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   * @return 每个轴的 row-major stride。Per-axis row-major strides.
   */
  [[nodiscard]] auto row_major_strides(const std::vector<cinder::Tensor::size_type> &extents)
      -> std::vector<cinder::Tensor::size_type>
  {
    /**
     * @brief 每个轴的 stride 输出。Output strides for each axis.
     */
    std::vector<cinder::Tensor::size_type> strides(extents.size(), 1U);

    /**
     * @brief 当前后缀元素乘积。Current suffix element product.
     */
    cinder::Tensor::size_type stride = 1U;

    for (std::size_t reverse_axis = extents.size(); reverse_axis > 0U; --reverse_axis)
    {
      /**
       * @brief 当前轴编号。Current axis index.
       */
      const auto axis = reverse_axis - 1U;

      strides[axis] = stride;
      stride = checked_multiply(stride, extents[axis], "Tensor transpose stride");
    }

    return strides;
  }

  /**
   * @brief 构造默认反序轴置换。Build the default reversed axis permutation.
   *
   * @param rank 输入 Tensor rank。Input Tensor rank.
   * @return 反序轴置换。Reversed axis permutation.
   */
  [[nodiscard]] auto reversed_axes(std::size_t rank) -> std::vector<cinder::Tensor::size_type>
  {
    /**
     * @brief 默认轴置换。Default axis permutation.
     */
    std::vector<cinder::Tensor::size_type> axes(rank, 0U);

    for (std::size_t axis = 0U; axis < rank; ++axis)
    {
      axes[axis] = rank - axis - 1U;
    }

    return axes;
  }

  /**
   * @brief 校验转置 rank 是否可放入 kernel 参数计划。Validate whether transpose rank fits in the kernel-parameter plan.
   *
   * @param rank 输入 Tensor rank。Input Tensor rank.
   */
  auto validate_rank(std::size_t rank) -> void
  {
    if (rank <= k_tensor_transpose_max_rank)
    {
      return;
    }

    throw std::length_error("Tensor rank exceeds Tensor transpose limit");
  }

  /**
   * @brief 校验轴置换并返回每个输入轴是否出现的 mask。Validate an axis permutation and return a per-input-axis mask.
   *
   * @param rank 输入 Tensor rank。Input Tensor rank.
   * @param axes 输出轴到输入轴的置换。Permutation from output axes to input axes.
   */
  auto validate_axes(std::size_t rank, const std::vector<cinder::Tensor::size_type> &axes) -> void
  {
    if (axes.size() != rank)
    {
      throw std::invalid_argument("Tensor transpose axes must match Tensor rank");
    }

    /**
     * @brief 输入轴是否已经出现的 mask。Mask indicating whether each input axis has already appeared.
     */
    std::vector<unsigned char> seen(rank, 0U);

    for (const auto axis : axes)
    {
      if (axis >= rank)
      {
        throw std::invalid_argument("Tensor transpose axis is out of range");
      }

      if (seen[axis] != 0U)
      {
        throw std::invalid_argument("Tensor transpose axes must be unique");
      }

      seen[axis] = 1U;
    }
  }

  /**
   * @brief 构造转置输出 shape 与 kernel 参数计划。Build the transpose output shape and kernel-parameter plan.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param axes 输出轴到输入轴的置换。Permutation from output axes to input axes.
   * @return 转置准备结果。Transpose preparation result.
   */
  [[nodiscard]] auto prepare_tensor_transpose(const cinder::Tensor &input,
                                              const std::vector<cinder::Tensor::size_type> &axes)
      -> TensorTransposePreparation
  {
    validate_rank(input.rank());
    validate_axes(input.rank(), axes);

    /**
     * @brief 输入 row-major stride。Input row-major strides.
     */
    const auto input_strides = row_major_strides(input.shape());

    /**
     * @brief 转置准备结果。Transpose preparation result.
     */
    TensorTransposePreparation preparation;

    preparation.output_extents.reserve(input.rank());
    preparation.plan.output_rank = input.rank();
    preparation.plan.output_element_count = input.size();

    for (std::size_t output_axis = 0U; output_axis < input.rank(); ++output_axis)
    {
      /**
       * @brief 当前输出轴对应的输入轴。Input axis selected for the current output axis.
       */
      const auto input_axis = axes[output_axis];

      /**
       * @brief 当前输出轴 extent。Extent for the current output axis.
       */
      const auto output_extent = input.shape()[input_axis];

      preparation.output_extents.push_back(output_extent);
      preparation.plan.output_extents[output_axis] = output_extent;
      preparation.plan.input_strides[output_axis] = input_strides[input_axis];
    }

    return preparation;
  }

  /**
   * @brief 根据输出线性索引计算输入 linear offset。Compute the input linear offset from an output linear index.
   *
   * @param output_index 输出线性索引。Output linear index.
   * @param plan 转置计划。Transpose plan.
   * @return 输入 linear offset。Input linear offset.
   */
  [[nodiscard]] __device__ auto input_offset_for_output_index(std::size_t output_index,
                                                              const TensorTransposePlan &plan) noexcept -> std::size_t
  {
    /**
     * @brief 输入 Tensor 中对应坐标的 linear offset。Linear offset for the corresponding input Tensor coordinate.
     */
    std::size_t input_offset = 0U;

    for (std::size_t reverse_axis = 0U; reverse_axis < plan.output_rank; ++reverse_axis)
    {
      /**
       * @brief 当前输出轴位置。Current output axis position.
       */
      const auto output_axis = plan.output_rank - reverse_axis - 1U;

      /**
       * @brief 当前输出轴 extent。Current output axis extent.
       */
      const auto output_extent = plan.output_extents[output_axis];

      /**
       * @brief 当前输出坐标。Current output coordinate.
       */
      const auto coordinate = output_index % output_extent;

      output_index /= output_extent;
      input_offset += coordinate * plan.input_strides[output_axis];
    }

    return input_offset;
  }

  /**
   * @brief Tensor 转置 kernel。Tensor transpose kernel.
   *
   * @param input 输入只读 packed Tensor buffer。Input read-only packed Tensor buffer.
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param plan 转置计划。Transpose plan.
   *
   * @note 第 0 个线程写 output header/extents；数据线程只依赖输入 data 与按值传入的 plan，不读取输出 metadata。
   *       Thread 0 writes the output header/extents; data threads depend only on input data and the by-value plan, and do not read output metadata.
   */
  __global__ void tensor_transpose_kernel(cinder::ConstTensorDeviceBuffer input,
                                          cinder::TensorDeviceBuffer output,
                                          TensorTransposePlan plan)
  {
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

      output_header->rank = plan.output_rank;
      output_header->element_count = plan.output_element_count;

      for (std::size_t axis = 0U; axis < plan.output_rank; ++axis)
      {
        output.extents()[axis] = plan.output_extents[axis];
      }
    }

    if (thread_index >= plan.output_element_count)
    {
      return;
    }

    /**
     * @brief 当前输出元素对应的输入 linear offset。Input linear offset for the current output element.
     */
    const auto input_offset = input_offset_for_output_index(thread_index, plan);

    output.data()[thread_index] = input.data()[input_offset];
  }

  /**
   * @brief 启动 Tensor 转置 kernel。Launch the Tensor transpose kernel.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param output 输出 Tensor。Output Tensor.
   * @param plan 转置计划。Transpose plan.
   */
  auto launch_tensor_transpose_kernel(const cinder::Tensor &input,
                                      cinder::Tensor &output,
                                      TensorTransposePlan plan) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(output.size(), 1U);

    /**
     * @brief kernel block 数量。Kernel block count.
     */
    const auto block_count =
        (work_items + static_cast<std::size_t>(k_tensor_transpose_threads_per_block) - 1U) /
        static_cast<std::size_t>(k_tensor_transpose_threads_per_block);

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
    const dim3 block(k_tensor_transpose_threads_per_block);

    tensor_transpose_kernel<<<grid, block>>>(input.device_buffer(), output.device_buffer(), plan);
    check_cuda(cudaGetLastError(), "launch Tensor transpose kernel");
  }

} // namespace

namespace cinder
{

  /**
   * @brief 反转所有轴顺序并返回转置 Tensor。Return a transposed Tensor with all axes reversed.
   *
   * @return 转置结果 Tensor。Transposed Tensor result.
   */
  auto Tensor::transpose() const -> Tensor
  {
    return cinder::transpose(*this);
  }

  /**
   * @brief 按给定轴置换返回转置 Tensor。Return a transposed Tensor using the given axis permutation.
   *
   * @param axes 输出轴到输入轴的置换。Permutation from output axes to input axes.
   * @return 转置结果 Tensor。Transposed Tensor result.
   */
  auto Tensor::transpose(const std::vector<size_type> &axes) const -> Tensor
  {
    return cinder::transpose(*this, axes);
  }

  /**
   * @brief 反转 Tensor 所有轴顺序。Reverse all Tensor axes.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @return 转置结果 Tensor。Transposed Tensor result.
   */
  auto transpose(const Tensor &input) -> Tensor
  {
    return cinder::transpose(input, reversed_axes(input.rank()));
  }

  /**
   * @brief 按给定轴置换转置 Tensor。Transpose a Tensor using the given axis permutation.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param axes 输出轴到输入轴的置换。Permutation from output axes to input axes.
   * @return 转置结果 Tensor。Transposed Tensor result.
   */
  auto transpose(const Tensor &input, const std::vector<Tensor::size_type> &axes) -> Tensor
  {
    if (input.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }

    /**
     * @brief host 侧转置准备结果。Host-side transpose preparation result.
     */
    const auto preparation = prepare_tensor_transpose(input, axes);

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(preparation.output_extents, Tensor::UninitializedTag{});

    launch_tensor_transpose_kernel(input, output, preparation.plan);

    return output;
  }

} // namespace cinder
