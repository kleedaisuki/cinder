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
   * @brief Tensor mode 乘法 kernel 的线程块大小。Thread block size for the Tensor mode-multiplication kernel.
   */
  constexpr unsigned int k_tensor_mode_multiply_threads_per_block = 256U;

  /**
   * @brief Tensor mode 乘法支持的最大 rank。Maximum supported rank for Tensor mode multiplication.
   */
  constexpr std::size_t k_tensor_mode_multiply_max_rank = 32U;

  /**
   * @brief Tensor mode 乘法 kernel 参数计划。Kernel-parameter plan for Tensor mode multiplication.
   *
   * @note 该结构按值传入 kernel，避免为 shape/stride 元数据做额外 device allocation 或 H2D copy。
   *       This structure is passed to the kernel by value, avoiding extra device allocation or H2D copies for shape and stride metadata.
   */
  struct TensorModeMultiplyPlan final
  {
    /**
     * @brief 输出张量秩。Output tensor rank.
     */
    std::size_t output_rank{};

    /**
     * @brief 被矩阵替换的 mode 轴。Mode axis replaced by the matrix.
     */
    std::size_t mode_axis{};

    /**
     * @brief 输入 Tensor 中 mode 轴的 extent。Extent of the selected mode axis in the input Tensor.
     */
    std::size_t input_mode_extent{};

    /**
     * @brief 输入 Tensor 中 mode 轴的 row-major stride。Row-major stride of the selected mode axis in the input Tensor.
     */
    std::size_t input_mode_stride{};

    /**
     * @brief 输出 dense 元素总数。Dense output element count.
     */
    std::size_t output_element_count{};

    /**
     * @brief 输出每个轴的 extent。Per-axis output extents.
     */
    std::size_t output_extents[k_tensor_mode_multiply_max_rank]{};

    /**
     * @brief 输入每个轴的 row-major stride。Per-axis input row-major strides.
     */
    std::size_t input_strides[k_tensor_mode_multiply_max_rank]{};
  };

  /**
   * @brief host 侧 mode 乘法准备结果。Host-side mode-multiplication preparation result.
   */
  struct TensorModeMultiplyPreparation final
  {
    /**
     * @brief 可按值传入 kernel 的 mode 乘法计划。Mode-multiplication plan that can be passed to the kernel.
     */
    TensorModeMultiplyPlan plan{};

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
   * @brief 校验 Tensor rank 是否可放入 kernel 参数计划。Validate whether a Tensor rank fits in the kernel-parameter plan.
   *
   * @param rank 输入 Tensor rank。Input Tensor rank.
   */
  auto validate_rank(std::size_t rank) -> void
  {
    if (rank <= k_tensor_mode_multiply_max_rank)
    {
      return;
    }

    throw std::length_error("Tensor rank exceeds Tensor mode multiplication limit");
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
      stride = checked_multiply(stride, extents[axis], "Tensor mode multiplication stride");
    }

    return strides;
  }

  /**
   * @brief 计算 dense shape 的元素总数。Compute the element count for a dense shape.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   * @return 元素总数。Element count.
   */
  [[nodiscard]] auto checked_element_count(const std::vector<cinder::Tensor::size_type> &extents)
      -> cinder::Tensor::size_type
  {
    /**
     * @brief 累积元素总数。Accumulated element count.
     */
    cinder::Tensor::size_type count = 1U;

    for (const auto extent : extents)
    {
      count = checked_multiply(count, extent, "Tensor mode multiplication element count");
    }

    return count;
  }

  /**
   * @brief 校验 mode 乘法输入组合。Validate a Tensor mode-multiplication input combination.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param matrix 左乘 mode fiber 的矩阵。Matrix that left-multiplies mode fibers.
   * @param mode 被替换的 mode 轴。Mode axis to replace.
   */
  auto validate_mode_multiply_inputs(const cinder::Tensor &input,
                                     const cinder::Tensor &matrix,
                                     cinder::Tensor::size_type mode) -> void
  {
    validate_rank(input.rank());

    if (matrix.rank() != 2U)
    {
      throw std::invalid_argument("Tensor mode multiplication matrix must have rank 2");
    }

    if (mode >= input.rank())
    {
      throw std::invalid_argument("Tensor mode multiplication mode is out of range");
    }

    if (matrix.shape()[1] != input.shape()[mode])
    {
      throw std::invalid_argument("Tensor mode multiplication matrix input extent must match selected mode extent");
    }
  }

  /**
   * @brief 构造 mode 乘法输出 shape 与 kernel 参数计划。Build the mode-multiplication output shape and kernel-parameter plan.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param matrix 左乘 mode fiber 的矩阵。Matrix that left-multiplies mode fibers.
   * @param mode 被替换的 mode 轴。Mode axis to replace.
   * @return mode 乘法准备结果。Mode-multiplication preparation result.
   */
  [[nodiscard]] auto prepare_tensor_mode_multiply(const cinder::Tensor &input,
                                                  const cinder::Tensor &matrix,
                                                  cinder::Tensor::size_type mode)
      -> TensorModeMultiplyPreparation
  {
    validate_mode_multiply_inputs(input, matrix, mode);

    /**
     * @brief 输入 row-major stride。Input row-major strides.
     */
    const auto input_strides = row_major_strides(input.shape());

    /**
     * @brief mode 乘法准备结果。Mode-multiplication preparation result.
     */
    TensorModeMultiplyPreparation preparation;

    preparation.output_extents = input.shape();
    preparation.output_extents[mode] = matrix.shape()[0];

    preparation.plan.output_rank = input.rank();
    preparation.plan.mode_axis = mode;
    preparation.plan.input_mode_extent = input.shape()[mode];
    preparation.plan.input_mode_stride = input_strides[mode];
    preparation.plan.output_element_count = checked_element_count(preparation.output_extents);

    for (std::size_t axis = 0U; axis < input.rank(); ++axis)
    {
      preparation.plan.output_extents[axis] = preparation.output_extents[axis];
      preparation.plan.input_strides[axis] = input_strides[axis];
    }

    return preparation;
  }

  /**
   * @brief 根据输出线性索引计算输入 base offset 与矩阵行。Compute the input base offset and matrix row from an output linear index.
   *
   * @param output_index 输出线性索引。Output linear index.
   * @param plan mode 乘法计划。Mode-multiplication plan.
   * @param matrix_row 输出参数，矩阵行坐标。Output parameter for the matrix row coordinate.
   * @return 不含 mode 坐标贡献的输入 base offset。Input base offset excluding the selected mode coordinate.
   */
  [[nodiscard]] __device__ auto input_base_offset_for_output_index(std::size_t output_index,
                                                                   const TensorModeMultiplyPlan &plan,
                                                                   std::size_t &matrix_row) noexcept -> std::size_t
  {
    /**
     * @brief 输入 Tensor 中对应 fiber 的 base offset。Base offset of the corresponding input Tensor fiber.
     */
    std::size_t input_offset = 0U;

    for (std::size_t reverse_axis = 0U; reverse_axis < plan.output_rank; ++reverse_axis)
    {
      /**
       * @brief 当前输出轴位置。Current output axis position.
       */
      const auto axis = plan.output_rank - reverse_axis - 1U;

      /**
       * @brief 当前输出轴 extent。Current output axis extent.
       */
      const auto output_extent = plan.output_extents[axis];

      /**
       * @brief 当前输出坐标。Current output coordinate.
       */
      const auto coordinate = output_index % output_extent;

      output_index /= output_extent;

      if (axis == plan.mode_axis)
      {
        matrix_row = coordinate;
        continue;
      }

      input_offset += coordinate * plan.input_strides[axis];
    }

    return input_offset;
  }

  /**
   * @brief Tensor mode 乘法 kernel。Tensor mode-multiplication kernel.
   *
   * @param input 输入只读 packed Tensor buffer。Input read-only packed Tensor buffer.
   * @param matrix 左乘 mode fiber 的只读 rank-2 矩阵 buffer。Read-only rank-2 matrix buffer that left-multiplies mode fibers.
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param plan mode 乘法计划。Mode-multiplication plan.
   *
   * @note 第 0 个线程写 output header/extents；数据线程只依赖输入 data、矩阵 data 与按值传入的 plan，不读取输出 metadata。
   *       Thread 0 writes the output header/extents; data threads depend only on input data, matrix data, and the by-value plan, and do not read output metadata.
   */
  __global__ void tensor_mode_multiply_kernel(cinder::ConstTensorDeviceBuffer input,
                                              cinder::ConstTensorDeviceBuffer matrix,
                                              cinder::TensorDeviceBuffer output,
                                              TensorModeMultiplyPlan plan)
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
     * @brief 当前输出元素对应的矩阵行。Matrix row for the current output element.
     */
    std::size_t matrix_row = 0U;

    /**
     * @brief 当前输出元素对应输入 fiber 的 base offset。Input fiber base offset for the current output element.
     */
    const auto input_base_offset = input_base_offset_for_output_index(thread_index, plan, matrix_row);

    /**
     * @brief 当前输出元素的 mode 乘法累加值。Mode-multiplication accumulator for the current output element.
     */
    cinder::Tensor::value_type sum = 0.0F;

    for (std::size_t mode_coordinate = 0U; mode_coordinate < plan.input_mode_extent; ++mode_coordinate)
    {
      /**
       * @brief 当前输入元素 offset。Current input element offset.
       */
      const auto input_offset = input_base_offset + (mode_coordinate * plan.input_mode_stride);

      /**
       * @brief 当前矩阵元素 offset。Current matrix element offset.
       */
      const auto matrix_offset = (matrix_row * plan.input_mode_extent) + mode_coordinate;

      sum += input.data()[input_offset] * matrix.data()[matrix_offset];
    }

    output.data()[thread_index] = sum;
  }

  /**
   * @brief 启动 Tensor mode 乘法 kernel。Launch the Tensor mode-multiplication kernel.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param matrix 左乘 mode fiber 的矩阵。Matrix that left-multiplies mode fibers.
   * @param output 输出 Tensor。Output Tensor.
   * @param plan mode 乘法计划。Mode-multiplication plan.
   */
  auto launch_tensor_mode_multiply_kernel(const cinder::Tensor &input,
                                          const cinder::Tensor &matrix,
                                          cinder::Tensor &output,
                                          TensorModeMultiplyPlan plan) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(output.size(), 1U);

    /**
     * @brief kernel block 数量。Kernel block count.
     */
    const auto block_count =
        (work_items + static_cast<std::size_t>(k_tensor_mode_multiply_threads_per_block) - 1U) /
        static_cast<std::size_t>(k_tensor_mode_multiply_threads_per_block);

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
    const dim3 block(k_tensor_mode_multiply_threads_per_block);

    tensor_mode_multiply_kernel<<<grid, block>>>(input.device_buffer(), matrix.device_buffer(), output.device_buffer(), plan);
    check_cuda(cudaGetLastError(), "launch Tensor mode multiplication kernel");
  }

} // namespace

namespace cinder
{

  /**
   * @brief 沿指定 mode 与矩阵相乘。Multiply this Tensor by a matrix along a selected mode.
   *
   * @param matrix 左乘当前 mode fiber 的 rank-2 矩阵。Rank-2 matrix that left-multiplies fibers of the selected mode.
   * @param mode 当前 Tensor 中被替换的 mode 轴。Mode axis in this Tensor to replace.
   * @return mode 乘法结果 Tensor。Mode-multiplication result Tensor.
   */
  auto Tensor::mode_multiply(const Tensor &matrix, size_type mode) const -> Tensor
  {
    return cinder::mode_multiply(*this, matrix, mode);
  }

  /**
   * @brief 沿指定 mode 计算 Tensor 与矩阵的乘法。Compute the mode multiplication of a Tensor and a matrix.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param matrix 左乘 mode fiber 的 rank-2 矩阵。Rank-2 matrix that left-multiplies mode fibers.
   * @param mode 输入 Tensor 中被替换的 mode 轴。Mode axis in the input Tensor to replace.
   * @return mode 乘法结果 Tensor。Mode-multiplication result Tensor.
   */
  auto mode_multiply(const Tensor &input, const Tensor &matrix, Tensor::size_type mode) -> Tensor
  {
    if (input.empty() || matrix.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }

    /**
     * @brief host 侧 mode 乘法准备结果。Host-side mode-multiplication preparation result.
     */
    const auto preparation = prepare_tensor_mode_multiply(input, matrix, mode);

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(preparation.output_extents, Tensor::UninitializedTag{});

    launch_tensor_mode_multiply_kernel(input, matrix, output, preparation.plan);

    return output;
  }

} // namespace cinder
