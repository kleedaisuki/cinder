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
   * @brief Tensor 缩并 kernel 的线程块大小。Thread block size for the Tensor contraction kernel.
   */
  constexpr unsigned int k_tensor_contraction_threads_per_block = 256U;

  /**
   * @brief 单个输入 Tensor 缩并支持的最大 rank。Maximum supported rank for one input Tensor in contraction.
   */
  constexpr std::size_t k_tensor_contraction_max_rank = 32U;

  /**
   * @brief Tensor 缩并 kernel 参数计划。Kernel-parameter plan for Tensor contraction.
   *
   * @note 该结构按值传入 kernel，避免为轴映射和 stride 元数据做额外 device allocation 或 H2D copy。
   *       This structure is passed to the kernel by value, avoiding extra device allocation or H2D copies for axis maps and stride metadata.
   */
  struct TensorContractionPlan final
  {
    /**
     * @brief 输出张量秩。Output tensor rank.
     */
    std::size_t output_rank{};

    /**
     * @brief 左侧未缩并轴数量。Number of uncontracted left-hand side axes.
     */
    std::size_t lhs_free_rank{};

    /**
     * @brief 右侧未缩并轴数量。Number of uncontracted right-hand side axes.
     */
    std::size_t rhs_free_rank{};

    /**
     * @brief 缩并轴对数量。Number of contracted axis pairs.
     */
    std::size_t contraction_rank{};

    /**
     * @brief 输出 dense 元素总数。Dense output element count.
     */
    std::size_t output_element_count{};

    /**
     * @brief 每个输出元素需要累加的缩并坐标总数。Number of contracted coordinates accumulated for each output element.
     */
    std::size_t contraction_element_count{};

    /**
     * @brief 左侧未缩并轴 extent。Extents of uncontracted left-hand side axes.
     */
    std::size_t lhs_free_extents[k_tensor_contraction_max_rank]{};

    /**
     * @brief 右侧未缩并轴 extent。Extents of uncontracted right-hand side axes.
     */
    std::size_t rhs_free_extents[k_tensor_contraction_max_rank]{};

    /**
     * @brief 缩并坐标 extent。Extents of contracted coordinates.
     */
    std::size_t contraction_extents[k_tensor_contraction_max_rank]{};

    /**
     * @brief 左侧未缩并轴 row-major stride。Row-major strides of uncontracted left-hand side axes.
     */
    std::size_t lhs_free_strides[k_tensor_contraction_max_rank]{};

    /**
     * @brief 右侧未缩并轴 row-major stride。Row-major strides of uncontracted right-hand side axes.
     */
    std::size_t rhs_free_strides[k_tensor_contraction_max_rank]{};

    /**
     * @brief 左侧缩并轴 row-major stride。Row-major strides of contracted left-hand side axes.
     */
    std::size_t lhs_contraction_strides[k_tensor_contraction_max_rank]{};

    /**
     * @brief 右侧缩并轴 row-major stride。Row-major strides of contracted right-hand side axes.
     */
    std::size_t rhs_contraction_strides[k_tensor_contraction_max_rank]{};
  };

  /**
   * @brief host 侧缩并准备结果。Host-side contraction preparation result.
   */
  struct TensorContractionPreparation final
  {
    /**
     * @brief 可按值传入 kernel 的缩并计划。Contraction plan that can be passed to the kernel by value.
     */
    TensorContractionPlan plan{};

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
   * @brief 检查输入 rank 是否可放入 kernel 参数计划。Check whether an input rank fits in the kernel-parameter plan.
   *
   * @param rank 输入 Tensor rank。Input Tensor rank.
   * @param side 输入侧描述。Input side description.
   */
  auto validate_rank(std::size_t rank, std::string_view side) -> void
  {
    if (rank <= k_tensor_contraction_max_rank)
    {
      return;
    }

    throw std::length_error(std::string(side) + " Tensor rank exceeds Tensor contraction limit");
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
      stride = checked_multiply(stride, extents[axis], "Tensor contraction stride");
    }

    return strides;
  }

  /**
   * @brief 校验轴列表并返回 contracted-axis mask。Validate an axis list and return a contracted-axis mask.
   *
   * @param rank 输入 Tensor rank。Input Tensor rank.
   * @param axes 参与缩并的轴。Axes to contract.
   * @param side 输入侧描述。Input side description.
   * @return 每个输入轴是否参与缩并的 mask。Per-input-axis mask indicating contraction participation.
   */
  [[nodiscard]] auto contracted_axis_mask(std::size_t rank,
                                          const std::vector<cinder::Tensor::size_type> &axes,
                                          std::string_view side) -> std::vector<unsigned char>
  {
    /**
     * @brief contracted-axis mask。Contracted-axis mask.
     */
    std::vector<unsigned char> mask(rank, 0U);

    for (const auto axis : axes)
    {
      if (axis >= rank)
      {
        throw std::invalid_argument(std::string(side) + " Tensor contraction axis is out of range");
      }

      if (mask[axis] != 0U)
      {
        throw std::invalid_argument(std::string(side) + " Tensor contraction axes must be unique");
      }

      mask[axis] = 1U;
    }

    return mask;
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
      count = checked_multiply(count, extent, "Tensor contraction element count");
    }

    return count;
  }

  /**
   * @brief 把左侧未缩并轴写入缩并准备结果。Append uncontracted left-hand side axes to the contraction preparation.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param lhs_mask 左侧 contracted-axis mask。Left-hand side contracted-axis mask.
   * @param lhs_strides 左侧 row-major stride。Left-hand side row-major strides.
   * @param preparation 缩并准备结果。Contraction preparation result.
   */
  auto append_lhs_free_axes(const cinder::Tensor &lhs,
                            const std::vector<unsigned char> &lhs_mask,
                            const std::vector<cinder::Tensor::size_type> &lhs_strides,
                            TensorContractionPreparation &preparation) -> void
  {
    for (std::size_t axis = 0U; axis < lhs.rank(); ++axis)
    {
      if (lhs_mask[axis] != 0U)
      {
        continue;
      }

      /**
       * @brief 当前左侧未缩并轴位置。Current left-hand side free-axis position.
       */
      const auto free_axis = preparation.plan.lhs_free_rank;

      preparation.plan.lhs_free_extents[free_axis] = lhs.shape()[axis];
      preparation.plan.lhs_free_strides[free_axis] = lhs_strides[axis];
      preparation.output_extents.push_back(lhs.shape()[axis]);
      ++preparation.plan.lhs_free_rank;
    }
  }

  /**
   * @brief 把右侧未缩并轴写入缩并准备结果。Append uncontracted right-hand side axes to the contraction preparation.
   *
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param rhs_mask 右侧 contracted-axis mask。Right-hand side contracted-axis mask.
   * @param rhs_strides 右侧 row-major stride。Right-hand side row-major strides.
   * @param preparation 缩并准备结果。Contraction preparation result.
   */
  auto append_rhs_free_axes(const cinder::Tensor &rhs,
                            const std::vector<unsigned char> &rhs_mask,
                            const std::vector<cinder::Tensor::size_type> &rhs_strides,
                            TensorContractionPreparation &preparation) -> void
  {
    for (std::size_t axis = 0U; axis < rhs.rank(); ++axis)
    {
      if (rhs_mask[axis] != 0U)
      {
        continue;
      }

      /**
       * @brief 当前右侧未缩并轴位置。Current right-hand side free-axis position.
       */
      const auto free_axis = preparation.plan.rhs_free_rank;

      preparation.plan.rhs_free_extents[free_axis] = rhs.shape()[axis];
      preparation.plan.rhs_free_strides[free_axis] = rhs_strides[axis];
      preparation.output_extents.push_back(rhs.shape()[axis]);
      ++preparation.plan.rhs_free_rank;
    }
  }

  /**
   * @brief 把缩并轴对写入缩并计划。Append contracted axis pairs to the contraction plan.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param lhs_axes 左侧 Tensor 中参与缩并的轴。Axes in the left-hand side Tensor to contract.
   * @param rhs_axes 右侧 Tensor 中参与缩并的轴。Axes in the right-hand side Tensor to contract.
   * @param lhs_strides 左侧 row-major stride。Left-hand side row-major strides.
   * @param rhs_strides 右侧 row-major stride。Right-hand side row-major strides.
   * @param plan 缩并计划。Contraction plan.
   */
  auto append_contraction_axes(const cinder::Tensor &lhs,
                               const cinder::Tensor &rhs,
                               const std::vector<cinder::Tensor::size_type> &lhs_axes,
                               const std::vector<cinder::Tensor::size_type> &rhs_axes,
                               const std::vector<cinder::Tensor::size_type> &lhs_strides,
                               const std::vector<cinder::Tensor::size_type> &rhs_strides,
                               TensorContractionPlan &plan) -> void
  {
    for (std::size_t axis_pair = 0U; axis_pair < lhs_axes.size(); ++axis_pair)
    {
      /**
       * @brief 左侧缩并轴。Left-hand side contraction axis.
       */
      const auto lhs_axis = lhs_axes[axis_pair];

      /**
       * @brief 右侧缩并轴。Right-hand side contraction axis.
       */
      const auto rhs_axis = rhs_axes[axis_pair];

      if (lhs.shape()[lhs_axis] != rhs.shape()[rhs_axis])
      {
        throw std::invalid_argument("Tensor contraction axis extents must match");
      }

      plan.contraction_extents[axis_pair] = lhs.shape()[lhs_axis];
      plan.lhs_contraction_strides[axis_pair] = lhs_strides[lhs_axis];
      plan.rhs_contraction_strides[axis_pair] = rhs_strides[rhs_axis];
    }
  }

  /**
   * @brief 构造缩并输出 shape 与 kernel 参数计划。Build the contraction output shape and kernel-parameter plan.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param lhs_axes 左侧 Tensor 中参与缩并的轴。Axes in the left-hand side Tensor to contract.
   * @param rhs_axes 右侧 Tensor 中参与缩并的轴。Axes in the right-hand side Tensor to contract.
   * @return 缩并准备结果。Contraction preparation result.
   */
  [[nodiscard]] auto prepare_tensor_contraction(const cinder::Tensor &lhs,
                                                const cinder::Tensor &rhs,
                                                const std::vector<cinder::Tensor::size_type> &lhs_axes,
                                                const std::vector<cinder::Tensor::size_type> &rhs_axes)
      -> TensorContractionPreparation
  {
    if (lhs_axes.size() != rhs_axes.size())
    {
      throw std::invalid_argument("Tensor contraction axis lists must have the same length");
    }

    validate_rank(lhs.rank(), "left-hand side");
    validate_rank(rhs.rank(), "right-hand side");

    /**
     * @brief 左侧 contracted-axis mask。Left-hand side contracted-axis mask.
     */
    const auto lhs_mask = contracted_axis_mask(lhs.rank(), lhs_axes, "left-hand side");

    /**
     * @brief 右侧 contracted-axis mask。Right-hand side contracted-axis mask.
     */
    const auto rhs_mask = contracted_axis_mask(rhs.rank(), rhs_axes, "right-hand side");

    /**
     * @brief 左侧 row-major stride。Left-hand side row-major strides.
     */
    const auto lhs_strides = row_major_strides(lhs.shape());

    /**
     * @brief 右侧 row-major stride。Right-hand side row-major strides.
     */
    const auto rhs_strides = row_major_strides(rhs.shape());

    /**
     * @brief 缩并准备结果。Contraction preparation result.
     */
    TensorContractionPreparation preparation;

    preparation.output_extents.reserve((lhs.rank() - lhs_axes.size()) + (rhs.rank() - rhs_axes.size()));
    preparation.plan.contraction_rank = lhs_axes.size();
    preparation.plan.contraction_element_count = 1U;

    append_contraction_axes(lhs, rhs, lhs_axes, rhs_axes, lhs_strides, rhs_strides, preparation.plan);
    append_lhs_free_axes(lhs, lhs_mask, lhs_strides, preparation);
    append_rhs_free_axes(rhs, rhs_mask, rhs_strides, preparation);

    for (std::size_t axis = 0U; axis < preparation.plan.contraction_rank; ++axis)
    {
      preparation.plan.contraction_element_count =
          checked_multiply(preparation.plan.contraction_element_count,
                           preparation.plan.contraction_extents[axis],
                           "Tensor contraction contracted element count");
    }

    preparation.plan.output_rank = preparation.output_extents.size();
    preparation.plan.output_element_count = checked_element_count(preparation.output_extents);

    return preparation;
  }

  /**
   * @brief 从 row-major 线性索引消费一组轴坐标并计算输入 offset。Consume row-major coordinates from a linear index and compute an input offset.
   *
   * @param linear_index 当前线性索引，会被除去已消费的坐标。Current linear index, divided by consumed coordinates.
   * @param extents 被消费轴的 extent。Extents for consumed axes.
   * @param strides 被消费轴在输入 Tensor 中的 stride。Input Tensor strides for consumed axes.
   * @param rank 被消费轴数量。Number of consumed axes.
   * @return 输入 Tensor 中对应坐标的线性 offset。Linear offset for the corresponding input Tensor coordinates.
   */
  [[nodiscard]] __device__ auto consume_row_major_offset(std::size_t &linear_index,
                                                         const std::size_t *extents,
                                                         const std::size_t *strides,
                                                         std::size_t rank) noexcept -> std::size_t
  {
    /**
     * @brief 累积输入 offset。Accumulated input offset.
     */
    std::size_t offset = 0U;

    for (std::size_t reverse_axis = 0U; reverse_axis < rank; ++reverse_axis)
    {
      /**
       * @brief 当前轴位置。Current axis position.
       */
      const auto axis = rank - reverse_axis - 1U;

      /**
       * @brief 当前轴 extent。Current axis extent.
       */
      const auto extent = extents[axis];

      /**
       * @brief 当前轴坐标。Current axis coordinate.
       */
      const auto coordinate = linear_index % extent;

      linear_index /= extent;
      offset += coordinate * strides[axis];
    }

    return offset;
  }

  /**
   * @brief 根据缩并线性索引累加左右输入 offset。Accumulate left and right input offsets from a contraction linear index.
   *
   * @param contraction_index 缩并坐标线性索引。Linear index over contracted coordinates.
   * @param plan 缩并计划。Contraction plan.
   * @param lhs_offset 左侧输入 offset。Left-hand side input offset.
   * @param rhs_offset 右侧输入 offset。Right-hand side input offset.
   */
  __device__ auto add_contraction_offsets(std::size_t contraction_index,
                                          const TensorContractionPlan &plan,
                                          std::size_t &lhs_offset,
                                          std::size_t &rhs_offset) noexcept -> void
  {
    for (std::size_t reverse_axis = 0U; reverse_axis < plan.contraction_rank; ++reverse_axis)
    {
      /**
       * @brief 当前缩并轴位置。Current contraction axis position.
       */
      const auto axis = plan.contraction_rank - reverse_axis - 1U;

      /**
       * @brief 当前缩并轴 extent。Current contraction axis extent.
       */
      const auto extent = plan.contraction_extents[axis];

      /**
       * @brief 当前缩并坐标。Current contraction coordinate.
       */
      const auto coordinate = contraction_index % extent;

      contraction_index /= extent;
      lhs_offset += coordinate * plan.lhs_contraction_strides[axis];
      rhs_offset += coordinate * plan.rhs_contraction_strides[axis];
    }
  }

  /**
   * @brief Tensor 缩并 kernel。Tensor contraction kernel.
   *
   * @param lhs 左侧只读 packed Tensor buffer。Left-hand side read-only packed Tensor buffer.
   * @param rhs 右侧只读 packed Tensor buffer。Right-hand side read-only packed Tensor buffer.
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param plan 缩并计划。Contraction plan.
   *
   * @note 第 0 个线程写 output header/extents；数据线程只依赖输入 data 与按值传入的 plan，不读取输出 metadata。
   *       Thread 0 writes the output header/extents; data threads depend only on input data and the by-value plan, and do not read output metadata.
   */
  __global__ void tensor_contraction_kernel(cinder::ConstTensorDeviceBuffer lhs,
                                            cinder::ConstTensorDeviceBuffer rhs,
                                            cinder::TensorDeviceBuffer output,
                                            TensorContractionPlan plan)
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

      for (std::size_t axis = 0U; axis < plan.lhs_free_rank; ++axis)
      {
        output.extents()[axis] = plan.lhs_free_extents[axis];
      }

      for (std::size_t axis = 0U; axis < plan.rhs_free_rank; ++axis)
      {
        output.extents()[plan.lhs_free_rank + axis] = plan.rhs_free_extents[axis];
      }
    }

    if (thread_index >= plan.output_element_count)
    {
      return;
    }

    /**
     * @brief 输出线性索引用于拆解自由轴坐标。Output linear index used to decode free-axis coordinates.
     */
    auto free_index = thread_index;

    /**
     * @brief 右侧自由轴贡献的输入 offset。Input offset contributed by right-hand side free axes.
     */
    const auto rhs_base_offset =
        consume_row_major_offset(free_index, plan.rhs_free_extents, plan.rhs_free_strides, plan.rhs_free_rank);

    /**
     * @brief 左侧自由轴贡献的输入 offset。Input offset contributed by left-hand side free axes.
     */
    const auto lhs_base_offset =
        consume_row_major_offset(free_index, plan.lhs_free_extents, plan.lhs_free_strides, plan.lhs_free_rank);

    /**
     * @brief 当前输出元素的缩并累加值。Contraction accumulator for the current output element.
     */
    cinder::Tensor::value_type sum = 0.0F;

    for (std::size_t contraction_index = 0U; contraction_index < plan.contraction_element_count; ++contraction_index)
    {
      /**
       * @brief 当前左侧输入 offset。Current left-hand side input offset.
       */
      auto lhs_offset = lhs_base_offset;

      /**
       * @brief 当前右侧输入 offset。Current right-hand side input offset.
       */
      auto rhs_offset = rhs_base_offset;

      add_contraction_offsets(contraction_index, plan, lhs_offset, rhs_offset);
      sum += lhs.data()[lhs_offset] * rhs.data()[rhs_offset];
    }

    output.data()[thread_index] = sum;
  }

  /**
   * @brief 启动 Tensor 缩并 kernel。Launch the Tensor contraction kernel.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param output 输出 Tensor。Output Tensor.
   * @param plan 缩并计划。Contraction plan.
   */
  auto launch_tensor_contraction_kernel(const cinder::Tensor &lhs,
                                        const cinder::Tensor &rhs,
                                        cinder::Tensor &output,
                                        TensorContractionPlan plan) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(output.size(), 1U);

    /**
     * @brief kernel block 数量。Kernel block count.
     */
    const auto block_count =
        (work_items + static_cast<std::size_t>(k_tensor_contraction_threads_per_block) - 1U) /
        static_cast<std::size_t>(k_tensor_contraction_threads_per_block);

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
    const dim3 block(k_tensor_contraction_threads_per_block);

    tensor_contraction_kernel<<<grid, block>>>(lhs.device_buffer(), rhs.device_buffer(), output.device_buffer(), plan);
    check_cuda(cudaGetLastError(), "launch Tensor contraction kernel");
  }

} // namespace

namespace cinder
{

  /**
   * @brief 计算当前 Tensor 与另一个 Tensor 沿指定轴对的缩并。Compute the contraction of this Tensor and another Tensor along axis pairs.
   *
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param lhs_axes 当前 Tensor 中参与缩并的轴。Axes in this Tensor to contract.
   * @param rhs_axes 右侧 Tensor 中参与缩并的轴。Axes in the right-hand side Tensor to contract.
   * @return 缩并结果 Tensor。Tensor contraction result Tensor.
   */
  auto Tensor::contract(const Tensor &rhs,
                        const std::vector<size_type> &lhs_axes,
                        const std::vector<size_type> &rhs_axes) const -> Tensor
  {
    return cinder::contract(*this, rhs, lhs_axes, rhs_axes);
  }

  /**
   * @brief 计算两个 Tensor 沿指定轴对的缩并。Compute the contraction of two Tensors along axis pairs.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param lhs_axes 左侧 Tensor 中参与缩并的轴。Axes in the left-hand side Tensor to contract.
   * @param rhs_axes 右侧 Tensor 中参与缩并的轴。Axes in the right-hand side Tensor to contract.
   * @return 缩并结果 Tensor。Tensor contraction result Tensor.
   */
  auto contract(const Tensor &lhs,
                const Tensor &rhs,
                const std::vector<Tensor::size_type> &lhs_axes,
                const std::vector<Tensor::size_type> &rhs_axes) -> Tensor
  {
    if (lhs.empty() || rhs.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }

    /**
     * @brief host 侧缩并准备结果。Host-side contraction preparation result.
     */
    const auto preparation = prepare_tensor_contraction(lhs, rhs, lhs_axes, rhs_axes);

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(preparation.output_extents, Tensor::UninitializedTag{});

    launch_tensor_contraction_kernel(lhs, rhs, output, preparation.plan);

    return output;
  }

} // namespace cinder
