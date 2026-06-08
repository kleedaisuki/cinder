#include "cinder/tensor.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

  /**
   * @brief Tensor shape 操作 kernel 的线程块大小。Thread block size for Tensor shape-operation kernels.
   */
  constexpr unsigned int k_tensor_shape_threads_per_block = 256U;

  /**
   * @brief shape 操作支持的最大 rank。Maximum supported rank for shape operations.
   */
  constexpr std::size_t k_tensor_shape_max_rank = 32U;

  /**
   * @brief concat 单次 kernel 支持的最大输入 Tensor 数。Maximum input Tensor count supported by one concat kernel.
   */
  constexpr std::size_t k_tensor_concat_max_inputs = 32U;

  /**
   * @brief broadcast kernel 参数计划。Kernel-parameter plan for broadcast.
   *
   * @note 该结构按值传入 kernel，避免为 output shape 和 input stride 做额外 device allocation 或显式 H2D copy。
   *       This structure is passed to the kernel by value, avoiding extra device allocation or explicit H2D copies for output shape and input strides.
   */
  struct TensorBroadcastPlan final
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
    std::size_t output_extents[k_tensor_shape_max_rank]{};

    /**
     * @brief 每个输出轴对应的输入 row-major stride；广播轴 stride 为 0。Input row-major stride for each output axis; broadcast axes use stride 0.
     */
    std::size_t input_strides[k_tensor_shape_max_rank]{};
  };

  /**
   * @brief slice kernel 参数计划。Kernel-parameter plan for slice.
   *
   * @note 该结构按值传入 kernel，避免为 starts/extents/strides 做额外 device allocation 或显式 H2D copy。
   *       This structure is passed to the kernel by value, avoiding extra device allocation or explicit H2D copies for starts, extents, and strides.
   */
  struct TensorSlicePlan final
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
     * @brief 输入每个轴的切片起点。Per-axis slice starts in the input Tensor.
     */
    std::size_t starts[k_tensor_shape_max_rank]{};

    /**
     * @brief 输出每个轴的 extent。Per-axis output extents.
     */
    std::size_t output_extents[k_tensor_shape_max_rank]{};

    /**
     * @brief 输入每个轴的 row-major stride。Per-axis input row-major strides.
     */
    std::size_t input_strides[k_tensor_shape_max_rank]{};
  };

  /**
   * @brief concat kernel 参数计划。Kernel-parameter plan for concat.
   *
   * @note 输入 buffer 描述符与轴向 offset 都按值传入 kernel，避免构造临时 device 指针表。
   *       Input buffer descriptors and axis offsets are passed by value to the kernel, avoiding a temporary device pointer table.
   */
  struct TensorConcatPlan final
  {
    /**
     * @brief 输入 Tensor 数量。Input Tensor count.
     */
    std::size_t input_count{};

    /**
     * @brief 输出张量秩。Output tensor rank.
     */
    std::size_t output_rank{};

    /**
     * @brief 拼接轴。Concatenation axis.
     */
    std::size_t axis{};

    /**
     * @brief 输出 dense 元素总数。Dense output element count.
     */
    std::size_t output_element_count{};

    /**
     * @brief 输出每个轴的 extent。Per-axis output extents.
     */
    std::size_t output_extents[k_tensor_shape_max_rank]{};

    /**
     * @brief 每个输入在拼接轴上的累积起点；最后一个元素是总 extent。Cumulative starts on the concat axis; the last entry is the total extent.
     */
    std::size_t axis_offsets[k_tensor_concat_max_inputs + 1U]{};

    /**
     * @brief 每个输入 Tensor 的只读 packed buffer。Read-only packed buffer for each input Tensor.
     */
    cinder::ConstTensorDeviceBuffer inputs[k_tensor_concat_max_inputs]{};
  };

  /**
   * @brief host 侧 broadcast 准备结果。Host-side broadcast preparation result.
   */
  struct TensorBroadcastPreparation final
  {
    /**
     * @brief 可按值传入 kernel 的 broadcast 计划。Broadcast plan that can be passed to the kernel.
     */
    TensorBroadcastPlan plan{};

    /**
     * @brief 输出 Tensor 的 host 侧 shape。Host-side shape for the output Tensor.
     */
    std::vector<cinder::Tensor::size_type> output_extents{};
  };

  /**
   * @brief host 侧 slice 准备结果。Host-side slice preparation result.
   */
  struct TensorSlicePreparation final
  {
    /**
     * @brief 可按值传入 kernel 的 slice 计划。Slice plan that can be passed to the kernel.
     */
    TensorSlicePlan plan{};

    /**
     * @brief 输出 Tensor 的 host 侧 shape。Host-side shape for the output Tensor.
     */
    std::vector<cinder::Tensor::size_type> output_extents{};
  };

  /**
   * @brief host 侧 concat 准备结果。Host-side concat preparation result.
   */
  struct TensorConcatPreparation final
  {
    /**
     * @brief 可按值传入 kernel 的 concat 计划。Concat plan that can be passed to the kernel.
     */
    TensorConcatPlan plan{};

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
   * @brief 计算 dense shape 的元素总数。Compute the element count for a dense shape.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   * @param context 错误上下文。Error context.
   * @return 元素总数。Element count.
   */
  [[nodiscard]] auto checked_element_count(const std::vector<cinder::Tensor::size_type> &extents,
                                           std::string_view context) -> cinder::Tensor::size_type
  {
    /**
     * @brief 累积元素总数。Accumulated element count.
     */
    cinder::Tensor::size_type count = 1U;

    for (const auto extent : extents)
    {
      count = checked_multiply(count, extent, context);
    }

    return count;
  }

  /**
   * @brief 校验 Tensor 是否可参与 shape 操作。Validate whether a Tensor can participate in a shape operation.
   *
   * @param input 输入 Tensor。Input Tensor.
   */
  auto validate_shape_input(const cinder::Tensor &input) -> void
  {
    if (input.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }
  }

  /**
   * @brief 校验 rank 是否可放入固定大小 kernel 参数计划。Validate whether a rank fits in the fixed-size kernel-parameter plan.
   *
   * @param rank 张量 rank。Tensor rank.
   * @param operation 操作名称。Operation name.
   */
  auto validate_rank(std::size_t rank, std::string_view operation) -> void
  {
    if (rank <= k_tensor_shape_max_rank)
    {
      return;
    }

    throw std::length_error(std::string("Tensor ") + std::string(operation) + " rank exceeds Tensor shape operation limit");
  }

  /**
   * @brief 计算 dense row-major stride。Compute dense row-major strides.
   *
   * @param extents 每个轴的 extent。Per-axis extents.
   * @param context 错误上下文。Error context.
   * @return 每个轴的 row-major stride。Per-axis row-major strides.
   */
  [[nodiscard]] auto row_major_strides(const std::vector<cinder::Tensor::size_type> &extents,
                                       std::string_view context) -> std::vector<cinder::Tensor::size_type>
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
      stride = checked_multiply(stride, extents[axis], context);
    }

    return strides;
  }

  /**
   * @brief 初始化未初始化 Tensor 的 device metadata。Initialize device metadata for an uninitialized Tensor.
   *
   * @param output 输出 Tensor。Output Tensor.
   *
   * @note 只拷贝 header、extents 和 padding，不写 data 区域，适合 reshape 这种零 kernel 路径。
   *       Only header, extents, and padding are copied, leaving the data region untouched; this suits the zero-kernel reshape path.
   */
  auto initialize_tensor_metadata(cinder::Tensor &output) -> void
  {
    /**
     * @brief 输出 packed buffer 描述符。Output packed buffer descriptor.
     */
    const auto output_buffer = output.device_buffer();

    /**
     * @brief metadata 字节数，等于 data 区域偏移。Metadata byte count, equal to the data-region offset.
     */
    const auto metadata_bytes = output_buffer.data_offset;

    /**
     * @brief host metadata staging buffer。Host metadata staging buffer.
     */
    std::vector<unsigned char> metadata(metadata_bytes, 0U);

    /**
     * @brief packed storage header。Packed storage header.
     */
    const cinder::TensorStorageHeader header{
        output.rank(),
        output.size(),
    };

    std::memcpy(metadata.data(), &header, sizeof(header));

    if (!output.shape().empty())
    {
      /**
       * @brief extent 元数据字节数。Extent metadata byte size.
       */
      const auto extent_bytes = checked_multiply(output.rank(), sizeof(cinder::Tensor::size_type), "Tensor metadata extent size");

      std::memcpy(metadata.data() + sizeof(cinder::TensorStorageHeader), output.shape().data(), extent_bytes);
    }

    check_cuda(cudaMemcpy(output_buffer.storage, metadata.data(), metadata.size(), cudaMemcpyHostToDevice),
               "copy Tensor metadata to device");
  }

  /**
   * @brief 复制 dense data 区域。Copy the dense data region.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param output 输出 Tensor。Output Tensor.
   */
  auto copy_tensor_data(const cinder::Tensor &input, cinder::Tensor &output) -> void
  {
    if (input.size() == 0U)
    {
      return;
    }

    /**
     * @brief 输入 packed buffer 描述符。Input packed buffer descriptor.
     */
    const auto input_buffer = input.device_buffer();

    /**
     * @brief 输出 packed buffer 描述符。Output packed buffer descriptor.
     */
    const auto output_buffer = output.device_buffer();

    /**
     * @brief dense data 字节数。Dense data byte count.
     */
    const auto data_bytes = checked_multiply(input.size(), sizeof(cinder::Tensor::value_type), "Tensor data copy size");

    check_cuda(cudaMemcpy(output_buffer.data(), input_buffer.data(), data_bytes, cudaMemcpyDeviceToDevice),
               "copy Tensor data on device");
  }

  /**
   * @brief 构造 broadcast 输出 shape 与 kernel 参数计划。Build the broadcast output shape and kernel-parameter plan.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param extents 目标 Tensor 每个轴的 extent。Per-axis extents of the target Tensor.
   * @return broadcast 准备结果。Broadcast preparation result.
   */
  [[nodiscard]] auto prepare_tensor_broadcast(const cinder::Tensor &input,
                                              const std::vector<cinder::Tensor::size_type> &extents)
      -> TensorBroadcastPreparation
  {
    validate_rank(extents.size(), "broadcast");

    if (extents.size() < input.rank())
    {
      throw std::invalid_argument("Tensor broadcast target rank must be at least input rank");
    }

    /**
     * @brief 输入 row-major stride。Input row-major strides.
     */
    const auto input_strides = row_major_strides(input.shape(), "Tensor broadcast stride");

    /**
     * @brief broadcast 准备结果。Broadcast preparation result.
     */
    TensorBroadcastPreparation preparation;

    preparation.output_extents = extents;
    preparation.plan.output_rank = extents.size();
    preparation.plan.output_element_count = checked_element_count(extents, "Tensor broadcast element count");

    /**
     * @brief 输出 rank 与输入 rank 的前缀差。Prefix-rank difference between output and input.
     */
    const auto leading_rank = extents.size() - input.rank();

    for (std::size_t output_axis = 0U; output_axis < extents.size(); ++output_axis)
    {
      preparation.plan.output_extents[output_axis] = extents[output_axis];

      if (output_axis < leading_rank)
      {
        preparation.plan.input_strides[output_axis] = 0U;
        continue;
      }

      /**
       * @brief 当前输出轴对应的输入轴。Input axis corresponding to the current output axis.
       */
      const auto input_axis = output_axis - leading_rank;

      /**
       * @brief 当前输入轴 extent。Current input-axis extent.
       */
      const auto input_extent = input.shape()[input_axis];

      /**
       * @brief 当前输出轴 extent。Current output-axis extent.
       */
      const auto output_extent = extents[output_axis];

      if ((input_extent != output_extent) && (input_extent != 1U))
      {
        throw std::invalid_argument("Tensor broadcast input extent is incompatible with output shape");
      }

      preparation.plan.input_strides[output_axis] = (input_extent == 1U) ? 0U : input_strides[input_axis];
    }

    return preparation;
  }

  /**
   * @brief 构造 slice 输出 shape 与 kernel 参数计划。Build the slice output shape and kernel-parameter plan.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param starts 每个轴的起始坐标。Per-axis start coordinates.
   * @param extents 输出切片每个轴的 extent。Per-axis extents of the output slice.
   * @return slice 准备结果。Slice preparation result.
   */
  [[nodiscard]] auto prepare_tensor_slice(const cinder::Tensor &input,
                                          const std::vector<cinder::Tensor::size_type> &starts,
                                          const std::vector<cinder::Tensor::size_type> &extents)
      -> TensorSlicePreparation
  {
    validate_rank(input.rank(), "slice");

    if ((starts.size() != input.rank()) || (extents.size() != input.rank()))
    {
      throw std::invalid_argument("Tensor slice starts and extents must match Tensor rank");
    }

    /**
     * @brief 输入 row-major stride。Input row-major strides.
     */
    const auto input_strides = row_major_strides(input.shape(), "Tensor slice stride");

    /**
     * @brief slice 准备结果。Slice preparation result.
     */
    TensorSlicePreparation preparation;

    preparation.output_extents = extents;
    preparation.plan.output_rank = input.rank();
    preparation.plan.output_element_count = checked_element_count(extents, "Tensor slice element count");

    for (std::size_t axis = 0U; axis < input.rank(); ++axis)
    {
      /**
       * @brief 当前输入轴 extent。Current input-axis extent.
       */
      const auto input_extent = input.shape()[axis];

      if (starts[axis] > input_extent)
      {
        throw std::invalid_argument("Tensor slice start is out of range");
      }

      if (extents[axis] > (input_extent - starts[axis]))
      {
        throw std::invalid_argument("Tensor slice extent is out of range");
      }

      preparation.plan.starts[axis] = starts[axis];
      preparation.plan.output_extents[axis] = extents[axis];
      preparation.plan.input_strides[axis] = input_strides[axis];
    }

    return preparation;
  }

  /**
   * @brief 校验 concat 输入集合的基本约束。Validate basic constraints for concat inputs.
   *
   * @param inputs 非拥有 Tensor 指针列表。Non-owning Tensor pointer list.
   */
  auto validate_concat_inputs(const std::vector<const cinder::Tensor *> &inputs) -> void
  {
    if (inputs.empty())
    {
      throw std::invalid_argument("Tensor concat inputs must not be empty");
    }

    if (inputs.size() > k_tensor_concat_max_inputs)
    {
      throw std::length_error("Tensor concat input count exceeds Tensor concat limit");
    }

    for (const auto *input : inputs)
    {
      if (input == nullptr)
      {
        throw std::invalid_argument("Tensor concat inputs must not contain null Tensor pointers");
      }

      validate_shape_input(*input);
    }
  }

  /**
   * @brief 判断 slice 是否覆盖整个输入 Tensor。Check whether a slice covers the whole input Tensor.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param starts 每个轴的起始坐标。Per-axis start coordinates.
   * @param extents 输出切片每个轴的 extent。Per-axis extents of the output slice.
   * @return 若 slice 是完整 Tensor 副本则为 true。True when the slice is a full Tensor copy.
   */
  [[nodiscard]] auto is_full_tensor_slice(const cinder::Tensor &input,
                                          const std::vector<cinder::Tensor::size_type> &starts,
                                          const std::vector<cinder::Tensor::size_type> &extents) -> bool
  {
    if ((starts.size() != input.rank()) || (extents != input.shape()))
    {
      return false;
    }

    return std::all_of(starts.begin(), starts.end(), [](cinder::Tensor::size_type start) {
      return start == 0U;
    });
  }

  /**
   * @brief 构造 concat 输出 shape 与 kernel 参数计划。Build the concat output shape and kernel-parameter plan.
   *
   * @param inputs 非拥有 Tensor 指针列表。Non-owning Tensor pointer list.
   * @param axis 拼接轴。Concatenation axis.
   * @return concat 准备结果。Concat preparation result.
   */
  [[nodiscard]] auto prepare_tensor_concat(const std::vector<const cinder::Tensor *> &inputs,
                                           cinder::Tensor::size_type axis) -> TensorConcatPreparation
  {
    validate_concat_inputs(inputs);

    /**
     * @brief 第一个输入 Tensor。First input Tensor.
     */
    const auto &first = *inputs.front();

    validate_rank(first.rank(), "concat");

    if (axis >= first.rank())
    {
      throw std::invalid_argument("Tensor concat axis is out of range");
    }

    /**
     * @brief concat 准备结果。Concat preparation result.
     */
    TensorConcatPreparation preparation;

    preparation.output_extents = first.shape();
    preparation.output_extents[axis] = 0U;
    preparation.plan.input_count = inputs.size();
    preparation.plan.output_rank = first.rank();
    preparation.plan.axis = axis;

    for (std::size_t input_index = 0U; input_index < inputs.size(); ++input_index)
    {
      /**
       * @brief 当前输入 Tensor。Current input Tensor.
       */
      const auto &input = *inputs[input_index];

      if (input.rank() != first.rank())
      {
        throw std::invalid_argument("Tensor concat inputs must have the same rank");
      }

      for (std::size_t current_axis = 0U; current_axis < first.rank(); ++current_axis)
      {
        if (current_axis == axis)
        {
          continue;
        }

        if (input.shape()[current_axis] != first.shape()[current_axis])
        {
          throw std::invalid_argument("Tensor concat non-axis extents must match");
        }
      }

      preparation.plan.inputs[input_index] = input.device_buffer();
      preparation.plan.axis_offsets[input_index] = preparation.output_extents[axis];
      preparation.output_extents[axis] =
          checked_add(preparation.output_extents[axis], input.shape()[axis], "Tensor concat axis extent");
    }

    preparation.plan.axis_offsets[inputs.size()] = preparation.output_extents[axis];
    preparation.plan.output_element_count = checked_element_count(preparation.output_extents, "Tensor concat element count");

    for (std::size_t current_axis = 0U; current_axis < first.rank(); ++current_axis)
    {
      preparation.plan.output_extents[current_axis] = preparation.output_extents[current_axis];
    }

    return preparation;
  }

  /**
   * @brief 根据 work item 数计算 1D CUDA block 数。Compute the 1D CUDA block count from a work-item count.
   *
   * @param work_items work item 数。Work-item count.
   * @return CUDA block 数。CUDA block count.
   */
  [[nodiscard]] auto block_count_for(std::size_t work_items) -> unsigned int
  {
    /**
     * @brief 向上取整后的 block 数。Rounded-up block count.
     */
    const auto block_count =
        (work_items + static_cast<std::size_t>(k_tensor_shape_threads_per_block) - 1U) /
        static_cast<std::size_t>(k_tensor_shape_threads_per_block);

    if (block_count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    {
      throw std::length_error("Tensor is too large for a 1D CUDA launch");
    }

    return static_cast<unsigned int>(block_count);
  }

  /**
   * @brief 根据 broadcast 输出线性索引计算输入 linear offset。Compute the input linear offset from a broadcast output linear index.
   *
   * @param output_index 输出线性索引。Output linear index.
   * @param plan broadcast 计划。Broadcast plan.
   * @return 输入 linear offset。Input linear offset.
   */
  [[nodiscard]] __device__ auto broadcast_input_offset_for_output_index(std::size_t output_index,
                                                                        const TensorBroadcastPlan &plan) noexcept
      -> std::size_t
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
       * @brief 当前输出轴 extent。Current output-axis extent.
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
   * @brief 根据 slice 输出线性索引计算输入 linear offset。Compute the input linear offset from a slice output linear index.
   *
   * @param output_index 输出线性索引。Output linear index.
   * @param plan slice 计划。Slice plan.
   * @return 输入 linear offset。Input linear offset.
   */
  [[nodiscard]] __device__ auto slice_input_offset_for_output_index(std::size_t output_index,
                                                                    const TensorSlicePlan &plan) noexcept -> std::size_t
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
       * @brief 当前输出轴 extent。Current output-axis extent.
       */
      const auto output_extent = plan.output_extents[output_axis];

      /**
       * @brief 当前输出坐标。Current output coordinate.
       */
      const auto coordinate = output_index % output_extent;

      output_index /= output_extent;
      input_offset += (coordinate + plan.starts[output_axis]) * plan.input_strides[output_axis];
    }

    return input_offset;
  }

  /**
   * @brief 从 concat 输出坐标选择输入 Tensor。Select the input Tensor from a concat output coordinate.
   *
   * @param axis_coordinate 输出在拼接轴上的坐标。Output coordinate on the concat axis.
   * @param plan concat 计划。Concat plan.
   * @return 输入 Tensor 编号。Input Tensor index.
   */
  [[nodiscard]] __device__ auto concat_input_index_for_axis_coordinate(std::size_t axis_coordinate,
                                                                       const TensorConcatPlan &plan) noexcept
      -> std::size_t
  {
    /**
     * @brief 被选中的输入编号。Selected input index.
     */
    std::size_t input_index = 0U;

    while (((input_index + 1U) < plan.input_count) && (axis_coordinate >= plan.axis_offsets[input_index + 1U]))
    {
      ++input_index;
    }

    return input_index;
  }

  /**
   * @brief 根据 concat 输出坐标计算输入 linear offset。Compute the input linear offset from concat output coordinates.
   *
   * @param coordinates 输出坐标数组。Output coordinate array.
   * @param input 被选中的输入 Tensor buffer。Selected input Tensor buffer.
   * @param input_axis_coordinate 输入在拼接轴上的坐标。Input coordinate on the concat axis.
   * @param plan concat 计划。Concat plan.
   * @return 输入 linear offset。Input linear offset.
   */
  [[nodiscard]] __device__ auto concat_input_offset_for_coordinates(const std::size_t *coordinates,
                                                                    cinder::ConstTensorDeviceBuffer input,
                                                                    std::size_t input_axis_coordinate,
                                                                    const TensorConcatPlan &plan) noexcept
      -> std::size_t
  {
    /**
     * @brief 输入 Tensor 中对应坐标的 linear offset。Linear offset for the corresponding input Tensor coordinate.
     */
    std::size_t input_offset = 0U;

    /**
     * @brief 当前输入 row-major stride。Current input row-major stride.
     */
    std::size_t input_stride = 1U;

    for (std::size_t reverse_axis = 0U; reverse_axis < plan.output_rank; ++reverse_axis)
    {
      /**
       * @brief 当前轴位置。Current axis position.
       */
      const auto axis = plan.output_rank - reverse_axis - 1U;

      /**
       * @brief 当前输入坐标。Current input coordinate.
       */
      const auto coordinate = (axis == plan.axis) ? input_axis_coordinate : coordinates[axis];

      input_offset += coordinate * input_stride;
      input_stride *= input.extents()[axis];
    }

    return input_offset;
  }

  /**
   * @brief broadcast kernel。Broadcast kernel.
   *
   * @param input 输入只读 packed Tensor buffer。Input read-only packed Tensor buffer.
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param plan broadcast 计划。Broadcast plan.
   *
   * @note 第 0 个线程写 output header/extents；数据线程只依赖输入 data 与按值传入的 plan，不读取输出 metadata。
   *       Thread 0 writes output header/extents; data threads depend only on input data and the by-value plan, and do not read output metadata.
   */
  __global__ void tensor_broadcast_kernel(cinder::ConstTensorDeviceBuffer input,
                                          cinder::TensorDeviceBuffer output,
                                          TensorBroadcastPlan plan)
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
    const auto input_offset = broadcast_input_offset_for_output_index(thread_index, plan);

    output.data()[thread_index] = input.data()[input_offset];
  }

  /**
   * @brief slice kernel。Slice kernel.
   *
   * @param input 输入只读 packed Tensor buffer。Input read-only packed Tensor buffer.
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param plan slice 计划。Slice plan.
   *
   * @note 第 0 个线程写 output header/extents；数据线程只依赖输入 data 与按值传入的 plan，不读取输出 metadata。
   *       Thread 0 writes output header/extents; data threads depend only on input data and the by-value plan, and do not read output metadata.
   */
  __global__ void tensor_slice_kernel(cinder::ConstTensorDeviceBuffer input,
                                      cinder::TensorDeviceBuffer output,
                                      TensorSlicePlan plan)
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
    const auto input_offset = slice_input_offset_for_output_index(thread_index, plan);

    output.data()[thread_index] = input.data()[input_offset];
  }

  /**
   * @brief concat kernel。Concat kernel.
   *
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param plan concat 计划。Concat plan.
   *
   * @note 输入 buffer 描述符位于 plan 内；第 0 个线程写 output header/extents，数据线程不读取输出 metadata。
   *       Input buffer descriptors live in the plan; thread 0 writes output header/extents, and data threads do not read output metadata.
   */
  __global__ void tensor_concat_kernel(cinder::TensorDeviceBuffer output, TensorConcatPlan plan)
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
     * @brief 输出坐标数组。Output coordinate array.
     */
    std::size_t coordinates[k_tensor_shape_max_rank]{};

    /**
     * @brief 剩余待展开的输出线性索引。Remaining output linear index to unravel.
     */
    auto remaining_index = thread_index;

    for (std::size_t reverse_axis = 0U; reverse_axis < plan.output_rank; ++reverse_axis)
    {
      /**
       * @brief 当前输出轴位置。Current output axis position.
       */
      const auto axis = plan.output_rank - reverse_axis - 1U;

      coordinates[axis] = remaining_index % plan.output_extents[axis];
      remaining_index /= plan.output_extents[axis];
    }

    /**
     * @brief 输出在拼接轴上的坐标。Output coordinate on the concat axis.
     */
    const auto axis_coordinate = coordinates[plan.axis];

    /**
     * @brief 当前输出元素来自哪个输入 Tensor。Input Tensor index for the current output element.
     */
    const auto input_index = concat_input_index_for_axis_coordinate(axis_coordinate, plan);

    /**
     * @brief 被选中的输入 Tensor buffer。Selected input Tensor buffer.
     */
    const auto input = plan.inputs[input_index];

    /**
     * @brief 输入在拼接轴上的坐标。Input coordinate on the concat axis.
     */
    const auto input_axis_coordinate = axis_coordinate - plan.axis_offsets[input_index];

    /**
     * @brief 当前输出元素对应的输入 linear offset。Input linear offset for the current output element.
     */
    const auto input_offset = concat_input_offset_for_coordinates(coordinates, input, input_axis_coordinate, plan);

    output.data()[thread_index] = input.data()[input_offset];
  }

  /**
   * @brief 启动 broadcast kernel。Launch the broadcast kernel.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param output 输出 Tensor。Output Tensor.
   * @param plan broadcast 计划。Broadcast plan.
   */
  auto launch_tensor_broadcast_kernel(const cinder::Tensor &input,
                                      cinder::Tensor &output,
                                      TensorBroadcastPlan plan) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(output.size(), 1U);

    /**
     * @brief CUDA grid 维度。CUDA grid dimensions.
     */
    const dim3 grid(block_count_for(work_items));

    /**
     * @brief CUDA block 维度。CUDA block dimensions.
     */
    const dim3 block(k_tensor_shape_threads_per_block);

    tensor_broadcast_kernel<<<grid, block>>>(input.device_buffer(), output.device_buffer(), plan);
    check_cuda(cudaGetLastError(), "launch Tensor broadcast kernel");
  }

  /**
   * @brief 启动 slice kernel。Launch the slice kernel.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param output 输出 Tensor。Output Tensor.
   * @param plan slice 计划。Slice plan.
   */
  auto launch_tensor_slice_kernel(const cinder::Tensor &input,
                                  cinder::Tensor &output,
                                  TensorSlicePlan plan) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(output.size(), 1U);

    /**
     * @brief CUDA grid 维度。CUDA grid dimensions.
     */
    const dim3 grid(block_count_for(work_items));

    /**
     * @brief CUDA block 维度。CUDA block dimensions.
     */
    const dim3 block(k_tensor_shape_threads_per_block);

    tensor_slice_kernel<<<grid, block>>>(input.device_buffer(), output.device_buffer(), plan);
    check_cuda(cudaGetLastError(), "launch Tensor slice kernel");
  }

  /**
   * @brief 启动 concat kernel。Launch the concat kernel.
   *
   * @param output 输出 Tensor。Output Tensor.
   * @param plan concat 计划。Concat plan.
   */
  auto launch_tensor_concat_kernel(cinder::Tensor &output, TensorConcatPlan plan) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(output.size(), 1U);

    /**
     * @brief CUDA grid 维度。CUDA grid dimensions.
     */
    const dim3 grid(block_count_for(work_items));

    /**
     * @brief CUDA block 维度。CUDA block dimensions.
     */
    const dim3 block(k_tensor_shape_threads_per_block);

    tensor_concat_kernel<<<grid, block>>>(output.device_buffer(), plan);
    check_cuda(cudaGetLastError(), "launch Tensor concat kernel");
  }

} // namespace

namespace cinder
{

  /**
   * @brief 返回具有新 shape 的 dense Tensor。Return a dense Tensor with a new shape.
   *
   * @param extents 新 Tensor 每个轴的 extent。Per-axis extents of the new Tensor.
   * @return reshape 结果 Tensor。Reshaped result Tensor.
   */
  auto Tensor::reshape(const std::vector<size_type> &extents) const -> Tensor
  {
    return cinder::reshape(*this, extents);
  }

  /**
   * @brief 按广播规则返回具有目标 shape 的 dense Tensor。Return a dense Tensor with the target shape by broadcasting.
   *
   * @param extents 目标 Tensor 每个轴的 extent。Per-axis extents of the target Tensor.
   * @return broadcast 结果 Tensor。Broadcast result Tensor.
   */
  auto Tensor::broadcast(const std::vector<size_type> &extents) const -> Tensor
  {
    return cinder::broadcast(*this, extents);
  }

  /**
   * @brief 返回当前 Tensor 的 dense 切片副本。Return a dense slice copy of the current Tensor.
   *
   * @param starts 每个轴的起始坐标。Per-axis start coordinates.
   * @param extents 输出切片每个轴的 extent。Per-axis extents of the output slice.
   * @return slice 结果 Tensor。Slice result Tensor.
   */
  auto Tensor::slice(const std::vector<size_type> &starts, const std::vector<size_type> &extents) const -> Tensor
  {
    return cinder::slice(*this, starts, extents);
  }

  /**
   * @brief 沿指定轴与另一个 Tensor 拼接。Concatenate this Tensor with another Tensor along an axis.
   *
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param axis 拼接轴。Concatenation axis.
   * @return concat 结果 Tensor。Concatenation result Tensor.
   */
  auto Tensor::concat(const Tensor &rhs, size_type axis) const -> Tensor
  {
    return cinder::concat(*this, rhs, axis);
  }

  /**
   * @brief 返回具有新 shape 的 dense Tensor。Return a dense Tensor with a new shape.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param extents 新 Tensor 每个轴的 extent。Per-axis extents of the new Tensor.
   * @return reshape 结果 Tensor。Reshaped result Tensor.
   */
  auto reshape(const Tensor &input, const std::vector<Tensor::size_type> &extents) -> Tensor
  {
    validate_shape_input(input);

    if (input.shape() == extents)
    {
      return Tensor(input);
    }

    /**
     * @brief 新 shape 的元素总数。Element count of the new shape.
     */
    const auto output_element_count = checked_element_count(extents, "Tensor reshape element count");

    if (output_element_count != input.size())
    {
      throw std::invalid_argument("Tensor reshape element count must match input size");
    }

    /**
     * @brief 未初始化输出 Tensor；metadata 由 host staging 写入，data 由 device-to-device copy 写入。Uninitialized output Tensor; metadata is written by host staging and data by a device-to-device copy.
     */
    Tensor output(extents, Tensor::UninitializedTag{});

    initialize_tensor_metadata(output);
    copy_tensor_data(input, output);

    return output;
  }

  /**
   * @brief 按广播规则返回具有目标 shape 的 dense Tensor。Return a dense Tensor with the target shape by broadcasting.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param extents 目标 Tensor 每个轴的 extent。Per-axis extents of the target Tensor.
   * @return broadcast 结果 Tensor。Broadcast result Tensor.
   */
  auto broadcast(const Tensor &input, const std::vector<Tensor::size_type> &extents) -> Tensor
  {
    validate_shape_input(input);

    if (input.shape() == extents)
    {
      return Tensor(input);
    }

    /**
     * @brief host 侧 broadcast 准备结果。Host-side broadcast preparation result.
     */
    const auto preparation = prepare_tensor_broadcast(input, extents);

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(preparation.output_extents, Tensor::UninitializedTag{});

    launch_tensor_broadcast_kernel(input, output, preparation.plan);

    return output;
  }

  /**
   * @brief 返回输入 Tensor 的 dense 切片副本。Return a dense slice copy of an input Tensor.
   *
   * @param input 输入 Tensor。Input Tensor.
   * @param starts 每个轴的起始坐标。Per-axis start coordinates.
   * @param extents 输出切片每个轴的 extent。Per-axis extents of the output slice.
   * @return slice 结果 Tensor。Slice result Tensor.
   */
  auto slice(const Tensor &input,
             const std::vector<Tensor::size_type> &starts,
             const std::vector<Tensor::size_type> &extents) -> Tensor
  {
    validate_shape_input(input);

    if (is_full_tensor_slice(input, starts, extents))
    {
      return Tensor(input);
    }

    /**
     * @brief host 侧 slice 准备结果。Host-side slice preparation result.
     */
    const auto preparation = prepare_tensor_slice(input, starts, extents);

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(preparation.output_extents, Tensor::UninitializedTag{});

    launch_tensor_slice_kernel(input, output, preparation.plan);

    return output;
  }

  /**
   * @brief 沿指定轴拼接两个 Tensor。Concatenate two Tensors along an axis.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param axis 拼接轴。Concatenation axis.
   * @return concat 结果 Tensor。Concatenation result Tensor.
   */
  auto concat(const Tensor &lhs, const Tensor &rhs, Tensor::size_type axis) -> Tensor
  {
    /**
     * @brief 非拥有二元输入列表。Non-owning binary input list.
     */
    const std::vector<const Tensor *> inputs{&lhs, &rhs};

    return cinder::concat(inputs, axis);
  }

  /**
   * @brief 沿指定轴拼接多个 Tensor。Concatenate multiple Tensors along an axis.
   *
   * @param inputs 非拥有 Tensor 指针列表。Non-owning Tensor pointer list.
   * @param axis 拼接轴。Concatenation axis.
   * @return concat 结果 Tensor。Concatenation result Tensor.
   */
  auto concat(const std::vector<const Tensor *> &inputs, Tensor::size_type axis) -> Tensor
  {
    if (inputs.size() == 1U)
    {
      validate_concat_inputs(inputs);

      /**
       * @brief 唯一输入 Tensor。The only input Tensor.
       */
      const auto &input = *inputs.front();

      if (axis >= input.rank())
      {
        throw std::invalid_argument("Tensor concat axis is out of range");
      }

      return Tensor(input);
    }

    /**
     * @brief host 侧 concat 准备结果。Host-side concat preparation result.
     */
    const auto preparation = prepare_tensor_concat(inputs, axis);

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(preparation.output_extents, Tensor::UninitializedTag{});

    launch_tensor_concat_kernel(output, preparation.plan);

    return output;
  }

} // namespace cinder
