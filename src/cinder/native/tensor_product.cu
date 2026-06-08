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
   * @brief Tensor 张量积 kernel 的线程块大小。Thread block size for the Tensor product kernel.
   */
  constexpr unsigned int k_tensor_product_threads_per_block = 256U;

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
   * @brief 张量积 kernel。Tensor product kernel.
   *
   * @param lhs 左侧只读 packed Tensor buffer。Left-hand side read-only packed Tensor buffer.
   * @param rhs 右侧只读 packed Tensor buffer。Right-hand side read-only packed Tensor buffer.
   * @param output 输出可变 packed Tensor buffer。Output mutable packed Tensor buffer.
   * @param output_rank 输出张量秩。Output tensor rank.
   * @param output_element_count 输出元素总数。Output element count.
   *
   * @note 第 0 个线程写 output header/extents；数据线程只依赖输入 metadata 和输出 data 指针，不需要同步等待 output metadata。
   *       Thread 0 writes the output header/extents; data threads depend only on input metadata and the output data pointer, so no sync on output metadata is needed.
   */
  __global__ void tensor_product_kernel(cinder::ConstTensorDeviceBuffer lhs,
                                        cinder::ConstTensorDeviceBuffer rhs,
                                        cinder::TensorDeviceBuffer output,
                                        std::size_t output_rank,
                                        std::size_t output_element_count)
  {
    /**
     * @brief 左侧 packed storage header。Left-hand side packed storage header.
     */
    const auto *lhs_header = lhs.header();

    /**
     * @brief 右侧 packed storage header。Right-hand side packed storage header.
     */
    const auto *rhs_header = rhs.header();

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

      output_header->rank = output_rank;
      output_header->element_count = output_element_count;

      for (std::size_t axis = 0U; axis < lhs_header->rank; ++axis)
      {
        output.extents()[axis] = lhs.extents()[axis];
      }

      for (std::size_t axis = 0U; axis < rhs_header->rank; ++axis)
      {
        output.extents()[lhs_header->rank + axis] = rhs.extents()[axis];
      }
    }

    if (thread_index >= output_element_count)
    {
      return;
    }

    /**
     * @brief 右侧元素总数；非零输出保证 rhs_count 非零。Right-hand side element count; nonzero output guarantees rhs_count is nonzero.
     */
    const auto rhs_count = rhs_header->element_count;

    /**
     * @brief 左侧线性索引。Left-hand side linear index.
     */
    const auto lhs_index = thread_index / rhs_count;

    /**
     * @brief 右侧线性索引。Right-hand side linear index.
     */
    const auto rhs_index = thread_index % rhs_count;

    output.data()[thread_index] = lhs.data()[lhs_index] * rhs.data()[rhs_index];
  }

  /**
   * @brief 启动 Tensor 张量积 kernel。Launch the Tensor product kernel.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @param output 输出 Tensor。Output Tensor.
   */
  auto launch_tensor_product_kernel(const cinder::Tensor &lhs,
                                    const cinder::Tensor &rhs,
                                    cinder::Tensor &output) -> void
  {
    /**
     * @brief 至少包含一个线程以便零元素 Tensor 也能写 metadata。At least one work item so zero-element tensors still write metadata.
     */
    const auto work_items = std::max<std::size_t>(output.size(), 1U);

    /**
     * @brief kernel block 数量。Kernel block count.
     */
    const auto block_count =
        (work_items + static_cast<std::size_t>(k_tensor_product_threads_per_block) - 1U) /
        static_cast<std::size_t>(k_tensor_product_threads_per_block);

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
    const dim3 block(k_tensor_product_threads_per_block);

    tensor_product_kernel<<<grid, block>>>(lhs.device_buffer(),
                                           rhs.device_buffer(),
                                           output.device_buffer(),
                                           output.rank(),
                                           output.size());
    check_cuda(cudaGetLastError(), "launch Tensor product kernel");
  }

  /**
   * @brief 构造张量积输出 shape。Build the Tensor product output shape.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 拼接后的输出 shape。Concatenated output shape.
   */
  [[nodiscard]] auto tensor_product_shape(const cinder::Tensor &lhs,
                                          const cinder::Tensor &rhs) -> std::vector<cinder::Tensor::size_type>
  {
    /**
     * @brief 左侧张量秩。Left-hand side tensor rank.
     */
    const auto lhs_rank = lhs.rank();

    /**
     * @brief 右侧张量秩。Right-hand side tensor rank.
     */
    const auto rhs_rank = rhs.rank();

    if (lhs_rank > (std::numeric_limits<cinder::Tensor::size_type>::max() - rhs_rank))
    {
      throw std::overflow_error("Tensor product rank would overflow");
    }

    /**
     * @brief 输出张量秩。Output tensor rank.
     */
    const auto output_rank = lhs_rank + rhs_rank;

    /**
     * @brief 输出 shape 元信息。Output shape metadata.
     */
    std::vector<cinder::Tensor::size_type> extents;

    extents.reserve(output_rank);
    extents.insert(extents.end(), lhs.shape().begin(), lhs.shape().end());
    extents.insert(extents.end(), rhs.shape().begin(), rhs.shape().end());

    return extents;
  }

} // namespace

namespace cinder
{

  /**
   * @brief 计算当前 Tensor 与另一个 Tensor 的张量积。Compute the tensor product of this Tensor and another Tensor.
   *
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 张量积结果 Tensor。Tensor product result Tensor.
   */
  auto Tensor::tensor_product(const Tensor &rhs) const -> Tensor
  {
    return cinder::tensor_product(*this, rhs);
  }

  /**
   * @brief 计算两个 Tensor 的张量积。Compute the tensor product of two Tensors.
   *
   * @param lhs 左侧 Tensor。Left-hand side Tensor.
   * @param rhs 右侧 Tensor。Right-hand side Tensor.
   * @return 张量积结果 Tensor。Tensor product result Tensor.
   */
  auto tensor_product(const Tensor &lhs, const Tensor &rhs) -> Tensor
  {
    if (lhs.empty() || rhs.empty())
    {
      throw std::invalid_argument("Cannot run Tensor operation on an empty Tensor");
    }

    /**
     * @brief 未初始化输出 Tensor；kernel 会写 output metadata 和 data。Uninitialized output Tensor; the kernel writes output metadata and data.
     */
    Tensor output(tensor_product_shape(lhs, rhs), Tensor::UninitializedTag{});

    launch_tensor_product_kernel(lhs, rhs, output);

    return output;
  }

} // namespace cinder
