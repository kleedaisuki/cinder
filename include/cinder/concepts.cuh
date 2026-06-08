#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

/**
 * @def CINDER_HOST_DEVICE
 * @brief 标记可在 host 与 device 上调用的轻量函数。Marks a lightweight function callable on host and device.
 *
 * @note 在 CUDA/HIP 编译器下展开为相应的 host/device 属性（attribute）；在普通 C++ 编译器下为空。
 *       Expands to the matching host/device attribute under CUDA/HIP compilers and to nothing under plain C++ compilers.
 */
#ifndef CINDER_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CINDER_HOST_DEVICE __host__ __device__
#else
#define CINDER_HOST_DEVICE
#endif
#endif

namespace cinder
{

  /**
   * @brief 约束张量元素类型为对象类型。Constrain a tensor element type to an object type.
   *
   * @tparam Type 待检查类型（type to check）。The type to check.
   */
  template <typename Type>
  concept TensorElement = std::is_object_v<std::remove_cv_t<Type>>;

  /**
   * @brief 约束索引类型为整数类型。Constrain an index type to an integral type.
   *
   * @tparam Type 待检查类型（type to check）。The type to check.
   */
  template <typename Type>
  concept IntegralIndex = std::is_integral_v<std::remove_cvref_t<Type>>;

  /**
   * @brief 约束映射策略为可复制对象。Constrain a mapping policy to a copy-constructible object.
   *
   * @tparam Type 待检查类型（type to check）。The type to check.
   *
   * @note TensorView 不保存 mapping policy 对象；这里要求 policy 无状态（stateless）且可默认构造，让映射保持纯函数语义。
   *       TensorView does not store a mapping policy object; this requires a stateless default-constructible policy to preserve pure-function semantics.
   */
  template <typename Type>
  concept TensorMapping = std::is_object_v<Type> && std::is_empty_v<Type> && std::is_default_constructible_v<Type>;

  /**
   * @brief 约束类型具备张量形状接口。Constrain a type to provide a tensor shape interface.
   *
   * @tparam Type 待检查类型（type to check）。The type to check.
   */
  template <typename Type>
  concept TensorShape = std::is_object_v<Type> && std::is_copy_constructible_v<Type> &&
                        requires(const Type &shape, std::size_t axis) {
                          { shape.rank() } -> std::convertible_to<std::size_t>;
                          { shape.extent(axis) } -> std::convertible_to<std::size_t>;
                        };

  /**
   * @brief 约束映射策略能把 shape 和索引元组映射为线性偏移。Constrain a mapping policy to map a shape and index tuple to a linear offset.
   *
   * @tparam Mapping 映射策略类型（mapping policy type）。The mapping policy type.
   * @tparam Shape 形状类型（shape type）。The shape type.
   * @tparam Index 索引元组元素类型（index tuple element type）。The element type of the index tuple.
   */
  template <typename Mapping, typename Shape, typename Index>
  concept TensorMappingFor = TensorMapping<Mapping> && TensorShape<Shape> && IntegralIndex<Index> &&
                             requires(const Mapping &mapping, const Shape &shape, const Index *indices) {
                               { mapping(shape, indices) } -> std::convertible_to<std::size_t>;
                             };

} // namespace cinder
