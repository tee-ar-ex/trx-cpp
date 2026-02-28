#ifndef TRX_DETAIL_DTYPE_HELPERS_H
#define TRX_DETAIL_DTYPE_HELPERS_H

#include <Eigen/Core>

#include <new>
#include <string>
#include <tuple>

#include <trx/trx_export.h>

namespace trx {
namespace detail {

// Central helper that performs the ONE placement-new + reinterpret_cast needed
// to (re)bind an Eigen::Map to a new memory region. All other code should call
// this instead of scattering placement-new across the codebase.
//
// MapType must be an Eigen::Map<Matrix<...>> type.
template <typename MapType>
inline void remap(MapType &map, void *data, int rows, int cols) {
  using Scalar = typename MapType::Scalar;
  new (&map) MapType(reinterpret_cast<Scalar *>(data), rows, cols); // NOLINT
}

// Overload for const data pointers (read-only maps).
template <typename MapType>
inline void remap(MapType &map, const void *data, int rows, int cols) {
  using Scalar = typename MapType::Scalar;
  new (&map) MapType(const_cast<Scalar *>(reinterpret_cast<const Scalar *>(data)), rows, cols); // NOLINT
}

// Convenience overloads that unpack a (rows, cols) shape tuple.
template <typename MapType>
inline void remap(MapType &map, void *data, const std::tuple<int, int> &shape) {
  remap(map, data, std::get<0>(shape), std::get<1>(shape));
}
template <typename MapType>
inline void remap(MapType &map, const void *data, const std::tuple<int, int> &shape) {
  remap(map, data, std::get<0>(shape), std::get<1>(shape));
}

TRX_EXPORT int _sizeof_dtype(const std::string &dtype);
TRX_EXPORT std::string _get_dtype(const std::string &dtype);
TRX_EXPORT bool _is_dtype_valid(const std::string &ext);
TRX_EXPORT std::tuple<std::string, int, std::string> _split_ext_with_dimensionality(const std::string &filename);

template <typename DT>
inline Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> _compute_lengths(const Eigen::MatrixBase<DT> &offsets,
                                                                   int nb_vertices) {
  static_cast<void>(nb_vertices);
  if (offsets.size() > 1) {
    const auto casted = offsets.template cast<uint64_t>();
    const Eigen::Index len = offsets.size() - 1;
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> lengths(len);
    for (Eigen::Index i = 0; i < len; ++i) {
      lengths(i) = static_cast<uint32_t>(casted(i + 1) - casted(i));
    }
    return lengths;
  }
  // If offsets are empty or only contain the sentinel, there are zero streamlines.
  return Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>(0);
}

template <typename DT>
inline int _dichotomic_search(const Eigen::MatrixBase<DT> &x, int l_bound = -1, int r_bound = -1) {
  if (l_bound == -1 && r_bound == -1) {
    l_bound = 0;
    r_bound = static_cast<int>(x.size()) - 1;
  }

  if (l_bound == r_bound) {
    int val = -1;
    if (x(l_bound) != 0) {
      val = l_bound;
    }
    return val;
  }

  int mid_bound = (l_bound + r_bound + 1) / 2;

  if (x(mid_bound) == 0) {
    return _dichotomic_search(x, l_bound, mid_bound - 1);
  }
  return _dichotomic_search(x, mid_bound, r_bound);
}

} // namespace detail
} // namespace trx

#endif // TRX_DETAIL_DTYPE_HELPERS_H
