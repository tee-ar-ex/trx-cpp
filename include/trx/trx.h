#ifndef TRX_H // include guard
#define TRX_H

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <functional>
#include <json11.hpp>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <random>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>
#include <zip.h>

#include <mio/mmap.hpp>
#include <mio/shared_mmap.hpp>

#include <trx/detail/exceptions.h>
#include <trx/detail/zip_raii.h>
#include <trx/trx_export.h>

namespace trx {
namespace fs = std::filesystem;
}

using json = json11::Json;

namespace trx {
enum class TrxSaveMode { Auto, Archive, Directory };

struct TrxSaveOptions {
  zip_uint32_t compression_standard = ZIP_CM_STORE;
  TrxSaveMode mode = TrxSaveMode::Auto;
  size_t memory_limit_bytes = 0; // Reserved for future save-path tuning.
  bool overwrite_existing = true;
};

inline json::object _json_object(const json &value) {
  if (value.is_object()) {
    return value.object_items();
  }
  return json::object();
}

inline json _json_set(const json &value, const std::string &key, const json &field) {
  auto obj = _json_object(value);
  obj[key] = field;
  return json(obj);
}
inline std::string path_basename(const std::string &path) {
  if (path.empty())
    return "";
  size_t end = path.find_last_not_of("/\\");
  if (end == std::string::npos)
    return "";
  size_t start = path.find_last_of("/\\", end);
  if (start == std::string::npos)
    return path.substr(0, end + 1);
  return path.substr(start + 1, end - start);
}

inline std::string path_dirname(const std::string &path) {
  if (path.empty())
    return ".";
  size_t end = path.find_last_not_of("/\\");
  if (end == std::string::npos)
    return ".";
  size_t sep = path.find_last_of("/\\", end);
  if (sep == std::string::npos)
    return ".";
  if (sep == 0)
    return std::string(1, path[0]);
#if defined(_WIN32) || defined(_WIN64)
  if (sep == 2 && path.size() >= 3 && path[1] == ':')
    return path.substr(0, 3);
#endif
  return path.substr(0, sep);
}

inline std::string to_utf8_string(const trx::fs::path &path) {
#if defined(__cpp_lib_char8_t)
  const auto u8 = path.u8string();
  return std::string(reinterpret_cast<const char *>(u8.data()), u8.size());
#else
  return path.u8string();
#endif
}

inline zip_t *open_zip_for_read(const std::string &path, int &errorp) {
  zip_t *zf = zip_open(path.c_str(), 0, &errorp);
  if (zf != nullptr) {
    return zf;
  }

  const trx::fs::path fs_path(path);
  const std::string generic = fs_path.generic_string();
  if (generic != path) {
    zf = zip_open(generic.c_str(), 0, &errorp);
    if (zf != nullptr) {
      return zf;
    }
  }

#if defined(_WIN32) || defined(_WIN64)
  const std::string u8 = to_utf8_string(fs_path);
  if (!u8.empty() && u8 != path && u8 != generic) {
    zf = zip_open(u8.c_str(), 0, &errorp);
  }
#endif

  return zf;
}

template <typename T> struct DTypeName {
  static constexpr bool supported = false;
  static constexpr std::string_view value() { return ""; }
};

template <> struct DTypeName<Eigen::half> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "float16"; }
};

template <> struct DTypeName<float> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "float32"; }
};

template <> struct DTypeName<double> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "float64"; }
};

template <> struct DTypeName<int8_t> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "int8"; }
};

template <> struct DTypeName<int16_t> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "int16"; }
};

template <> struct DTypeName<int32_t> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "int32"; }
};

template <> struct DTypeName<int64_t> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "int64"; }
};

template <> struct DTypeName<uint8_t> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "uint8"; }
};

template <> struct DTypeName<uint16_t> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "uint16"; }
};

template <> struct DTypeName<uint32_t> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "uint32"; }
};

template <> struct DTypeName<uint64_t> {
  static constexpr bool supported = true;
  static constexpr std::string_view value() { return "uint64"; }
};

template <typename T> inline std::string dtype_from_scalar() {
  using CleanT = std::remove_cv_t<std::remove_reference_t<T>>;
  static_assert(DTypeName<CleanT>::supported, "Unsupported dtype for TRX scalar.");
  return std::string(DTypeName<CleanT>::value());
}

inline constexpr const char *SEPARATOR = "/";
inline const std::array<std::string_view, 12> dtypes = {"float16",
                                                        "uint8",
                                                        "uint16",
                                                        "ushort",
                                                        "uint32",
                                                        "uint64",
                                                        "int8",
                                                        "int16",
                                                        "int32",
                                                        "int64",
                                                        "float32",
                                                        "float64"};

template <typename DT> struct ArraySequence {
  // Public accessors
  auto &data() { return _data; }
  const auto &data() const { return _data; }
  auto &offsets() { return _offsets; }
  const auto &offsets() const { return _offsets; }
  auto &lengths() { return _lengths; }
  const auto &lengths() const { return _lengths; }

  Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _data;
  Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _offsets;
  Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> _lengths;
  std::vector<uint64_t> _offsets_owned;
  mio::shared_mmap_sink mmap_pos;
  mio::shared_mmap_sink mmap_off;

  ArraySequence() : _data(nullptr, 1, 1), _offsets(nullptr, 1, 1) {}
};

template <typename DT> struct MMappedMatrix {
  // Public accessor
  auto &matrix() { return _matrix; }
  const auto &matrix() const { return _matrix; }

  Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> _matrix;
  mio::shared_mmap_sink mmap;

  MMappedMatrix() : _matrix(nullptr, 1, 1) {}
};

template <typename DT> class TrxFile {
  // Access specifier
public:
  // Data Members
  json header;
  std::unique_ptr<ArraySequence<DT>> streamlines;

  std::map<std::string, std::unique_ptr<MMappedMatrix<uint32_t>>> groups; // vector of indices

  // int or float --check python float precision (singletons)
  std::map<std::string, std::unique_ptr<MMappedMatrix<DT>>> data_per_streamline;
  std::map<std::string, std::unique_ptr<ArraySequence<DT>>> data_per_vertex;
  std::map<std::string, std::map<std::string, std::unique_ptr<MMappedMatrix<DT>>>> data_per_group;
  std::string _uncompressed_folder_handle;
  bool _copy_safe;
  bool _owns_uncompressed_folder = false;

  // Member Functions()
  // TrxFile(int nb_vertices = 0, int nb_streamlines = 0);
  TrxFile(int nb_vertices = 0,
          int nb_streamlines = 0,
          const TrxFile<DT> *init_as = nullptr,
          std::string reference = "");
  ~TrxFile();

  /**
   * @brief After reading the structure of a zip/folder, create a TrxFile
   *
   * @param header A TrxFile header dictionary which will be used for the new TrxFile
   * @param dict_pointer_size A dictionary containing the filenames of all the files within the
   * TrxFile disk file/folder
   * @param root_zip The path of the ZipFile pointer
   * @param root The dirname of the ZipFile pointer
   * @return TrxFile*
   */
  static std::unique_ptr<TrxFile<DT>>
  _create_trx_from_pointer(json header,
                           std::map<std::string, std::tuple<long long, long long>> dict_pointer_size,
                           std::string root_zip = "",
                           std::string root = "");

  template <typename> friend class TrxReader;
  template <typename U> friend std::unique_ptr<TrxFile<U>> load(const std::string &path);

  /**
   * @brief Create a deepcopy of the TrxFile
   *
   * @return TrxFile<DT>* A deepcopied TrxFile of the current object
   */
  std::unique_ptr<TrxFile<DT>> deepcopy();

  /**
   * @brief Remove the ununsed portion of preallocated memmaps
   *
   * @param nb_streamlines The number of streamlines to keep
   * @param nb_vertices The number of vertices to keep
   * @param delete_dpg Remove data_per_group when resizing
   */
  void resize(int nb_streamlines = -1, int nb_vertices = -1, bool delete_dpg = false);

  /**
   * @brief Save a TrxFile
   *
   * @param filename  The path to save the TrxFile to
   * @param compression_standard The compression standard to use, as defined by libzip (default: no compression)
   */
  void save(const std::string &filename, zip_uint32_t compression_standard = ZIP_CM_STORE);
  void save(const std::string &filename, const TrxSaveOptions &options);
  /**
   * @brief Normalize in-memory arrays for deterministic save semantics.
   *
   * This trims trailing preallocated rows when detected, rewrites lengths from
   * offsets, and synchronizes header counts with the actual payload.
   */
  void normalize_for_save();

  void add_dps_from_text(const std::string &name, const std::string &dtype, const std::string &path);
  template <typename T>
  void add_dps_from_vector(const std::string &name, const std::string &dtype, const std::vector<T> &values);
  /**
   * @brief Add per-vertex values as DPV from an in-memory vector.
   *
   * @param name DPV name (used as filename in dpv/).
   * @param dtype Output dtype (float16/float32/float64).
   * @param values Per-vertex values; size must match NB_VERTICES.
   */
  template <typename T>
  void add_dpv_from_vector(const std::string &name, const std::string &dtype, const std::vector<T> &values);
  /**
   * @brief Add a group from a list of streamline indices.
   *
   * @param name Group name (used as filename in groups/).
   * @param indices Streamline indices (uint32) belonging to the group.
   */
  void add_group_from_indices(const std::string &name, const std::vector<uint32_t> &indices);
  /**
   * @brief Set the VOXEL_TO_RASMM affine matrix in the TRX header.
   *
   * This updates header["VOXEL_TO_RASMM"] with the provided 4x4 matrix.
   */
  void set_voxel_to_rasmm(const Eigen::Matrix4f &affine);
  void add_dpv_from_tsf(const std::string &name, const std::string &dtype, const std::string &path);
  void export_dpv_to_tsf(const std::string &name,
                         const std::string &path,
                         const std::string &timestamp,
                         const std::string &dtype = "float32") const;

  /**
   * @brief Cleanup on-disk temporary folder and initialize an empty TrxFile
   *
   */
  void close();
  void _cleanup_temporary_directory();

  size_t num_vertices() const {
    if (streamlines && streamlines->_offsets.size() > 0) {
      const auto last = streamlines->_offsets(streamlines->_offsets.size() - 1);
      return static_cast<size_t>(last);
    }
    if (streamlines && streamlines->_data.size() > 0) {
      return static_cast<size_t>(streamlines->_data.rows());
    }
    if (header["NB_VERTICES"].is_number()) {
      return static_cast<size_t>(header["NB_VERTICES"].int_value());
    }
    return 0;
  }

  size_t num_streamlines() const {
    if (streamlines && streamlines->_offsets.size() > 0) {
      return static_cast<size_t>(streamlines->_offsets.size() - 1);
    }
    if (streamlines && streamlines->_lengths.size() > 0) {
      return static_cast<size_t>(streamlines->_lengths.size());
    }
    if (header["NB_STREAMLINES"].is_number()) {
      return static_cast<size_t>(header["NB_STREAMLINES"].int_value());
    }
    return 0;
  }

  /// Returns an empty TrxFile that inherits this file's header metadata but has
  /// NB_VERTICES and NB_STREAMLINES reset to zero.
  std::unique_ptr<TrxFile<DT>> make_empty_like() const;

  /**
   * @brief Build per-streamline axis-aligned bounding boxes (AABB).
   *
   * Each entry is {min_x, min_y, min_z, max_x, max_y, max_z} in TRX coordinates.
   */
  std::vector<std::array<Eigen::half, 6>> build_streamline_aabbs() const;
  const std::vector<std::array<Eigen::half, 6>> &get_or_build_streamline_aabbs() const;
  void invalidate_aabb_cache() const;

  /**
   * @brief Extract a subset of streamlines intersecting an axis-aligned box.
   *
   * The box is defined by min/max corners in TRX coordinates.
   * Returns a new TrxFile with positions, DPV/DPS, and groups remapped.
   * Optionally builds the AABB cache for the returned TrxFile.
   *
   * If max_streamlines > 0 and more streamlines intersect the box than that
   * limit, a random sample of exactly max_streamlines is returned instead of
   * the full intersection.  rng_seed controls the random draw so results are
   * reproducible.  The returned indices are sorted for efficient I/O.
   */
  std::unique_ptr<TrxFile<DT>>
  query_aabb(const std::array<float, 3> &min_corner,
             const std::array<float, 3> &max_corner,
             const std::vector<std::array<Eigen::half, 6>> *precomputed_aabbs = nullptr,
             bool build_cache_for_result = false,
             size_t max_streamlines = 0,
             uint32_t rng_seed = 42) const;

  /**
   * @brief Extract a subset of streamlines by index.
   *
   * The returned TrxFile remaps positions, DPV/DPS, and groups.
   * Optionally builds the AABB cache for the returned TrxFile.
   */
  std::unique_ptr<TrxFile<DT>>
  subset_streamlines(const std::vector<uint32_t> &streamline_ids,
                     bool build_cache_for_result = false) const;

  const MMappedMatrix<DT> *get_dps(const std::string &name) const;
  const ArraySequence<DT> *get_dpv(const std::string &name) const;
  std::vector<std::array<DT, 3>> get_streamline(size_t streamline_index) const;
  template <typename Fn> void for_each_streamline(Fn &&fn) const;

  /**
   * @brief Add a data-per-group (DPG) field from a flat vector.
   *
   * Values are stored as a matrix of shape (rows, cols). If cols is -1,
   * it is inferred from values.size() / rows.
   */
  template <typename T>
  void add_dpg_from_vector(const std::string &group,
                           const std::string &name,
                           const std::string &dtype,
                           const std::vector<T> &values,
                           int rows = 1,
                           int cols = -1);

  /**
   * @brief Add a data-per-group (DPG) field from an Eigen matrix.
   */
  template <typename Derived>
  void add_dpg_from_matrix(const std::string &group,
                           const std::string &name,
                           const std::string &dtype,
                           const Eigen::MatrixBase<Derived> &matrix);

  /**
   * @brief Get a DPG field or nullptr if missing.
   */
  const MMappedMatrix<DT> *get_dpg(const std::string &group, const std::string &name) const;

  /**
   * @brief List DPG groups present in this TrxFile.
   */
  std::vector<std::string> list_dpg_groups() const;

  /**
   * @brief List DPG fields for a given group.
   */
  std::vector<std::string> list_dpg_fields(const std::string &group) const;

  /**
   * @brief Remove a DPG field from a group.
   */
  void remove_dpg(const std::string &group, const std::string &name);

  /**
   * @brief Remove all DPG fields for a group.
   */
  void remove_dpg_group(const std::string &group);

  /// Load a TrxFile from a zip archive.
  static std::unique_ptr<TrxFile<DT>> load_from_zip(const std::string &path);

  /// Load a TrxFile from an on-disk directory.
  static std::unique_ptr<TrxFile<DT>> load_from_directory(const std::string &path);

  /// Load a TrxFile from either a zip archive or directory.
  static std::unique_ptr<TrxFile<DT>> load(const std::string &path);

  /// Access the backing directory path.
  const std::string &uncompressed_folder_handle() const { return _uncompressed_folder_handle; }
  std::string &uncompressed_folder_handle() { return _uncompressed_folder_handle; }

  /**
   * @brief Fill a TrxFile using another and start indexes (preallocation)
   *
   * @param trx TrxFile to copy data from
   * @param strs_start The start index of the streamline
   * @param pts_start The start index of the point
   * @param nb_strs_to_copy The number of streamlines to copy. If not set will copy all
   * @return std::tuple<int, int> A tuple representing the end of the copied streamlines and end
   * of copied points
   */
  std::tuple<int, int>
  _copy_fixed_arrays_from(TrxFile<DT> *trx, int strs_start = 0, int pts_start = 0, int nb_strs_to_copy = -1);
  int len();

private:
  mutable std::vector<std::array<Eigen::half, 6>> aabb_cache_;
  /**
   * @brief Get the real size of data (ignoring zeros of preallocation)
   *
   * @return std::tuple<int, int> A tuple representing the index of the last streamline and the
   * total length of all the streamlines
   */
  std::tuple<int, int> _get_real_len();
};

namespace detail {
TRX_EXPORT int _sizeof_dtype(const std::string &dtype);
} // namespace detail

struct TypedArray {
  std::string dtype;
  int rows = 0;
  int cols = 0;
  mio::shared_mmap_sink mmap;

  bool empty() const { return rows == 0 || cols == 0 || mmap.data() == nullptr; }
  size_t size() const { return static_cast<size_t>(rows) * static_cast<size_t>(cols); }

  /**
   * @brief View the buffer as a row-major Eigen matrix of type T.
   *
   * This is a zero-copy view over the underlying memory map. The dtype must
   * match the requested T or an exception is thrown.
   */
  template <typename T> Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_matrix() {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data_as<T>(), rows, cols);
  }

  /**
   * @brief View the buffer as a const row-major Eigen matrix of type T.
   *
   * This is a zero-copy view over the underlying memory map. The dtype must
   * match the requested T or an exception is thrown.
   */
  template <typename T>
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_matrix() const {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        data_as<T>(), rows, cols);
  }

  struct ByteView {
    const std::uint8_t *data = nullptr;
    size_t size = 0;
  };

  struct MutableByteView {
    std::uint8_t *data = nullptr;
    size_t size = 0;
  };

  /**
   * @brief Return a read-only byte view of the underlying buffer.
   *
   * The size is computed from dtype * rows * cols. This is useful for
   * interop, hashing, or serialization without exposing raw pointers.
   */
  ByteView to_bytes() const {
    if (empty()) {
      return {};
    }
    return {reinterpret_cast<const std::uint8_t *>(mmap.data()),
            static_cast<size_t>(detail::_sizeof_dtype(dtype)) * size()};
  }

  /**
   * @brief Return a mutable byte view of the underlying buffer.
   *
   * The size is computed from dtype * rows * cols. Use with care: mutating
   * the bytes will mutate the mapped file contents.
   */
  MutableByteView to_bytes_mutable() {
    if (empty()) {
      return {};
    }
    return {reinterpret_cast<std::uint8_t *>(mmap.data()), static_cast<size_t>(detail::_sizeof_dtype(dtype)) * size()};
  }

private:
  const void *data() const { return mmap.data(); }
  void *data() { return mmap.data(); }

  template <typename T> T *data_as() {
    const std::string expected = dtype_from_scalar<T>();
    if (dtype != expected) {
      throw std::invalid_argument("TypedArray dtype mismatch: expected " + expected + " got " + dtype);
    }
    return reinterpret_cast<T *>(mmap.data());
  }

  template <typename T> const T *data_as() const {
    const std::string expected = dtype_from_scalar<T>();
    if (dtype != expected) {
      throw std::invalid_argument("TypedArray dtype mismatch: expected " + expected + " got " + dtype);
    }
    return reinterpret_cast<const T *>(mmap.data());
  }
};

enum class TrxScalarType;

class TRX_EXPORT AnyTrxFile {
public:
  AnyTrxFile() = default;
  ~AnyTrxFile();

  AnyTrxFile(const AnyTrxFile &) = delete;
  AnyTrxFile &operator=(const AnyTrxFile &) = delete;
  AnyTrxFile(AnyTrxFile &&) noexcept = default;
  AnyTrxFile &operator=(AnyTrxFile &&) noexcept = default;

  json header;
  TypedArray positions;
  TypedArray offsets;
  std::vector<uint64_t> offsets_u64;
  std::vector<uint32_t> lengths;

  std::map<std::string, TypedArray> groups;
  std::map<std::string, TypedArray> data_per_streamline;
  std::map<std::string, TypedArray> data_per_vertex;
  std::map<std::string, std::map<std::string, TypedArray>> data_per_group;

  size_t num_vertices() const;
  size_t num_streamlines() const;
  void close();
  void save(const std::string &filename, zip_uint32_t compression_standard = ZIP_CM_STORE);
  void save(const std::string &filename, const TrxSaveOptions &options);

  const TypedArray *get_dps(const std::string &name) const;
  const TypedArray *get_dpv(const std::string &name) const;
  std::vector<std::array<double, 3>> get_streamline(size_t streamline_index) const;

  using PositionsChunkCallback =
      std::function<void(TrxScalarType dtype, const void *data, size_t point_offset, size_t point_count)>;
  using PositionsChunkMutableCallback =
      std::function<void(TrxScalarType dtype, void *data, size_t point_offset, size_t point_count)>;

  void for_each_positions_chunk(size_t chunk_bytes, const PositionsChunkCallback &fn) const;
  void for_each_positions_chunk_mutable(size_t chunk_bytes, const PositionsChunkMutableCallback &fn);

  static AnyTrxFile load(const std::string &path);
  static AnyTrxFile load_from_zip(const std::string &path);
  static AnyTrxFile load_from_directory(const std::string &path);

  /// Access the backing directory path.
  const std::string &backing_directory() const { return _backing_directory; }
  std::string &backing_directory() { return _backing_directory; }

  /// Access the uncompressed folder handle.
  const std::string &uncompressed_folder_handle() const { return _uncompressed_folder_handle; }
  std::string &uncompressed_folder_handle() { return _uncompressed_folder_handle; }

private:
  std::string _uncompressed_folder_handle;
  bool _owns_uncompressed_folder = false;
  std::string _backing_directory;

  static std::string _normalize_dtype(const std::string &dtype);
  static AnyTrxFile
  _create_from_pointer(json header,
                       const std::map<std::string, std::tuple<long long, long long>> &dict_pointer_size,
                       const std::string &root);
  void _cleanup_temporary_directory();
};

inline AnyTrxFile load_any(const std::string &path) { return AnyTrxFile::load(path); }

/**
 * @brief Streaming-friendly TRX builder that spools positions and finalizes to TrxFile.
 *
 * TrxStream allows appending streamlines without knowing totals up front.
 * It writes positions to a temporary binary file, tracks lengths, and can
 * add DPS/DPV/group metadata. Call finalize() to produce a standard TRX.
 */
class TRX_EXPORT TrxStream {
public:
  explicit TrxStream(std::string positions_dtype = "float32");
  ~TrxStream();

  TrxStream(const TrxStream &) = delete;
  TrxStream &operator=(const TrxStream &) = delete;
  TrxStream(TrxStream &&) noexcept = default;
  TrxStream &operator=(TrxStream &&) noexcept = default;

  /**
   * @brief Append a streamline from a flat xyz buffer.
   *
   * @param xyz Pointer to interleaved x,y,z data.
   * @param point_count Number of 3D points in the buffer.
   */
  void push_streamline(const float *xyz, size_t point_count);
  /**
   * @brief Append a streamline from a flat xyz vector.
   */
  void push_streamline(const std::vector<float> &xyz_flat);
  /**
   * @brief Append a streamline from a vector of 3D points.
   */
  void push_streamline(const std::vector<std::array<float, 3>> &points);

  /**
   * @brief Set max in-memory position buffer size (bytes).
   *
   * When set to a non-zero value, positions are buffered in memory and flushed
   * to the temp file once the buffer reaches this size. Useful for reducing
   * small I/O writes on slow disks.
   */
  void set_positions_buffer_max_bytes(std::size_t max_bytes);

  enum class MetadataMode { InMemory, OnDisk };

  /**
   * @brief Control how DPS/DPV/groups are stored during streaming.
   *
   * InMemory keeps metadata in RAM until finalize (default).
   * OnDisk writes metadata to temp files and copies them at finalize.
   */
  TrxStream &set_metadata_mode(MetadataMode mode);

  /**
   * @brief Set max in-memory buffer size for metadata writes (bytes).
   *
   * Applies when MetadataMode::OnDisk. Larger buffers reduce write calls.
   */
  TrxStream &set_metadata_buffer_max_bytes(std::size_t max_bytes);

  /**
   * @brief Set the VOXEL_TO_RASMM affine matrix in the header.
   */
  TrxStream &set_voxel_to_rasmm(const Eigen::Matrix4f &affine);

  /**
   * @brief Set DIMENSIONS in the header.
   */
  TrxStream &set_dimensions(const std::array<uint16_t, 3> &dims);

  /**
   * @brief Add per-streamline values (DPS) from an in-memory vector.
   */
  template <typename T>
  void push_dps_from_vector(const std::string &name, const std::string &dtype, const std::vector<T> &values);
  /**
   * @brief Add per-vertex values (DPV) from an in-memory vector.
   */
  template <typename T>
  void push_dpv_from_vector(const std::string &name, const std::string &dtype, const std::vector<T> &values);
  /**
   * @brief Add a group from a list of streamline indices.
   */
  void push_group_from_indices(const std::string &name, const std::vector<uint32_t> &indices);

  /**
   * @brief Finalize and write a TRX file.
   */
  template <typename DT> void finalize(const std::string &filename, zip_uint32_t compression_standard = ZIP_CM_STORE);
  void finalize(const std::string &filename,
                TrxScalarType output_dtype,
                zip_uint32_t compression_standard = ZIP_CM_STORE);
  void finalize(const std::string &filename, const TrxSaveOptions &options);

  /**
   * @brief Finalize and write a TRX directory (no zip).
   *
   * This method removes any existing directory at the output path before
   * writing. Use this for single-process writes or when you control the
   * entire output location lifecycle.
   *
   * @param directory Path where the uncompressed TRX directory will be created.
   *
   * @throws std::runtime_error if already finalized or if I/O fails.
   *
   * @see finalize_directory_persistent for multiprocess-safe variant.
   */
  void finalize_directory(const std::string &directory);

  /**
   * @brief Finalize and write a TRX directory without removing existing files.
   *
   * This variant is designed for multiprocess workflows where the output
   * directory is pre-created by a parent process. Unlike finalize_directory(),
   * this method does NOT remove the output directory if it exists, making it
   * safe for coordinated parallel writes where multiple processes may check
   * for the directory's existence.
   *
   * @param directory Path where the uncompressed TRX directory will be created.
   *                  If the directory exists, its contents will be overwritten
   *                  but the directory itself will not be removed and recreated.
   *
   * @throws std::runtime_error if already finalized or if I/O fails.
   *
   * @note Typical usage pattern:
   * @code
   *   // Parent process creates shard directories
   *   fs::create_directories("shards/shard_0");
   *   
   *   // Child process writes without removing directory
   *   trx::TrxStream stream("float16");
   *   // ... push streamlines ...
   *   stream.finalize_directory_persistent("shards/shard_0");
   *   std::ofstream("shards/shard_0/SHARD_OK") << "ok\n";
   *   
   *   // Parent waits for SHARD_OK before reading results
   * @endcode
   *
   * @see finalize_directory for single-process variant that ensures clean slate.
   */
  void finalize_directory_persistent(const std::string &directory);

  size_t num_streamlines() const { return lengths_.size(); }
  size_t num_vertices() const { return total_vertices_; }

  json header;

private:
  struct FieldValues {
    std::string dtype;
    std::vector<double> values;
  };

  struct MetadataFile {
    std::string relative_path;
    std::string absolute_path;
  };

  void ensure_positions_stream();
  void flush_positions_buffer();
  void cleanup_tmp();
  void ensure_metadata_dir(const std::string &subdir);
  void finalize_directory_impl(const std::string &directory, bool remove_existing);

  std::string positions_dtype_;
  std::string tmp_dir_;
  std::string positions_path_;
  std::ofstream positions_out_;
  std::vector<float> positions_buffer_float_;
  std::vector<Eigen::half> positions_buffer_half_;
  std::size_t positions_buffer_max_entries_ = 0;
  std::vector<uint32_t> lengths_;
  size_t total_vertices_ = 0;
  bool finalized_ = false;

  std::map<std::string, std::vector<uint32_t>> groups_;
  std::map<std::string, FieldValues> dps_;
  std::map<std::string, FieldValues> dpv_;
  MetadataMode metadata_mode_ = MetadataMode::InMemory;
  std::vector<MetadataFile> metadata_files_;
  std::size_t metadata_buffer_max_bytes_ = 8 * 1024 * 1024;
};

/**
 * @brief Copy header fields from a JSON root (currently unused; candidate for removal).
 *
 * @param[in] root a Json::Value root obtained from reading a header file
 * @return header a header containing the same elements as the original root
 */
TRX_EXPORT json assignHeader(const json &root);

/**
 * This function loads the header json file
 * stored within a Zip archive
 *
 * @param[in] zfolder a pointer to an opened zip archive
 * @param[out] header the JSONCpp root of the header. nullptr on error.
 *
 * */
TRX_EXPORT json load_header(zip_t *zfolder);

/**
 * Load the TRX file stored within a Zip archive.
 *
 * @param[in] path path to Zip archive
 * @param[out] status return 0 if success else 1
 *
 * */
template <typename DT> std::unique_ptr<TrxFile<DT>> load_from_zip(const std::string &path);

/**
 * @brief Load a TrxFile from a folder containing memmaps
 *
 * @tparam DT
 * @param path path of the zipped TrxFile
 * @return TrxFile<DT>* TrxFile representing the read data
 */
template <typename DT> std::unique_ptr<TrxFile<DT>> load_from_directory(const std::string &path);

/**
 * @brief Detect the dtype of the positions array for a TRX path.
 *
 * @param path Path to a TRX zip archive or directory.
 * @return std::string dtype (e.g., float16/float32/float64) or empty if unknown.
 */
TRX_EXPORT std::string detect_positions_dtype(const std::string &path);

enum class TrxScalarType { Float16, Float32, Float64 };

/**
 * @brief Return the canonical string name for a TrxScalarType.
 *
 * @param dtype Scalar type enum value.
 * @return std::string dtype (e.g., float16/float32/float64).
 */
inline std::string scalar_type_name(TrxScalarType dtype) {
  switch (dtype) {
  case TrxScalarType::Float16:
    return "float16";
  case TrxScalarType::Float32:
    return "float32";
  case TrxScalarType::Float64:
    return "float64";
  default:
    return "float32";
  }
}

struct PositionsOutputInfo {
  std::string directory;
  std::string positions_path;
  std::string dtype;
  size_t points = 0;
};

struct PrepareOutputOptions {
  bool overwrite_existing = true;
};

/**
 * @brief Prepare an output directory with copied metadata and offsets.
 *
 * Creates a new TRX directory (no zip) that contains header, offsets, and
 * metadata (groups, dps, dpv, dpg), and returns where the positions file
 * should be written.
 */
TRX_EXPORT PositionsOutputInfo prepare_positions_output(const AnyTrxFile &input,
                                                        const std::string &output_directory,
                                                        const PrepareOutputOptions &options = {});

struct MergeTrxShardsOptions {
  std::vector<std::string> shard_directories;
  std::string output_path;
  zip_uint32_t compression_standard = ZIP_CM_STORE;
  bool output_directory = false;
  bool overwrite_existing = true;
};

TRX_EXPORT void merge_trx_shards(const MergeTrxShardsOptions &options);

/**
 * @brief Detect the positions scalar type for a TRX path.
 *
 * @param path Path to a TRX zip archive or directory.
 * @param fallback Fallback type when dtype is unknown.
 * @return TrxScalarType
 */
TRX_EXPORT TrxScalarType detect_positions_scalar_type(const std::string &path, TrxScalarType fallback = TrxScalarType::Float32);

/**
 * @brief Return true if the TRX path is a directory.
 *
 * @param path Path to a TRX zip archive or directory.
 * @return bool
 */
TRX_EXPORT bool is_trx_directory(const std::string &path);

/**
 * @brief Load a TRX file from either a zip archive or directory.
 *
 * @tparam DT
 * @param path Path to TRX archive or directory
 * @return TrxFile<DT>* TrxFile representing the read data
 */
template <typename DT> std::unique_ptr<TrxFile<DT>> load(const std::string &path);

/**
 * @brief RAII wrapper for loading TRX files from a path.
 *
 * @tparam DT
 */
template <typename DT> class TrxReader {
public:
  explicit TrxReader(const std::string &path);
  ~TrxReader() = default;

  TrxReader(const TrxReader &) = delete;
  TrxReader &operator=(const TrxReader &) = delete;
  TrxReader(TrxReader &&other) noexcept;
  TrxReader &operator=(TrxReader &&other) noexcept;

  TrxFile<DT> *get() const { return trx_.get(); }
  TrxFile<DT> &operator*() const { return *trx_; }
  TrxFile<DT> *operator->() const { return trx_.get(); }

private:
  std::unique_ptr<TrxFile<DT>> trx_;
};

/**
 * @brief Load a TRX reader based on detected positions dtype.
 *
 * @tparam Fn Callable invoked with (TrxReader<T>&, TrxScalarType).
 * @param path Path to TRX archive or directory.
 * @param fn Callable for the loaded reader.
 * @return The return value of fn.
 */
template <typename Fn>
auto with_trx_reader(const std::string &path, Fn &&fn)
    -> decltype(fn(std::declval<TrxReader<float> &>(), TrxScalarType::Float32));

/**
 * Get affine and dimensions from a Nifti or Trk file (Adapted from dipy)
 *
 * @param[in] reference a string pointing to a NIfTI or trk file to be used as reference
 * @param[in] affine 4x4 affine matrix
 * @param[in] dimensions vector of size 3
 *
 * */
void get_reference_info(const std::string &reference,
                        const Eigen::MatrixXf &affine,
                        const Eigen::RowVectorXi &dimensions);

template <typename DT> std::ostream &operator<<(std::ostream &out, const TrxFile<DT> &trx);
// private:

TRX_EXPORT void allocate_file(const std::string &path, std::size_t size);

/**
 * @brief Wrapper to support empty array as memmaps
 *
 * @param filename filename of the file where the empty memmap should be created
 * @param shape shape of memmapped NDArray
 * @param mode file open mode
 * @param dtype datatype of memmapped NDArray
 * @param offset offset of the data within the file
 * @return mio::shared_mmap_sink
 */
// Known limitations: only row-major order supported; shape uses tuple (sufficient for 2D);
// dtype parameter is used only for byte-size computation.
TRX_EXPORT mio::shared_mmap_sink _create_memmap(std::string filename,
                                                 const std::tuple<int, int> &shape,
                                                 const std::string &mode = "r",
                                                 const std::string &dtype = "float32",
                                                 long long offset = 0);

template <typename DT>
std::string _generate_filename_from_data(const Eigen::MatrixBase<DT> &arr, const std::string filename);

/**
 * @brief Create on-disk memmaps of a certain size (preallocation)
 *
 * @param nb_streamlines The number of streamlines that the empty TrxFile will be initialized with
 * @param nb_vertices The number of vertices that the empty TrxFile will be initialized with
 * @param init_as A TrxFile to initialize the empty TrxFile with
 * @return TrxFile<DT> An empty TrxFile preallocated with a certain size
 */
template <typename DT>
std::unique_ptr<TrxFile<DT>>
_initialize_empty_trx(int nb_streamlines, int nb_vertices, const TrxFile<DT> *init_as = nullptr);

template <typename DT>
void ediff1d(Eigen::Matrix<DT, Eigen::Dynamic, 1> &lengths,
             const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> &tmp,
             uint32_t to_end);

/**
 * @brief Save a TrxFile
 *
 * @tparam DT
 * @param trx The TrxFile to save
 * @param filename  The path to save the TrxFile to
 * @param compression_standard The compression standard to use, as defined by libzip (default: no
 * compression)
 */

/**
 * @brief Utils function to zip on-disk memmaps
 *
 * @param directory The path to the on-disk memmap
 * @param filename The path where the zip file should be created
 * @param compression_standard The compression standard to use, as defined by the ZipFile library
 */
TRX_EXPORT void zip_from_folder(zip_t *zf,
                                const std::string &root,
                                const std::string &directory,
                                zip_uint32_t compression_standard = ZIP_CM_STORE,
                                const std::unordered_set<std::string> *skip = nullptr);

TRX_EXPORT std::string get_base(const std::string &delimiter, const std::string &str);
TRX_EXPORT std::string get_ext(const std::string &str);
TRX_EXPORT void populate_fps(const std::string &name, std::map<std::string, std::tuple<long long, long long>> &files_pointer_size);

TRX_EXPORT void copy_dir(const std::string &src, const std::string &dst);
TRX_EXPORT void copy_file(const std::string &src, const std::string &dst);
TRX_EXPORT int rm_dir(const std::string &d);
TRX_EXPORT std::string make_temp_dir(const std::string &prefix);
TRX_EXPORT std::string extract_zip_to_directory(zip_t *zfolder);

TRX_EXPORT std::string rm_root(const std::string &root, const std::string &path);
#ifndef TRX_TPP_STANDALONE
#endif

} // namespace trx

#include <trx/detail/dtype_helpers.h>

namespace trx {
#ifndef TRX_TPP_STANDALONE
#include <trx/trx.tpp>
#endif
} // namespace trx

#endif /* TRX_H */