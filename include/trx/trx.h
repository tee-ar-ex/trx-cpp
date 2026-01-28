#ifndef TRX_H // include guard
#define TRX_H

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <json11.hpp>
#include <limits.h>
#include <math.h>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <utility>
#include <vector>
#include <zip.h>

#include <mio/mmap.hpp>
#include <mio/shared_mmap.hpp>

namespace trx {
namespace fs = std::filesystem;
}

using namespace Eigen;
using json = json11::Json;

namespace trxmmap {
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

template <typename T> struct DTypeName {
  static const char *value() { return "float16"; }
};

template <> struct DTypeName<Eigen::half> {
  static const char *value() { return "float16"; }
};

template <> struct DTypeName<float> {
  static const char *value() { return "float32"; }
};

template <> struct DTypeName<double> {
  static const char *value() { return "float64"; }
};

template <> struct DTypeName<int8_t> {
  static const char *value() { return "int8"; }
};

template <> struct DTypeName<int16_t> {
  static const char *value() { return "int16"; }
};

template <> struct DTypeName<int32_t> {
  static const char *value() { return "int32"; }
};

template <> struct DTypeName<int64_t> {
  static const char *value() { return "int64"; }
};

template <> struct DTypeName<uint8_t> {
  static const char *value() { return "uint8"; }
};

template <> struct DTypeName<uint16_t> {
  static const char *value() { return "uint16"; }
};

template <> struct DTypeName<uint32_t> {
  static const char *value() { return "uint32"; }
};

template <> struct DTypeName<uint64_t> {
  static const char *value() { return "uint64"; }
};

template <> struct DTypeName<bool> {
  static const char *value() { return "bit"; }
};

template <typename T> inline std::string dtype_from_scalar() {
  typedef typename std::remove_cv<typename std::remove_reference<T>::type>::type CleanT;
  return DTypeName<CleanT>::value();
}

const std::string SEPARATOR = "/";
const std::vector<std::string> dtypes({"float16",
                                       "bit",
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
                                       "float64"});

template <typename DT> struct ArraySequence {
  Map<Matrix<DT, Dynamic, Dynamic, RowMajor>> _data;
  Map<Matrix<uint64_t, Dynamic, Dynamic, RowMajor>> _offsets;
  Matrix<uint32_t, Dynamic, 1> _lengths;
  std::vector<uint64_t> _offsets_owned;
  mio::shared_mmap_sink mmap_pos;
  mio::shared_mmap_sink mmap_off;

  ArraySequence() : _data(NULL, 1, 1), _offsets(NULL, 1, 1) {};
};

template <typename DT> struct MMappedMatrix {
  Map<Matrix<DT, Dynamic, Dynamic>> _matrix;
  mio::shared_mmap_sink mmap;

  MMappedMatrix() : _matrix(NULL, 1, 1) {};
};

template <typename DT> class TrxFile {
  // Access specifier
public:
  // Data Members
  json header;
  ArraySequence<DT> *streamlines;

  std::map<std::string, MMappedMatrix<uint32_t> *> groups; // vector of indices

  // int or float --check python floa<t precision (singletons)
  std::map<std::string, MMappedMatrix<DT> *> data_per_streamline;
  std::map<std::string, ArraySequence<DT> *> data_per_vertex;
  std::map<std::string, std::map<std::string, MMappedMatrix<DT> *>> data_per_group;
  std::string _uncompressed_folder_handle;
  bool _copy_safe;
  bool _owns_uncompressed_folder = false;

  // Member Functions()
  // TrxFile(int nb_vertices = 0, int nb_streamlines = 0);
  TrxFile(int nb_vertices = 0, int nb_streamlines = 0, const TrxFile<DT> *init_as = NULL, std::string reference = "");
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
  static TrxFile<DT> *
  _create_trx_from_pointer(json header,
                           std::map<std::string, std::tuple<long long, long long>> dict_pointer_size,
                           std::string root_zip = "",
                           std::string root = "");

  /**
   * @brief Create a deepcopy of the TrxFile
   *
   * @return TrxFile<DT>* A deepcopied TrxFile of the current object
   */
  TrxFile<DT> *deepcopy();

  /**
   * @brief Remove the ununsed portion of preallocated memmaps
   *
   * @param nb_streamlines The number of streamlines to keep
   * @param nb_vertices The number of vertices to keep
   * @param delete_dpg Remove data_per_group when resizing
   */
  void resize(int nb_streamlines = -1, int nb_vertices = -1, bool delete_dpg = false);

  /**
   * @brief Cleanup on-disk temporary folder and initialize an empty TrxFile
   *
   */
  void close();
  void _cleanup_temporary_directory();

private:
  /**
   * @brief Get the real size of data (ignoring zeros of preallocation)
   *
   * @return std::tuple<int, int> A tuple representing the index of the last streamline and the
   * total length of all the streamlines
   */
  std::tuple<int, int> _get_real_len();

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
};

/**
 * TODO: This function might be completely unecessary
 *
 * @param[in] root a Json::Value root obtained from reading a header file with JsonCPP
 * @param[out] header a header containing the same elements as the original root
 * */
json assignHeader(json root);

/**
 * Returns the properly formatted datatype name
 *
 * @param[in] dtype the returned Eigen datatype
 * @param[out] fmt_dtype the formatted datatype
 *
 * */
std::string _get_dtype(std::string dtype);

/**
 * @brief Get the size of the datatype
 *
 * @param dtype the string name of the datatype
 * @return int corresponding to the size of the datatype
 */
int _sizeof_dtype(std::string dtype);

/**
 * Determine whether the extension is a valid extension
 *
 *
 * @param[in] ext a string consisting of the extension starting by a .
 * @param[out] is_valid a boolean denoting whether the extension is valid.
 *
 * */
bool _is_dtype_valid(std::string &ext);

/**
 * This function loads the header json file
 * stored within a Zip archive
 *
 * @param[in] zfolder a pointer to an opened zip archive
 * @param[out] header the JSONCpp root of the header. NULL on error.
 *
 * */
json load_header(zip_t *zfolder);

/**
 * Load the TRX file stored within a Zip archive.
 *
 * @param[in] path path to Zip archive
 * @param[out] status return 0 if success else 1
 *
 * */
template <typename DT> TrxFile<DT> *load_from_zip(std::string path);

/**
 * @brief Load a TrxFile from a folder containing memmaps
 *
 * @tparam DT
 * @param path path of the zipped TrxFile
 * @return TrxFile<DT>* TrxFile representing the read data
 */
template <typename DT> TrxFile<DT> *load_from_directory(std::string path);

/**
 * @brief Detect the dtype of the positions array for a TRX path.
 *
 * @param path Path to a TRX zip archive or directory.
 * @return std::string dtype (e.g., float16/float32/float64) or empty if unknown.
 */
std::string detect_positions_dtype(const std::string &path);

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

/**
 * @brief Detect the positions scalar type for a TRX path.
 *
 * @param path Path to a TRX zip archive or directory.
 * @param fallback Fallback type when dtype is unknown.
 * @return TrxScalarType
 */
TrxScalarType detect_positions_scalar_type(const std::string &path, TrxScalarType fallback = TrxScalarType::Float32);

/**
 * @brief Return true if the TRX path is a directory.
 *
 * @param path Path to a TRX zip archive or directory.
 * @return bool
 */
bool is_trx_directory(const std::string &path);

/**
 * @brief Load a TRX file from either a zip archive or directory.
 *
 * @tparam DT
 * @param path Path to TRX archive or directory
 * @return TrxFile<DT>* TrxFile representing the read data
 */
template <typename DT> TrxFile<DT> *load(std::string path);

/**
 * @brief RAII wrapper for loading TRX files from a path.
 *
 * @tparam DT
 */
template <typename DT> class TrxReader {
public:
  explicit TrxReader(const std::string &path);
  ~TrxReader();

  TrxReader(const TrxReader &) = delete;
  TrxReader &operator=(const TrxReader &) = delete;
  TrxReader(TrxReader &&other) noexcept;
  TrxReader &operator=(TrxReader &&other) noexcept;

  TrxFile<DT> *get() const { return trx_; }
  TrxFile<DT> &operator*() const { return *trx_; }
  TrxFile<DT> *operator->() const { return trx_; }

private:
  TrxFile<DT> *trx_ = nullptr;
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
void get_reference_info(std::string reference, const MatrixXf &affine, const RowVectorXf &dimensions);

template <typename DT> std::ostream &operator<<(std::ostream &out, const TrxFile<DT> &trx);
// private:

void allocate_file(const std::string &path, const int size);

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
// TODO: ADD order??
// TODO: change tuple to vector to support ND arrays?
// TODO: remove data type as that's done outside of this function
mio::shared_mmap_sink _create_memmap(std::string filename,
                                     std::tuple<int, int> &shape,
                                     std::string mode = "r",
                                     std::string dtype = "float32",
                                     long long offset = 0);

template <typename DT> std::string _generate_filename_from_data(const MatrixBase<DT> &arr, const std::string filename);
std::tuple<std::string, int, std::string> _split_ext_with_dimensionality(const std::string filename);

/**
 * @brief Compute the lengths from offsets and header information
 *
 * @tparam DT The datatype (used for the input matrix)
 * @param[in] offsets An array of offsets
 * @param[in] nb_vertices the number of vertices
 * @return Matrix<uint32_t, Dynamic, Dynamic> of lengths
 */
template <typename DT> Matrix<uint32_t, Dynamic, 1> _compute_lengths(const MatrixBase<DT> &offsets, int nb_vertices);

/**
 * @brief Find where data of a contiguous array is actually ending
 *
 * @tparam DT (the datatype)
 * @param x Matrix of values
 * @param l_bound lower bound index for search
 * @param r_bound upper bound index for search
 * @return int index at which array value is 0 (if possible), otherwise returns -1
 */
template <typename DT> int _dichotomic_search(const MatrixBase<DT> &x, int l_bound = -1, int r_bound = -1);

/**
 * @brief Create on-disk memmaps of a certain size (preallocation)
 *
 * @param nb_streamlines The number of streamlines that the empty TrxFile will be initialized with
 * @param nb_vertices The number of vertices that the empty TrxFile will be initialized with
 * @param init_as A TrxFile to initialize the empty TrxFile with
 * @return TrxFile<DT> An empty TrxFile preallocated with a certain size
 */
template <typename DT>
TrxFile<DT> *_initialize_empty_trx(int nb_streamlines, int nb_vertices, const TrxFile<DT> *init_as = NULL);

template <typename DT>
void ediff1d(Matrix<DT, Dynamic, 1> &lengths, const Matrix<DT, Dynamic, Dynamic> &tmp, uint32_t to_end);

/**
 * @brief Save a TrxFile
 *
 * @tparam DT
 * @param trx The TrxFile to save
 * @param filename  The path to save the TrxFile to
 * @param compression_standard The compression standard to use, as defined by libzip (default: no
 * compression)
 */
template <typename DT>
void save(TrxFile<DT> &trx, const std::string filename, zip_uint32_t compression_standard = ZIP_CM_STORE);

/**
 * @brief Utils function to zip on-disk memmaps
 *
 * @param directory The path to the on-disk memmap
 * @param filename The path where the zip file should be created
 * @param compression_standard The compression standard to use, as defined by the ZipFile library
 */
void zip_from_folder(zip_t *zf,
                     const std::string root,
                     const std::string directory,
                     zip_uint32_t compression_standard = ZIP_CM_STORE);

std::string get_base(const std::string &delimiter, const std::string &str);
std::string get_ext(const std::string &str);
void populate_fps(const char *name, std::map<std::string, std::tuple<long long, long long>> &files_pointer_size);

void copy_dir(const char *src, const char *dst);
void copy_file(const char *src, const char *dst);
int rm_dir(const char *d);
std::string make_temp_dir(const std::string &prefix);
std::string extract_zip_to_directory(zip_t *zfolder);

std::string rm_root(std::string root, const std::string path);
#include <trx/trx.tpp>

} // namespace trxmmap

#endif /* TRX_H */