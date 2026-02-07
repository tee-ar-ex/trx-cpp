#include <Eigen/Core> // NOLINT(misc-include-cleaner)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <tuple>
#include <vector>
#include <zip.h>
#include <zipconf.h>

#include <mio/shared_mmap.hpp>
#include <trx/trx.h>

// #define ZIP_DD_SIG 0x08074b50
// #define ZIP_CD_SIG 0x06054b50
using namespace Eigen;
using std::array;
using std::ifstream;
using std::map;
using std::mt19937_64;
using std::ofstream;
using std::string;
using std::tuple;
using std::uniform_int_distribution;
using std::vector;

namespace trx {
namespace {
inline int sys_error() { return errno; }

inline string get_env(const string &name) {
  const auto *value = std::getenv(name.c_str()); // NOLINT(concurrency-mt-unsafe)
  return value == nullptr ? string() : string(value);
}

std::string normalize_slashes(std::string path) {
  std::replace(path.begin(), path.end(), '\\', '/');
  return path;
}

bool parse_positions_dtype(const std::string &filename, std::string &out_dtype) {
  const std::string normalized = normalize_slashes(filename);
  try {
    const auto tuple = trx::detail::_split_ext_with_dimensionality(normalized);
    const std::string &base = std::get<0>(tuple);
    if (base == "positions") {
      out_dtype = std::get<2>(tuple);
      return true;
    }
  } catch (const std::exception &) {
    // Ignore malformed entries and continue searching.
    return false;
  }
  return false;
}

bool is_path_within(const trx::fs::path &child, const trx::fs::path &parent) {
  const std::string parent_str = parent.lexically_normal().string();
  const std::string child_str = child.lexically_normal().string();
  if (parent_str.empty()) {
    return false;
  }
  if (child_str.size() < parent_str.size()) {
    return false;
  }
  if (child_str.compare(0, parent_str.size(), parent_str) != 0) {
    return false;
  }
  if (child_str.size() == parent_str.size()) {
    return true;
  }
  const char next = child_str[parent_str.size()];
  return next == '/' || next == '\\';
}

TypedArray make_typed_array(const std::string &filename, int rows, int cols, const std::string &dtype) {
  TypedArray array;
  array.dtype = dtype;
  array.rows = rows;
  array.cols = cols;
  if (rows > 0 && cols > 0) {
    array.mmap = _create_memmap(filename, std::make_tuple(rows, cols), "r+", dtype, 0);
  }
  return array;
}
} // namespace

std::string detect_positions_dtype(const std::string &path) {
  const trx::fs::path input(path);
  if (!trx::fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + path);
  }

  std::error_code ec;
  if (trx::fs::is_directory(input, ec) && !ec) {
    std::map<std::string, std::tuple<long long, long long>> files;
    trx::populate_fps(path, files);
    for (const auto &kv : files) {
      std::string dtype;
      if (parse_positions_dtype(kv.first, dtype)) {
        return dtype;
      }
    }
    return "";
  }

  int err = 0;
  zip_t *zf = open_zip_for_read(path, err);
  if (zf == nullptr) {
    throw std::runtime_error("Could not open zip file: " + path);
  }
  std::string dtype;
  const zip_int64_t count = zip_get_num_entries(zf, 0);
  for (zip_int64_t i = 0; i < count; ++i) {
    const auto *name = zip_get_name(zf, i, 0);
    if (name == nullptr) {
      continue;
    }
    if (parse_positions_dtype(name, dtype)) {
      break;
    }
  }
  zip_close(zf);
  return dtype;
}

TrxScalarType detect_positions_scalar_type(const std::string &path, TrxScalarType fallback) {
  const std::string dtype = detect_positions_dtype(path);
  if (dtype == "float16") {
    return TrxScalarType::Float16;
  }
  if (dtype == "float64") {
    return TrxScalarType::Float64;
  }
  if (dtype == "float32") {
    return TrxScalarType::Float32;
  }
  return fallback;
}

bool is_trx_directory(const std::string &path) {
  const trx::fs::path input(path);
  if (!trx::fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + path);
  }
  std::error_code ec;
  return trx::fs::is_directory(input, ec) && !ec;
}

AnyTrxFile::~AnyTrxFile() { _cleanup_temporary_directory(); }

std::string AnyTrxFile::_normalize_dtype(const std::string &dtype) {
  if (dtype == "bool") {
    return "bit";
  }
  if (dtype == "ushort") {
    return "uint16";
  }
  return dtype;
}

size_t AnyTrxFile::num_vertices() const {
  if (!positions.empty()) {
    return static_cast<size_t>(positions.rows);
  }
  if (header["NB_VERTICES"].is_number()) {
    return static_cast<size_t>(header["NB_VERTICES"].int_value());
  }
  return 0;
}

size_t AnyTrxFile::num_streamlines() const {
  if (!lengths.empty()) {
    return lengths.size();
  }
  if (header["NB_STREAMLINES"].is_number()) {
    return static_cast<size_t>(header["NB_STREAMLINES"].int_value());
  }
  return 0;
}

void AnyTrxFile::close() {
  _cleanup_temporary_directory();
  positions = TypedArray();
  offsets = TypedArray();
  offsets_u64.clear();
  lengths.clear();
  groups.clear();
  data_per_streamline.clear();
  data_per_vertex.clear();
  data_per_group.clear();
  _uncompressed_folder_handle.clear();
  _owns_uncompressed_folder = false;

  std::vector<std::vector<float>> affine(4, std::vector<float>(4, 0.0f));
  for (int i = 0; i < 4; i++) {
    affine[i][i] = 1.0f;
  }
  std::vector<uint16_t> dimensions{1, 1, 1};
  json::object header_obj;
  header_obj["VOXEL_TO_RASMM"] = affine;
  header_obj["DIMENSIONS"] = dimensions;
  header_obj["NB_VERTICES"] = 0;
  header_obj["NB_STREAMLINES"] = 0;
  header = json(header_obj);
}

void AnyTrxFile::_cleanup_temporary_directory() {
  if (_owns_uncompressed_folder && !_uncompressed_folder_handle.empty()) {
    if (rm_dir(_uncompressed_folder_handle) != 0) {
    }
    _uncompressed_folder_handle.clear();
    _owns_uncompressed_folder = false;
  }
}

AnyTrxFile AnyTrxFile::load(const std::string &path) {
  trx::fs::path input(path);
  if (!trx::fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + path);
  }
  std::error_code ec;
  if (trx::fs::is_directory(input, ec) && !ec) {
    return AnyTrxFile::load_from_directory(path);
  }
  return AnyTrxFile::load_from_zip(path);
}

AnyTrxFile AnyTrxFile::load_from_zip(const std::string &filename) {
  int errorp = 0;
  zip_t *zf = open_zip_for_read(filename, errorp);
  if (zf == nullptr) {
    throw std::runtime_error("Could not open zip file: " + filename);
  }

  std::string temp_dir = extract_zip_to_directory(zf);
  zip_close(zf);

  auto trx = AnyTrxFile::load_from_directory(temp_dir);
  trx._uncompressed_folder_handle = temp_dir;
  trx._owns_uncompressed_folder = true;
  return trx;
}

AnyTrxFile AnyTrxFile::load_from_directory(const std::string &path) {
  std::string directory = path;
  {
    std::error_code ec;
    trx::fs::path resolved = trx::fs::weakly_canonical(trx::fs::path(path), ec);
    if (!ec) {
      directory = resolved.string();
    }
  }

  std::string header_name = directory + SEPARATOR + "header.json";
  std::ifstream header_file(header_name);
  if (!header_file.is_open()) {
    throw std::runtime_error("Failed to open header.json at: " + header_name);
  }
  std::string jstream((std::istreambuf_iterator<char>(header_file)), std::istreambuf_iterator<char>());
  header_file.close();
  std::string err;
  json header = json::parse(jstream, err);
  if (!err.empty()) {
    throw std::runtime_error("Failed to parse header.json: " + err);
  }

  std::map<std::string, std::tuple<long long, long long>> files_pointer_size;
  populate_fps(directory, files_pointer_size);

  auto trx = AnyTrxFile::_create_from_pointer(header, files_pointer_size, directory);
  trx._backing_directory = directory;
  return trx;
}

AnyTrxFile
AnyTrxFile::_create_from_pointer(json header,
                                 const std::map<std::string, std::tuple<long long, long long>> &dict_pointer_size,
                                 const std::string &root) {
  AnyTrxFile trx;
  trx.header = header;

  if (!header["NB_VERTICES"].is_number() || !header["NB_STREAMLINES"].is_number()) {
    throw std::invalid_argument("Missing NB_VERTICES or NB_STREAMLINES in header.json");
  }

  const int nb_vertices = header["NB_VERTICES"].int_value();
  const int nb_streamlines = header["NB_STREAMLINES"].int_value();

  for (auto x = dict_pointer_size.rbegin(); x != dict_pointer_size.rend(); ++x) {
    const std::string elem_filename = x->first;

    trx::fs::path elem_path(elem_filename);
    trx::fs::path folder_path = elem_path.parent_path();
    std::string folder;
    if (!root.empty()) {
      trx::fs::path rel_path = elem_path.lexically_relative(trx::fs::path(root));
      std::string rel_str = rel_path.string();
      if (!rel_str.empty() && rel_str.rfind("..", 0) != 0) {
        folder = rel_path.parent_path().string();
      } else {
        folder = folder_path.string();
      }
    } else {
      folder = folder_path.string();
    }
    if (folder == ".") {
      folder.clear();
    }

    std::tuple<std::string, int, std::string> base_tuple = trx::detail::_split_ext_with_dimensionality(elem_filename);
    std::string base(std::get<0>(base_tuple));
    int dim = std::get<1>(base_tuple);
    std::string ext(std::get<2>(base_tuple));

    ext = _normalize_dtype(ext);

    long long size = std::get<1>(x->second);

    if (base == "positions" && (folder.empty() || folder == ".")) {
      if (size != static_cast<long long>(nb_vertices) * 3 || dim != 3) {
        throw std::invalid_argument("Wrong positions size/dimensionality");
      }
      if (ext != "float16" && ext != "float32" && ext != "float64") {
        throw std::invalid_argument("Unsupported positions dtype: " + ext);
      }
      trx.positions = make_typed_array(elem_filename, nb_vertices, 3, ext);
    } else if (base == "offsets" && (folder.empty() || folder == ".")) {
      if (size != static_cast<long long>(nb_streamlines) + 1 || dim != 1) {
        throw std::invalid_argument("Wrong offsets size/dimensionality");
      }
      if (ext != "uint32" && ext != "uint64") {
        throw std::invalid_argument("Unsupported offsets dtype: " + ext);
      }
      trx.offsets = make_typed_array(elem_filename, nb_streamlines + 1, 1, ext);
    } else if (folder == "dps") {
      const int nb_scalar = nb_streamlines > 0 ? static_cast<int>(size / nb_streamlines) : 0;
      if (nb_streamlines == 0 || size % nb_streamlines != 0 || nb_scalar != dim) {
        throw std::invalid_argument("Wrong dps size/dimensionality");
      }
      trx.data_per_streamline.emplace(base, make_typed_array(elem_filename, nb_streamlines, nb_scalar, ext));
    } else if (folder == "dpv") {
      const int nb_scalar = nb_vertices > 0 ? static_cast<int>(size / nb_vertices) : 0;
      if (nb_vertices == 0 || size % nb_vertices != 0 || nb_scalar != dim) {
        throw std::invalid_argument("Wrong dpv size/dimensionality");
      }
      trx.data_per_vertex.emplace(base, make_typed_array(elem_filename, nb_vertices, nb_scalar, ext));
    } else if (folder.rfind("dpg", 0) == 0) {
      if (size != dim) {
        throw std::invalid_argument("Wrong dpg size/dimensionality");
      }
      std::string data_name = path_basename(base);
      std::string sub_folder = path_basename(folder);
      trx.data_per_group[sub_folder].emplace(data_name,
                                             make_typed_array(elem_filename, 1, static_cast<int>(size), ext));
    } else if (folder == "groups") {
      if (dim != 1) {
        throw std::invalid_argument("Wrong group dimensionality");
      }
      if (ext != "uint32") {
        throw std::invalid_argument("Unsupported group dtype: " + ext);
      }
      trx.groups.emplace(base, make_typed_array(elem_filename, static_cast<int>(size), 1, ext));
    } else {
      throw std::invalid_argument("Entry is not part of a valid TRX structure: " + elem_filename);
    }
  }

  if (trx.positions.empty() || trx.offsets.empty()) {
    throw std::invalid_argument("Missing essential data.");
  }

  const size_t offsets_count = trx.offsets.size();
  if (offsets_count > 0) {
    trx.offsets_u64.resize(offsets_count);
    const auto bytes = trx.offsets.to_bytes();
    if (trx.offsets.dtype == "uint64") {
      const auto *src = reinterpret_cast<const uint64_t *>(bytes.data);
      for (size_t i = 0; i < offsets_count; ++i) {
        trx.offsets_u64[i] = src[i];
      }
    } else if (trx.offsets.dtype == "uint32") {
      const auto *src = reinterpret_cast<const uint32_t *>(bytes.data);
      for (size_t i = 0; i < offsets_count; ++i) {
        trx.offsets_u64[i] = static_cast<uint64_t>(src[i]);
      }
    } else {
      throw std::invalid_argument("Unsupported offsets datatype: " + trx.offsets.dtype);
    }
  }

  if (offsets_count > 1) {
    trx.lengths.resize(offsets_count - 1);
    for (size_t i = 0; i + 1 < offsets_count; ++i) {
      const uint64_t diff = trx.offsets_u64[i + 1] - trx.offsets_u64[i];
      if (diff > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Offset difference exceeds uint32 range");
      }
      trx.lengths[i] = static_cast<uint32_t>(diff);
    }
  }

  return trx;
}

void AnyTrxFile::save(const std::string &filename, zip_uint32_t compression_standard) {
  const std::string ext = get_ext(filename);
  if (ext.size() > 0 && (ext != "zip" && ext != "trx")) {
    throw std::invalid_argument("Unsupported extension." + ext);
  }

  if (offsets.empty()) {
    throw std::runtime_error("Cannot save TRX without offsets data");
  }
  if (offsets_u64.empty()) {
    throw std::runtime_error("Cannot save TRX without decoded offsets");
  }
  if (header["NB_STREAMLINES"].is_number()) {
    const auto nb_streamlines = static_cast<size_t>(header["NB_STREAMLINES"].int_value());
    if (offsets_u64.size() != nb_streamlines + 1) {
      throw std::runtime_error("TRX offsets size does not match NB_STREAMLINES");
    }
  }
  if (header["NB_VERTICES"].is_number()) {
    const auto nb_vertices = static_cast<uint64_t>(header["NB_VERTICES"].int_value());
    const auto last = offsets_u64.back();
    if (last != nb_vertices) {
      throw std::runtime_error("TRX offsets sentinel does not match NB_VERTICES");
    }
  }
  for (size_t i = 1; i < offsets_u64.size(); ++i) {
    if (offsets_u64[i] < offsets_u64[i - 1]) {
      throw std::runtime_error("TRX offsets must be monotonically increasing");
    }
  }
  if (!positions.empty()) {
    const auto last = offsets_u64.back();
    if (last != static_cast<uint64_t>(positions.rows)) {
      throw std::runtime_error("TRX positions row count does not match offsets sentinel");
    }
  }

  const std::string source_dir =
      !_uncompressed_folder_handle.empty() ? _uncompressed_folder_handle : _backing_directory;
  if (source_dir.empty()) {
    throw std::runtime_error("TRX file has no backing directory to save from");
  }

  std::string tmp_dir = make_temp_dir("trx_runtime");
  copy_dir(source_dir, tmp_dir);

  {
    const trx::fs::path header_path = trx::fs::path(tmp_dir) / "header.json";
    std::ofstream out_json(header_path);
    if (!out_json.is_open()) {
      throw std::runtime_error("Failed to write header.json to: " + header_path.string());
    }
    out_json << header.dump() << std::endl;
  }

  if (ext.size() > 0 && (ext == "zip" || ext == "trx")) {
    int errorp;
    zip_t *zf;
    if ((zf = zip_open(filename.c_str(), ZIP_CREATE + ZIP_TRUNCATE, &errorp)) == nullptr) {
      rm_dir(tmp_dir);
      throw std::runtime_error("Could not open archive " + filename + ": " + strerror(errorp));
    }
    zip_from_folder(zf, tmp_dir, tmp_dir, compression_standard);
    if (zip_close(zf) != 0) {
      rm_dir(tmp_dir);
      throw std::runtime_error("Unable to close archive " + filename + ": " + zip_strerror(zf));
    }
  } else {
    std::error_code ec;
    if (trx::fs::exists(filename, ec) && trx::fs::is_directory(filename, ec)) {
      if (rm_dir(filename) != 0) {
        rm_dir(tmp_dir);
        throw std::runtime_error("Could not remove existing directory " + filename);
      }
    }
    trx::fs::path dest_path(filename);
    if (dest_path.has_parent_path()) {
      std::error_code parent_ec;
      trx::fs::create_directories(dest_path.parent_path(), parent_ec);
      if (parent_ec) {
        rm_dir(tmp_dir);
        throw std::runtime_error("Could not create output parent directory: " + dest_path.parent_path().string());
      }
    }
    copy_dir(tmp_dir, filename);
    ec.clear();
    if (!trx::fs::exists(filename, ec) || !trx::fs::is_directory(filename, ec)) {
      rm_dir(tmp_dir);
      throw std::runtime_error("Failed to create output directory: " + filename);
    }
    const trx::fs::path header_path = dest_path / "header.json";
    if (!trx::fs::exists(header_path)) {
      rm_dir(tmp_dir);
      throw std::runtime_error("Missing header.json in output directory: " + header_path.string());
    }
  }

  rm_dir(tmp_dir);
}

void populate_fps(const string &name, std::map<std::string, std::tuple<long long, long long>> &files_pointer_size) {
  const trx::fs::path root(name);
  std::error_code ec;
  if (!trx::fs::exists(root, ec) || !trx::fs::is_directory(root, ec)) {
    return;
  }
  ec.clear();
  for (trx::fs::recursive_directory_iterator it(root, ec), end; it != end; it.increment(ec)) {
    if (ec) {
      throw std::runtime_error("Failed to read directory: " + root.string());
    }
    const trx::fs::path entry_path = it->path();
    const std::string filename = entry_path.filename().string();
    if (!filename.empty() && filename[0] == '.') {
      if (it->is_directory(ec)) {
        it.disable_recursion_pending();
      }
      continue;
    }
    if (filename == "__MACOSX") {
      if (it->is_directory(ec)) {
        it.disable_recursion_pending();
      }
      continue;
    }

    std::error_code entry_ec;
    if (!it->is_regular_file(entry_ec)) {
      continue;
    }

    const std::string elem_filename = entry_path.string();
    std::string ext = get_ext(elem_filename);

    if (ext == "json") {
      continue;
    }

    if (!trx::detail::_is_dtype_valid(ext)) {
      throw std::invalid_argument(std::string("The dtype of ") + elem_filename + std::string(" is not supported"));
    }

    const int dtype_size = trx::detail::_sizeof_dtype(ext);
    std::error_code size_ec;
    auto raw_size = trx::fs::file_size(entry_path, size_ec);
    if (size_ec) {
      throw std::runtime_error("Failed to stat file: " + elem_filename);
    }

    if (raw_size % static_cast<std::uintmax_t>(dtype_size) == 0) {
      auto size = raw_size / static_cast<std::uintmax_t>(dtype_size);
      files_pointer_size[elem_filename] = std::make_tuple(0, static_cast<long long>(size));
    } else if (raw_size == 1) {
      files_pointer_size[elem_filename] = std::make_tuple(0, 0);
    } else {
      throw std::invalid_argument("Wrong size of datatype");
    }
  }
}
std::string get_base(const std::string &delimiter, const std::string &str) {
  std::string token;

  if (str.rfind(delimiter) + 1 < str.length()) {
    token = str.substr(str.rfind(delimiter) + 1);
  } else {
    token = str;
  }
  return token;
}

std::string get_ext(const std::string &str) {
  std::string ext;
  constexpr char kDelimiter = '.';

  const std::size_t pos = str.rfind(kDelimiter);
  if (pos != std::string::npos && pos + 1 < str.length()) {
    ext = str.substr(pos + 1);
  }
  return ext;
}

json load_header(zip_t *zfolder) {
  if (zfolder == nullptr) {
    throw std::invalid_argument("Zip archive pointer is null");
  }
  // load file
  zip_file_t *zh = zip_fopen(zfolder, "header.json", ZIP_FL_UNCHANGED);
  if (zh == nullptr) {
    throw std::runtime_error("Failed to open header.json in zip archive");
  }

  // read data from file in chunks of 255 characters until data is fully loaded
  const std::size_t buff_len = 255;
  std::vector<char> buffer(buff_len);

  std::string jstream;
  zip_int64_t nbytes = 0;
  while ((nbytes = zip_fread(zh, buffer.data(), buff_len - 1)) > 0) {
    jstream.append(buffer.data(), static_cast<std::size_t>(nbytes));
  }

  zip_fclose(zh);

  // convert jstream data into Json.
  std::string err;
  auto root = json::parse(jstream, err);
  if (!err.empty()) {
    throw std::runtime_error("Failed to parse header.json: " + err);
  }
  return root;
}

void allocate_file(const std::string &path, std::size_t size) {
  std::ofstream file(path);
  if (file.is_open()) {
    const std::string s(size, '\0');
    file << s;
    file.flush();
    file.close();
  } else {
    std::cerr << "Failed to allocate file : " << sys_error() << '\n';
  }
}

mio::shared_mmap_sink _create_memmap(std::string filename,
                                     const std::tuple<int, int> &shape,
                                     const std::string &mode,
                                     const std::string &dtype,
                                     long long offset) {
  static_cast<void>(mode);
  const std::size_t filesize = static_cast<std::size_t>(std::get<0>(shape)) *
                               static_cast<std::size_t>(std::get<1>(shape)) *
                               static_cast<std::size_t>(trx::detail::_sizeof_dtype(dtype));
  // if file does not exist, create and allocate it

  struct stat buffer {};
  if (stat(filename.c_str(), &buffer) != 0) {
    allocate_file(filename, filesize);
  }

  if (filesize == 0) {
    return mio::shared_mmap_sink();
  }

  // std::error_code error;

  mio::shared_mmap_sink rw_mmap(filename, offset, filesize);

  return rw_mmap;
}

// TODO: support FORTRAN ORDERING
// template <typename Derived>

json assignHeader(const json &root) {
  json header = root;
  // MatrixXf affine(4, 4);
  // RowVectorXi dimensions(3);

  // for (int i = 0; i < 4; i++)
  // {
  // 	for (int j = 0; j < 4; j++)
  // 	{
  // 		affine << root["VOXEL_TO_RASMM"][i][j].asFloat();
  // 	}
  // }

  // for (int i = 0; i < 3; i++)
  // {
  // 	dimensions[i] << root["DIMENSIONS"][i].asUInt();
  // }
  // header["VOXEL_TO_RASMM"] = affine;
  // header["DIMENSIONS"] = dimensions;
  // header["NB_VERTICES"] = (int)root["NB_VERTICES"].asUInt();
  // header["NB_STREAMLINES"] = (int)root["NB_STREAMLINES"].asUInt();

  return header;
}

void get_reference_info(
    const std::string &reference,
    const Eigen::MatrixXf &affine,          // NOLINT(misc-include-cleaner)
    const Eigen::RowVectorXi &dimensions) { // NOLINT(misc-use-internal-linkage,misc-include-cleaner)
  static_cast<void>(affine);
  static_cast<void>(dimensions);
  // TODO: find a library to use for nifti and trk (MRtrix??)
  //  if (reference.find(".nii") != std::string::npos)
  //  {
  //  }
  if (reference.find(".trk") != std::string::npos) {
    // TODO: Create exception class
    throw std::runtime_error("Trk reference not implemented");
  }
  // TODO: Create exception class
  throw std::runtime_error("Trk reference not implemented");
}

void copy_dir(const string &src, const string &dst) {
  const trx::fs::path src_path(src);
  const trx::fs::path dst_path(dst);
  std::error_code ec;
  if (!trx::fs::exists(src_path, ec) || !trx::fs::is_directory(src_path, ec)) {
    return;
  }

  if (!trx::fs::create_directories(dst_path, ec) && ec) {
    throw std::runtime_error(std::string("Could not create directory ") + dst);
  }

  ec.clear();
  const auto options = trx::fs::copy_options::recursive | trx::fs::copy_options::overwrite_existing |
                       trx::fs::copy_options::skip_symlinks;
  trx::fs::copy(src_path, dst_path, options, ec);
  if (ec) {
    throw std::runtime_error("Failed to copy directory: " + ec.message());
  }
}

void copy_file(const string &src, const string &dst) {
  std::error_code ec;
  trx::fs::copy_file(src, dst, trx::fs::copy_options::overwrite_existing, ec);
  if (ec) {
    throw std::runtime_error(std::string("Failed to copy file ") + src + ": " + ec.message());
  }
}
int rm_dir(const string &d) {
  std::error_code ec;
  trx::fs::remove_all(d, ec);
  return ec ? -1 : 0;
}

std::string make_temp_dir(const std::string &prefix) {
  const string env_tmp = get_env("TRX_TMPDIR");
  std::string base_dir;

  if (!env_tmp.empty()) {
    if (env_tmp == "use_working_dir") {
      base_dir = ".";
    } else {
      const trx::fs::path env_path(env_tmp);
      std::error_code ec;
      if (trx::fs::exists(env_path, ec) && trx::fs::is_directory(env_path, ec)) {
        base_dir = env_path.string();
      }
    }
  }

  if (base_dir.empty()) {
    const std::array<string, 3> candidates{get_env("TMPDIR"), get_env("TEMP"), get_env("TMP")};
    for (const auto &candidate : candidates) {
      if (candidate.empty()) {
        continue;
      }
      const trx::fs::path path(candidate);
      std::error_code ec;
      if (trx::fs::exists(path, ec) && trx::fs::is_directory(path, ec)) {
        base_dir = path.string();
        break;
      }
    }
  }
  if (base_dir.empty()) {
    std::error_code ec;
    auto sys_tmp = trx::fs::temp_directory_path(ec);
    if (!ec) {
      base_dir = sys_tmp.string();
    }
  }
  if (base_dir.empty()) {
#if defined(_WIN32) || defined(_WIN64)
    base_dir = ".";
#else
    base_dir = "/tmp";
#endif
  }

  const trx::fs::path base_path(base_dir);
  std::error_code ec;
  if (!trx::fs::exists(base_path, ec)) {
    ec.clear();
    trx::fs::create_directories(base_path, ec);
    if (ec) {
      throw std::runtime_error("Failed to create base temp directory: " + base_dir);
    }
  }

  static std::mt19937_64 rng(std::random_device{}());
  std::uniform_int_distribution<uint64_t> dist;
  for (int attempt = 0; attempt < 100; ++attempt) {
    const trx::fs::path candidate = base_path / (prefix + "_" + std::to_string(dist(rng)));
    ec.clear();
    if (trx::fs::create_directory(candidate, ec)) {
      return candidate.string();
    }
    if (ec && ec != std::errc::file_exists) {
      throw std::runtime_error("Failed to create temporary directory: " + ec.message());
    }
  }
  throw std::runtime_error("Failed to create temporary directory");
}

std::string extract_zip_to_directory(zip_t *zfolder) {
  if (zfolder == nullptr) {
    throw std::invalid_argument("Zip archive pointer is null");
  }
  const std::string root_dir = make_temp_dir("trx_zip");
  const trx::fs::path normalized_root = trx::fs::path(root_dir).lexically_normal();

  const zip_int64_t num_entries = zip_get_num_entries(zfolder, ZIP_FL_UNCHANGED);
  for (zip_int64_t i = 0; i < num_entries; ++i) {
    const auto *entry_name = zip_get_name(zfolder, i, ZIP_FL_UNCHANGED);
    if (entry_name == nullptr) {
      continue;
    }
    std::string entry(entry_name);

    const trx::fs::path entry_path(entry);
    if (entry_path.is_absolute()) {
      throw std::runtime_error("Zip entry has absolute path: " + entry);
    }

    const trx::fs::path normalized_entry = entry_path.lexically_normal();
    const trx::fs::path out_path = normalized_root / normalized_entry;
    const trx::fs::path normalized_out = out_path.lexically_normal();

    if (!is_path_within(normalized_out, normalized_root)) {
      throw std::runtime_error("Zip entry escapes temporary directory: " + entry);
    }

    if (!entry.empty() && entry.back() == '/') {
      std::error_code ec;
      trx::fs::create_directories(normalized_out, ec);
      if (ec) {
        throw std::runtime_error("Failed to create directory: " + normalized_out.string());
      }
      continue;
    }

    std::error_code ec;
    trx::fs::create_directories(normalized_out.parent_path(), ec);
    if (ec) {
      throw std::runtime_error("Failed to create parent directory: " + normalized_out.parent_path().string());
    }

    zip_file_t *zf = zip_fopen_index(zfolder, i, ZIP_FL_UNCHANGED);
    if (zf == nullptr) {
      throw std::runtime_error("Failed to open zip entry: " + entry);
    }

    std::ofstream out(normalized_out.string(), std::ios::binary);
    if (!out.is_open()) {
      zip_fclose(zf);
      throw std::runtime_error("Failed to open output file: " + normalized_out.string());
    }

    std::array<char, 4096> buffer{};
    zip_int64_t nbytes = 0;
    while ((nbytes = zip_fread(zf, buffer.data(), buffer.size())) > 0) {
      out.write(buffer.data(), nbytes);
      if (!out) {
        out.close();
        zip_fclose(zf);
        throw std::runtime_error("Failed to write to output file: " + normalized_out.string());
      }
    }
    if (nbytes < 0) {
      out.close();
      zip_fclose(zf);
      throw std::runtime_error("Failed to read data from zip entry: " + entry);
    }

    out.flush();
    if (!out) {
      out.close();
      zip_fclose(zf);
      throw std::runtime_error("Failed to flush output file: " + normalized_out.string());
    }

    out.close();
    zip_fclose(zf);
  }

  return root_dir;
}

void zip_from_folder(zip_t *zf,
                     const std::string &root,
                     const std::string &directory,
                     zip_uint32_t compression_standard) {
  std::error_code ec;
  for (trx::fs::recursive_directory_iterator it(directory, ec), end; it != end; it.increment(ec)) {
    if (ec) {
      throw std::runtime_error("Failed to read directory: " + directory);
    }
    const trx::fs::path current = it->path();
    const std::string zip_fname = rm_root(root, current.string());
    std::error_code entry_ec;
    if (it->is_directory(entry_ec)) {
      zip_dir_add(zf, zip_fname.c_str(), ZIP_FL_ENC_GUESS);
      continue;
    }
    if (!it->is_regular_file(entry_ec)) {
      continue;
    }

    const std::string fullpath = current.string();
    zip_source_t *source = zip_source_file(zf, fullpath.c_str(), 0, 0);
    if (source == nullptr) {
      throw std::runtime_error(std::string("Error adding file ") + zip_fname + ": " + zip_strerror(zf));
    }
    const zip_int64_t file_idx = zip_file_add(zf, zip_fname.c_str(), source, ZIP_FL_ENC_UTF_8);
    if (file_idx < 0) {
      zip_source_free(source);
      throw std::runtime_error(std::string("Error adding file ") + zip_fname + ": " + zip_strerror(zf));
    }
    const zip_int32_t compression = static_cast<zip_int32_t>(compression_standard);
    if (zip_set_file_compression(zf, file_idx, compression, 0) < 0) {
      throw std::runtime_error(std::string("Error setting compression for ") + zip_fname + ": " + zip_strerror(zf));
    }
  }
}

std::string rm_root(const std::string &root, const std::string &path) {
  std::string stripped;

  const std::size_t index = path.find(root);
  if (index != std::string::npos) {
    stripped = path.substr(index + root.size() + 1, path.size() - index - root.size() - 1);
  }
  return stripped;
}
}; // namespace trx