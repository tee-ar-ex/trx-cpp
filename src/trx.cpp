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
#include <unordered_set>
#include <sys/stat.h>
#include <system_error>
#include <tuple>
#include <vector>
#include <zip.h>
#include <zipconf.h>
#if defined(_WIN32) || defined(_WIN64)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <mio/shared_mmap.hpp>
#include <trx/trx.h>
#include <trx/detail/zip_raii.h>

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
    auto [base, dim, dtype] = trx::detail::_split_ext_with_dimensionality(normalized);
    if (base == "positions") {
      out_dtype = dtype;
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

TrxSaveMode resolve_save_mode(const std::string &filename, TrxSaveMode requested) {
  if (requested != TrxSaveMode::Auto) {
    return requested;
  }
  const std::string ext = get_ext(filename);
  if (ext == "zip" || ext == "trx") {
    return TrxSaveMode::Archive;
  }
  return TrxSaveMode::Directory;
}

std::array<double, 3> read_xyz_as_double(const TypedArray &positions, size_t row_index) {
  if (positions.cols != 3) {
    throw TrxFormatError("Positions must have 3 columns.");
  }
  if (row_index >= static_cast<size_t>(positions.rows)) {
    throw std::out_of_range("Position row index out of range");
  }
  if (positions.dtype == "float16") {
    const auto view = positions.as_matrix<Eigen::half>();
    return {static_cast<double>(view(static_cast<Eigen::Index>(row_index), 0)),
            static_cast<double>(view(static_cast<Eigen::Index>(row_index), 1)),
            static_cast<double>(view(static_cast<Eigen::Index>(row_index), 2))};
  }
  if (positions.dtype == "float32") {
    const auto view = positions.as_matrix<float>();
    return {static_cast<double>(view(static_cast<Eigen::Index>(row_index), 0)),
            static_cast<double>(view(static_cast<Eigen::Index>(row_index), 1)),
            static_cast<double>(view(static_cast<Eigen::Index>(row_index), 2))};
  }
  if (positions.dtype == "float64") {
    const auto view = positions.as_matrix<double>();
    return {view(static_cast<Eigen::Index>(row_index), 0),
            view(static_cast<Eigen::Index>(row_index), 1),
            view(static_cast<Eigen::Index>(row_index), 2)};
  }
  throw TrxDTypeError("Unsupported positions dtype for streamline extraction: " + positions.dtype);
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
    throw TrxIOError("Input path does not exist: " + path);
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
  detail::ZipArchive zf(open_zip_for_read(path, err));
  if (!zf) {
    throw TrxIOError("Could not open zip file: " + path);
  }
  std::string dtype;
  const zip_int64_t count = zip_get_num_entries(zf.get(), 0);
  for (zip_int64_t i = 0; i < count; ++i) {
    const auto *name = zip_get_name(zf.get(), i, 0);
    if (name == nullptr) {
      continue;
    }
    if (parse_positions_dtype(name, dtype)) {
      break;
    }
  }
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
    throw TrxIOError("Input path does not exist: " + path);
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

const TypedArray *AnyTrxFile::get_dps(const std::string &name) const {
  auto it = data_per_streamline.find(name);
  if (it == data_per_streamline.end()) {
    return nullptr;
  }
  return &it->second;
}

const TypedArray *AnyTrxFile::get_dpv(const std::string &name) const {
  auto it = data_per_vertex.find(name);
  if (it == data_per_vertex.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<std::array<double, 3>> AnyTrxFile::get_streamline(size_t streamline_index) const {
  if (offsets_u64.empty()) {
    throw TrxFormatError("TRX offsets are empty.");
  }
  const size_t n_streamlines = offsets_u64.size() - 1;
  if (streamline_index >= n_streamlines) {
    throw std::out_of_range("Streamline index out of range");
  }
  const uint64_t start = offsets_u64[streamline_index];
  const uint64_t end = offsets_u64[streamline_index + 1];
  std::vector<std::array<double, 3>> out;
  out.reserve(static_cast<size_t>(end - start));
  for (uint64_t i = start; i < end; ++i) {
    out.push_back(read_xyz_as_double(positions, static_cast<size_t>(i)));
  }
  return out;
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

  header = default_header();
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
    throw TrxIOError("Input path does not exist: " + path);
  }
  std::error_code ec;
  if (trx::fs::is_directory(input, ec) && !ec) {
    return AnyTrxFile::load_from_directory(path);
  }
  return AnyTrxFile::load_from_zip(path);
}

AnyTrxFile AnyTrxFile::load_from_zip(const std::string &filename) {
  int errorp = 0;
  detail::ZipArchive zf(open_zip_for_read(filename, errorp));
  if (!zf) {
    throw TrxIOError("Could not open zip file: " + filename);
  }

  std::string temp_dir = extract_zip_to_directory(zf.get());

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
  std::ifstream header_file;
  for (int attempt = 0; attempt < 5; ++attempt) {
    header_file.open(header_name);
    if (header_file.is_open()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!header_file.is_open()) {
    std::error_code ec;
    const bool exists = trx::fs::exists(directory, ec);
    const int open_err = errno;
    std::string detail = "Failed to open header.json at: " + header_name;
    detail += " exists=" + std::string(exists ? "true" : "false");
    detail += " errno=" + std::to_string(open_err) + " msg=" + std::string(std::strerror(open_err));
    if (exists) {
      std::vector<std::string> files;
      for (const auto &entry : trx::fs::directory_iterator(directory, ec)) {
        if (ec) {
          break;
        }
        files.push_back(entry.path().filename().string());
      }
      if (!files.empty()) {
        std::sort(files.begin(), files.end());
        detail += " files=[";
        for (size_t i = 0; i < files.size(); ++i) {
          if (i > 0) {
            detail += ",";
          }
          detail += files[i];
        }
        detail += "]";
      }
    }
    throw TrxIOError(detail);
  }
  std::string jstream((std::istreambuf_iterator<char>(header_file)), std::istreambuf_iterator<char>());
  header_file.close();
  std::string err;
  json header = json::parse(jstream, err);
  if (!err.empty()) {
    throw TrxIOError("Failed to parse header.json: " + err);
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
    throw TrxFormatError("Missing NB_VERTICES or NB_STREAMLINES in header.json");
  }

  const int nb_vertices = header["NB_VERTICES"].int_value();
  const int nb_streamlines = header["NB_STREAMLINES"].int_value();

  for (auto x = dict_pointer_size.rbegin(); x != dict_pointer_size.rend(); ++x) {
    const std::string elem_filename = x->first;

    std::string folder = folder_from_path(elem_filename, root);

    auto [base, dim, ext] = trx::detail::_split_ext_with_dimensionality(elem_filename);

    ext = _normalize_dtype(ext);

    long long size = std::get<1>(x->second);

    if (base == "positions" && (folder.empty() || folder == ".")) {
      if (size != static_cast<long long>(nb_vertices) * 3 || dim != 3) {
        throw TrxFormatError("Wrong positions size/dimensionality");
      }
      if (ext != "float16" && ext != "float32" && ext != "float64") {
        throw TrxDTypeError("Unsupported positions dtype: " + ext);
      }
      trx.positions = make_typed_array(elem_filename, nb_vertices, 3, ext);
    } else if (base == "offsets" && (folder.empty() || folder == ".")) {
      if (size != static_cast<long long>(nb_streamlines) + 1 || dim != 1) {
        throw TrxFormatError("Wrong offsets size/dimensionality");
      }
      if (ext != "uint32" && ext != "uint64") {
        throw TrxDTypeError("Unsupported offsets dtype: " + ext);
      }
      trx.offsets = make_typed_array(elem_filename, nb_streamlines + 1, 1, ext);
    } else if (folder == "dps") {
      const int nb_scalar = nb_streamlines > 0 ? static_cast<int>(size / nb_streamlines) : 0;
      if (nb_streamlines == 0 || size % nb_streamlines != 0 || nb_scalar != dim) {
        throw TrxFormatError("Wrong dps size/dimensionality");
      }
      trx.data_per_streamline.emplace(base, make_typed_array(elem_filename, nb_streamlines, nb_scalar, ext));
    } else if (folder == "dpv") {
      const int nb_scalar = nb_vertices > 0 ? static_cast<int>(size / nb_vertices) : 0;
      if (nb_vertices == 0 || size % nb_vertices != 0 || nb_scalar != dim) {
        throw TrxFormatError("Wrong dpv size/dimensionality");
      }
      trx.data_per_vertex.emplace(base, make_typed_array(elem_filename, nb_vertices, nb_scalar, ext));
    } else if (folder.rfind("dpg", 0) == 0) {
      if (size != dim) {
        throw TrxFormatError("Wrong dpg size/dimensionality");
      }
      std::string data_name = path_basename(base);
      std::string sub_folder = path_basename(folder);
      trx.data_per_group[sub_folder].emplace(data_name,
                                             make_typed_array(elem_filename, 1, static_cast<int>(size), ext));
    } else if (folder == "groups") {
      if (dim != 1) {
        throw TrxFormatError("Wrong group dimensionality");
      }
      if (ext != "uint32") {
        throw TrxDTypeError("Unsupported group dtype: " + ext);
      }
      trx.groups.emplace(base, make_typed_array(elem_filename, static_cast<int>(size), 1, ext));
    } else {
      throw TrxFormatError("Entry is not part of a valid TRX structure: " + elem_filename);
    }
  }

  if (trx.positions.empty() || trx.offsets.empty()) {
    throw TrxFormatError("Missing essential data.");
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
      throw TrxDTypeError("Unsupported offsets datatype: " + trx.offsets.dtype);
    }
  }

  if (offsets_count > 1) {
    trx.lengths.resize(offsets_count - 1);
    for (size_t i = 0; i + 1 < offsets_count; ++i) {
      const uint64_t diff = trx.offsets_u64[i + 1] - trx.offsets_u64[i];
      if (diff > std::numeric_limits<uint32_t>::max()) {
        throw TrxFormatError("Offset difference exceeds uint32 range");
      }
      trx.lengths[i] = static_cast<uint32_t>(diff);
    }
  }

  return trx;
}

void AnyTrxFile::save(const std::string &filename, zip_uint32_t compression_standard) {
  TrxSaveOptions options;
  options.compression_standard = compression_standard;
  save(filename, options);
}

void AnyTrxFile::save(const std::string &filename, const TrxSaveOptions &options) {
  const std::string ext = get_ext(filename);
  const TrxSaveMode save_mode = resolve_save_mode(filename, options.mode);
  if (ext.size() > 0 && ext != "zip" && ext != "trx") {
    throw TrxDTypeError("Unsupported extension: " + ext);
  }

  if (offsets.empty()) {
    throw TrxFormatError("Cannot save TRX without offsets data");
  }
  if (offsets_u64.empty()) {
    throw TrxFormatError("Cannot save TRX without decoded offsets");
  }
  if (header["NB_STREAMLINES"].is_number()) {
    const auto nb_streamlines = static_cast<size_t>(header["NB_STREAMLINES"].int_value());
    if (offsets_u64.size() != nb_streamlines + 1) {
      throw TrxFormatError("TRX offsets size does not match NB_STREAMLINES");
    }
  }
  if (header["NB_VERTICES"].is_number()) {
    const auto nb_vertices = static_cast<uint64_t>(header["NB_VERTICES"].int_value());
    const auto last = offsets_u64.back();
    if (last != nb_vertices) {
      throw TrxFormatError("TRX offsets sentinel does not match NB_VERTICES");
    }
  }
  for (size_t i = 1; i < offsets_u64.size(); ++i) {
    if (offsets_u64[i] < offsets_u64[i - 1]) {
      throw TrxFormatError("TRX offsets must be monotonically increasing");
    }
  }
  if (!positions.empty()) {
    const auto last = offsets_u64.back();
    if (last != static_cast<uint64_t>(positions.rows)) {
      throw TrxFormatError("TRX positions row count does not match offsets sentinel");
    }
  }

  const std::string source_dir =
      !_uncompressed_folder_handle.empty() ? _uncompressed_folder_handle : _backing_directory;
  if (source_dir.empty()) {
    throw TrxIOError("TRX file has no backing directory to save from");
  }

  if (save_mode == TrxSaveMode::Archive) {
    int errorp;
    detail::ZipArchive zf(zip_open(filename.c_str(), ZIP_CREATE + ZIP_TRUNCATE, &errorp));
    if (!zf) {
      throw TrxIOError("Could not open archive " + filename + ": " + strerror(errorp));
    }

    const std::string header_payload = header.dump() + "\n";
    zip_source_t *header_source =
        zip_source_buffer(zf.get(), header_payload.data(), header_payload.size(), 0 /* do not free */);
    if (header_source == nullptr) {
      throw TrxIOError("Failed to create zip source for header.json: " +
                               std::string(zip_strerror(zf.get())));
    }
    const zip_int64_t header_idx =
        zip_file_add(zf.get(), "header.json", header_source, ZIP_FL_ENC_UTF_8 | ZIP_FL_OVERWRITE);
    if (header_idx < 0) {
      throw TrxIOError("Failed to add header.json to archive: " + std::string(zip_strerror(zf.get())));
    }
    const zip_int32_t compression = static_cast<zip_int32_t>(options.compression_standard);
    if (zip_set_file_compression(zf.get(), header_idx, compression, 0) < 0) {
      throw TrxIOError("Failed to set compression for header.json: " +
                               std::string(zip_strerror(zf.get())));
    }

    const std::unordered_set<std::string> skip = {"header.json"};
    zip_from_folder(zf.get(), source_dir, source_dir, options.compression_standard, &skip);
    zf.commit(filename);
  } else {
    std::error_code ec;
    if (trx::fs::exists(filename, ec) && trx::fs::is_directory(filename, ec)) {
      if (!options.overwrite_existing) {
        throw TrxIOError("Output directory already exists: " + filename);
      }
      if (rm_dir(filename) != 0) {
        throw TrxIOError("Could not remove existing directory " + filename);
      }
    }
    trx::fs::path dest_path(filename);
    if (dest_path.has_parent_path()) {
      mkdir_or_throw(dest_path.parent_path().string());
    }
    std::error_code source_ec;
    const trx::fs::path source_path = trx::fs::weakly_canonical(trx::fs::path(source_dir), source_ec);
    std::error_code dest_ec;
    const trx::fs::path normalized_dest = trx::fs::weakly_canonical(dest_path, dest_ec);
    const bool same_directory = !source_ec && !dest_ec && source_path == normalized_dest;

    if (!same_directory) {
      copy_dir(source_dir, filename);
    }

    const trx::fs::path final_header_path = dest_path / "header.json";
    std::ofstream out_json(final_header_path, std::ios::out | std::ios::trunc);
    if (!out_json.is_open()) {
      throw TrxIOError("Failed to write header.json to: " + final_header_path.string());
    }
    out_json << header.dump() << std::endl;
    out_json.close();

    ec.clear();
    if (!trx::fs::exists(filename, ec) || !trx::fs::is_directory(filename, ec)) {
      throw TrxIOError("Failed to create output directory: " + filename);
    }
    if (!trx::fs::exists(final_header_path)) {
      throw TrxFormatError("Missing header.json in output directory: " + final_header_path.string());
    }
  }
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
      throw TrxIOError("Failed to read directory: " + root.string());
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
      throw TrxDTypeError(std::string("The dtype of ") + elem_filename + std::string(" is not supported"));
    }

    const int dtype_size = trx::detail::_sizeof_dtype(ext);
    std::error_code size_ec;
    auto raw_size = trx::fs::file_size(entry_path, size_ec);
    if (size_ec) {
      throw TrxIOError("Failed to stat file: " + elem_filename);
    }

    if (raw_size % static_cast<std::uintmax_t>(dtype_size) == 0) {
      auto size = raw_size / static_cast<std::uintmax_t>(dtype_size);
      files_pointer_size[elem_filename] = std::make_tuple(0, static_cast<long long>(size));
    } else if (raw_size == 1) {
      files_pointer_size[elem_filename] = std::make_tuple(0, 0);
    } else {
      throw TrxFormatError("Wrong size of datatype");
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
    throw TrxArgumentError("Zip archive pointer is null");
  }
  // load file
  detail::ZipFile zh(zip_fopen(zfolder, "header.json", ZIP_FL_UNCHANGED));
  if (!zh) {
    throw TrxIOError("Failed to open header.json in zip archive");
  }

  // read data from file in chunks of 255 characters until data is fully loaded
  const std::size_t buff_len = 255;
  std::vector<char> buffer(buff_len);

  std::string jstream;
  zip_int64_t nbytes = 0;
  while ((nbytes = zip_fread(zh.get(), buffer.data(), buff_len - 1)) > 0) {
    jstream.append(buffer.data(), static_cast<std::size_t>(nbytes));
  }

  // convert jstream data into Json.
  std::string err;
  auto root = json::parse(jstream, err);
  if (!err.empty()) {
    throw TrxIOError("Failed to parse header.json: " + err);
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

// Known limitation: only C (row-major) ordering is supported; Fortran ordering is not.
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
  // Known limitation: NIfTI support is partially addressed by nifti_io.cpp; TRK is not yet supported.
  //  if (reference.find(".nii") != std::string::npos)
  //  {
  //  }
  if (reference.find(".trk") != std::string::npos) {
    throw TrxError("Trk reference not implemented");
  }
  throw TrxError("Trk reference not implemented");
}

void copy_dir(const string &src, const string &dst) {
  const trx::fs::path src_path(src);
  const trx::fs::path dst_path(dst);
  std::error_code ec;
  if (!trx::fs::exists(src_path, ec) || !trx::fs::is_directory(src_path, ec)) {
    return;
  }

  mkdir_or_throw(dst);

  ec.clear();
  const auto options = trx::fs::copy_options::recursive | trx::fs::copy_options::overwrite_existing |
                       trx::fs::copy_options::skip_symlinks;
  trx::fs::copy(src_path, dst_path, options, ec);
  if (ec) {
    throw TrxIOError("Failed to copy directory: " + ec.message());
  }
}

void copy_file(const string &src, const string &dst) {
  std::error_code ec;
  trx::fs::copy_file(src, dst, trx::fs::copy_options::overwrite_existing, ec);
  if (ec) {
    throw TrxIOError(std::string("Failed to copy file ") + src + ": " + ec.message());
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
    mkdir_or_throw(base_path.string());
  }

  static std::mt19937_64 rng(std::random_device{}());
  std::uniform_int_distribution<uint64_t> dist;
  const uint64_t pid =
#if defined(_WIN32) || defined(_WIN64)
      static_cast<uint64_t>(_getpid());
#else
      static_cast<uint64_t>(getpid());
#endif
  for (int attempt = 0; attempt < 100; ++attempt) {
    const trx::fs::path candidate =
        base_path / (prefix + "_" + std::to_string(pid) + "_" + std::to_string(dist(rng)));
    ec.clear();
    if (trx::fs::create_directory(candidate, ec)) {
      return candidate.string();
    }
    if (ec && ec != std::errc::file_exists) {
      throw TrxIOError("Failed to create temporary directory: " + ec.message());
    }
  }
  throw TrxIOError("Failed to create temporary directory");
}

std::string extract_zip_to_directory(zip_t *zfolder) {
  if (zfolder == nullptr) {
    throw TrxArgumentError("Zip archive pointer is null");
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
      throw TrxIOError("Zip entry has absolute path: " + entry);
    }

    const trx::fs::path normalized_entry = entry_path.lexically_normal();
    const trx::fs::path out_path = normalized_root / normalized_entry;
    const trx::fs::path normalized_out = out_path.lexically_normal();

    if (!is_path_within(normalized_out, normalized_root)) {
      throw TrxIOError("Zip entry escapes temporary directory: " + entry);
    }

    if (!entry.empty() && entry.back() == '/') {
      mkdir_or_throw(normalized_out.string());
      continue;
    }

    mkdir_or_throw(normalized_out.parent_path().string());

    detail::ZipFile zf(zip_fopen_index(zfolder, i, ZIP_FL_UNCHANGED));
    if (!zf) {
      throw TrxIOError("Failed to open zip entry: " + entry);
    }

    std::ofstream out(normalized_out.string(), std::ios::binary);
    if (!out.is_open()) {
      throw TrxIOError("Failed to open output file: " + normalized_out.string());
    }

    std::array<char, 4096> buffer{};
    zip_int64_t nbytes = 0;
    while ((nbytes = zip_fread(zf.get(), buffer.data(), buffer.size())) > 0) {
      out.write(buffer.data(), nbytes);
      if (!out) {
        throw TrxIOError("Failed to write to output file: " + normalized_out.string());
      }
    }
    if (nbytes < 0) {
      throw TrxIOError("Failed to read data from zip entry: " + entry);
    }

    out.flush();
    if (!out) {
      throw TrxIOError("Failed to flush output file: " + normalized_out.string());
    }
  }

  return root_dir;
}

void zip_from_folder(zip_t *zf,
                     const std::string &root,
                     const std::string &directory,
                     zip_uint32_t compression_standard,
                     const std::unordered_set<std::string> *skip) {
  std::error_code ec;
  for (trx::fs::recursive_directory_iterator it(directory, ec), end; it != end; it.increment(ec)) {
    if (ec) {
      throw TrxIOError("Failed to read directory: " + directory);
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
      throw TrxIOError(std::string("Error adding file ") + zip_fname + ": " + zip_strerror(zf));
    }
    if (skip && skip->find(zip_fname) != skip->end()) {
      zip_source_free(source);
      continue;
    }
    const zip_int64_t file_idx = zip_file_add(zf, zip_fname.c_str(), source, ZIP_FL_ENC_UTF_8);
    if (file_idx < 0) {
      zip_source_free(source);
      throw TrxIOError(std::string("Error adding file ") + zip_fname + ": " + zip_strerror(zf));
    }
    const zip_int32_t compression = static_cast<zip_int32_t>(compression_standard);
    if (zip_set_file_compression(zf, file_idx, compression, 0) < 0) {
      throw TrxIOError(std::string("Error setting compression for ") + zip_fname + ": " + zip_strerror(zf));
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

namespace {
TrxScalarType scalar_type_from_dtype(const std::string &dtype) {
  if (dtype == "float16") {
    return TrxScalarType::Float16;
  }
  if (dtype == "float32") {
    return TrxScalarType::Float32;
  }
  if (dtype == "float64") {
    return TrxScalarType::Float64;
  }
  return TrxScalarType::Float32;
}

std::string typed_array_filename(const std::string &base, const TypedArray &arr) {
  if (arr.cols <= 1) {
    return base + "." + arr.dtype;
  }
  return base + "." + std::to_string(arr.cols) + "." + arr.dtype;
}

void write_typed_array_file(const std::string &path, const TypedArray &arr) {
  const auto bytes = arr.to_bytes();
  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    throw TrxIOError("Failed to open output file: " + path);
  }
  if (bytes.data && bytes.size > 0) {
    out.write(reinterpret_cast<const char *>(bytes.data), static_cast<std::streamsize>(bytes.size));
  }
  out.flush();
  out.close();
}
} // namespace

void AnyTrxFile::for_each_positions_chunk(size_t chunk_bytes, const PositionsChunkCallback &fn) const {
  if (positions.empty()) {
    throw TrxFormatError("TRX positions are empty.");
  }
  if (positions.cols != 3) {
    throw TrxFormatError("Positions must have 3 columns.");
  }
  if (!fn) {
    return;
  }
  const size_t elem_size = static_cast<size_t>(detail::_sizeof_dtype(positions.dtype));
  const size_t bytes_per_point = elem_size * 3;
  const size_t total_points = static_cast<size_t>(positions.rows);
  size_t points_per_chunk = 0;
  if (chunk_bytes == 0) {
    points_per_chunk = total_points;
  } else {
    points_per_chunk = std::max<size_t>(1, chunk_bytes / bytes_per_point);
  }
  const auto bytes = positions.to_bytes();
  const auto *base = bytes.data;
  const auto dtype = scalar_type_from_dtype(positions.dtype);
  for (size_t offset = 0; offset < total_points; offset += points_per_chunk) {
    const size_t count = std::min(points_per_chunk, total_points - offset);
    const void *ptr = base + offset * bytes_per_point;
    fn(dtype, ptr, offset, count);
  }
}

void AnyTrxFile::for_each_positions_chunk_mutable(size_t chunk_bytes, const PositionsChunkMutableCallback &fn) {
  if (positions.empty()) {
    throw TrxFormatError("TRX positions are empty.");
  }
  if (positions.cols != 3) {
    throw TrxFormatError("Positions must have 3 columns.");
  }
  if (!fn) {
    return;
  }
  const size_t elem_size = static_cast<size_t>(detail::_sizeof_dtype(positions.dtype));
  const size_t bytes_per_point = elem_size * 3;
  const size_t total_points = static_cast<size_t>(positions.rows);
  size_t points_per_chunk = 0;
  if (chunk_bytes == 0) {
    points_per_chunk = total_points;
  } else {
    points_per_chunk = std::max<size_t>(1, chunk_bytes / bytes_per_point);
  }
  const auto bytes = positions.to_bytes_mutable();
  auto *base = bytes.data;
  const auto dtype = scalar_type_from_dtype(positions.dtype);
  for (size_t offset = 0; offset < total_points; offset += points_per_chunk) {
    const size_t count = std::min(points_per_chunk, total_points - offset);
    void *ptr = base + offset * bytes_per_point;
    fn(dtype, ptr, offset, count);
  }
}

PositionsOutputInfo prepare_positions_output(const AnyTrxFile &input,
                                             const std::string &output_directory,
                                             const PrepareOutputOptions &options) {
  if (input.positions.empty() || input.offsets.empty()) {
    throw TrxFormatError("Input TRX missing positions/offsets.");
  }
  if (input.positions.cols != 3) {
    throw TrxFormatError("Positions must have 3 columns.");
  }

  std::error_code ec;
  if (trx::fs::exists(output_directory, ec)) {
    if (options.overwrite_existing) {
      trx::fs::remove_all(output_directory, ec);
    } else {
      throw TrxIOError("Output directory already exists: " + output_directory);
    }
  }
  mkdir_or_throw(output_directory);

  const std::string header_path = output_directory + SEPARATOR + "header.json";
  {
    std::ofstream out(header_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      throw TrxIOError("Failed to write header.json to: " + header_path);
    }
    out << input.header.dump() << std::endl;
  }

  write_typed_array_file(output_directory + SEPARATOR + typed_array_filename("offsets", input.offsets), input.offsets);

  if (!input.groups.empty()) {
    const std::string groups_dir = output_directory + SEPARATOR + "groups";
    trx::fs::create_directories(groups_dir, ec);
    for (const auto &kv : input.groups) {
      write_typed_array_file(groups_dir + SEPARATOR + typed_array_filename(kv.first, kv.second), kv.second);
    }
  }

  if (!input.data_per_streamline.empty()) {
    const std::string dps_dir = output_directory + SEPARATOR + "dps";
    trx::fs::create_directories(dps_dir, ec);
    for (const auto &kv : input.data_per_streamline) {
      write_typed_array_file(dps_dir + SEPARATOR + typed_array_filename(kv.first, kv.second), kv.second);
    }
  }

  if (!input.data_per_vertex.empty()) {
    const std::string dpv_dir = output_directory + SEPARATOR + "dpv";
    trx::fs::create_directories(dpv_dir, ec);
    for (const auto &kv : input.data_per_vertex) {
      write_typed_array_file(dpv_dir + SEPARATOR + typed_array_filename(kv.first, kv.second), kv.second);
    }
  }

  if (!input.data_per_group.empty()) {
    const std::string dpg_dir = output_directory + SEPARATOR + "dpg";
    trx::fs::create_directories(dpg_dir, ec);
    for (const auto &group_kv : input.data_per_group) {
      const std::string group_dir = dpg_dir + SEPARATOR + group_kv.first;
      trx::fs::create_directories(group_dir, ec);
      for (const auto &kv : group_kv.second) {
        write_typed_array_file(group_dir + SEPARATOR + typed_array_filename(kv.first, kv.second), kv.second);
      }
    }
  }

  PositionsOutputInfo info;
  info.directory = output_directory;
  info.dtype = input.positions.dtype;
  info.points = static_cast<size_t>(input.positions.rows);
  info.positions_path = output_directory + SEPARATOR + typed_array_filename("positions", input.positions);
  return info;
}

void merge_trx_shards(const MergeTrxShardsOptions &options) {
  if (options.shard_directories.empty()) {
    throw TrxArgumentError("merge_trx_shards requires at least one shard directory");
  }

  auto read_header = [](const std::string &dir) {
    const std::string path = dir + SEPARATOR + "header.json";
    std::ifstream in(path);
    if (!in.is_open()) {
      throw TrxIOError("Failed to open shard header: " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string err;
    json parsed = json::parse(ss.str(), err);
    if (!err.empty()) {
      throw TrxIOError("Failed to parse shard header " + path + ": " + err);
    }
    return parsed;
  };

  auto find_file_with_prefix = [](const std::string &dir, const std::string &prefix) -> std::string {
    std::error_code ec;
    for (trx::fs::directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
      if (ec) {
        break;
      }
      if (!it->is_regular_file()) {
        continue;
      }
      const std::string name = it->path().filename().string();
      if (name.rfind(prefix, 0) == 0) {
        return it->path().string();
      }
    }
    return "";
  };

  auto append_binary = [](const std::string &dst, const std::string &src) {
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open()) {
      throw TrxIOError("Failed to open source for append: " + src);
    }
    std::ofstream out(dst, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
      throw TrxIOError("Failed to open destination for append: " + dst);
    }
    std::vector<char> buffer(1 << 20);
    while (in) {
      in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      const auto n = in.gcount();
      if (n > 0) {
        out.write(buffer.data(), n);
      }
    }
  };

  auto append_offsets_with_base = [](const std::string &dst, const std::string &src, uint64_t base_vertices, bool skip_first) {
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open()) {
      throw TrxIOError("Failed to open source offsets: " + src);
    }
    std::ofstream out(dst, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
      throw TrxIOError("Failed to open destination offsets: " + dst);
    }
    constexpr size_t kChunkElems = (8 * 1024 * 1024) / sizeof(uint64_t);
    std::vector<uint64_t> buffer(kChunkElems);
    bool first_value_pending = skip_first;
    while (in) {
      in.read(reinterpret_cast<char *>(buffer.data()),
              static_cast<std::streamsize>(buffer.size() * sizeof(uint64_t)));
      const std::streamsize bytes = in.gcount();
      if (bytes <= 0) {
        break;
      }
      if (bytes % static_cast<std::streamsize>(sizeof(uint64_t)) != 0) {
        throw TrxFormatError("Offsets file has invalid byte count: " + src);
      }
      const size_t count = static_cast<size_t>(bytes) / sizeof(uint64_t);
      size_t start_index = 0;
      if (first_value_pending) {
        if (count == 0) {
          continue;
        }
        start_index = 1;
        first_value_pending = false;
      }
      for (size_t i = start_index; i < count; ++i) {
        buffer[i] += base_vertices;
      }
      if (count > start_index) {
        out.write(reinterpret_cast<const char *>(buffer.data() + start_index),
                  static_cast<std::streamsize>((count - start_index) * sizeof(uint64_t)));
      }
    }
  };

  auto append_group_indices_with_base = [](const std::string &dst, const std::string &src, uint32_t base_streamlines) {
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open()) {
      throw TrxIOError("Failed to open source group file: " + src);
    }
    std::ofstream out(dst, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
      throw TrxIOError("Failed to open destination group file: " + dst);
    }
    constexpr size_t kChunkElems = (8 * 1024 * 1024) / sizeof(uint32_t);
    std::vector<uint32_t> buffer(kChunkElems);
    while (in) {
      in.read(reinterpret_cast<char *>(buffer.data()),
              static_cast<std::streamsize>(buffer.size() * sizeof(uint32_t)));
      const std::streamsize bytes = in.gcount();
      if (bytes <= 0) {
        break;
      }
      if (bytes % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) {
        throw TrxFormatError("Group file has invalid byte count: " + src);
      }
      const size_t count = static_cast<size_t>(bytes) / sizeof(uint32_t);
      for (size_t i = 0; i < count; ++i) {
        buffer[i] += base_streamlines;
      }
      out.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(count * sizeof(uint32_t)));
    }
  };

  auto list_subdir_files = [](const std::string &dir, const std::string &subdir) {
    std::vector<std::string> files;
    std::error_code ec;
    const trx::fs::path path = trx::fs::path(dir) / subdir;
    if (!trx::fs::exists(path, ec)) {
      return files;
    }
    if (!trx::fs::is_directory(path, ec)) {
      throw TrxFormatError("Expected directory for subdir: " + path.string());
    }
    for (trx::fs::directory_iterator it(path, ec), end; it != end; it.increment(ec)) {
      if (ec) {
        throw TrxIOError("Failed to read directory: " + path.string());
      }
      if (!it->is_regular_file()) {
        continue;
      }
      files.push_back(it->path().filename().string());
    }
    std::sort(files.begin(), files.end());
    return files;
  };

  auto ensure_schema_match = [&](const std::string &subdir, const std::vector<std::string> &schema_files, const std::string &shard) {
    const auto shard_files = list_subdir_files(shard, subdir);
    if (shard_files != schema_files) {
      throw TrxFormatError("Shard schema mismatch for subdir '" + subdir + "': " + shard);
    }
  };

  std::error_code ec;
  for (const auto &dir : options.shard_directories) {
    if (!trx::fs::exists(dir, ec) || !trx::fs::is_directory(dir, ec)) {
      throw TrxFormatError("Shard directory does not exist: " + dir);
    }
  }

  const std::string output_dir = options.output_directory ? options.output_path : make_temp_dir("trx_merge");
  if (trx::fs::exists(output_dir, ec)) {
    if (!options.overwrite_existing) {
      throw TrxIOError("Output already exists: " + output_dir);
    }
    trx::fs::remove_all(output_dir, ec);
  }
  mkdir_or_throw(output_dir);

  for (const auto &dir : options.shard_directories) {
    if (trx::fs::exists(dir + SEPARATOR + "dpg", ec)) {
      throw TrxArgumentError("merge_trx_shards currently does not support dpg/ merges");
    }
  }

  json merged_header = read_header(options.shard_directories.front());
  const std::string first_positions = find_file_with_prefix(options.shard_directories.front(), "positions.");
  const std::string first_offsets = find_file_with_prefix(options.shard_directories.front(), "offsets.");
  if (first_positions.empty() || first_offsets.empty()) {
    throw TrxFormatError("Shard missing positions/offsets: " + options.shard_directories.front());
  }
  if (get_ext(first_offsets) != "uint64") {
    throw TrxArgumentError("merge_trx_shards currently requires offsets.uint64");
  }

  const std::string positions_filename = trx::fs::path(first_positions).filename().string();
  const std::string offsets_filename = trx::fs::path(first_offsets).filename().string();
  const std::string positions_out = output_dir + SEPARATOR + positions_filename;
  const std::string offsets_out = output_dir + SEPARATOR + offsets_filename;
  {
    std::ofstream clear_positions(positions_out, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!clear_positions.is_open()) {
      throw TrxIOError("Failed to create output positions file: " + positions_out);
    }
  }
  {
    std::ofstream clear_offsets(offsets_out, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!clear_offsets.is_open()) {
      throw TrxIOError("Failed to create output offsets file: " + offsets_out);
    }
  }

  const auto dps_schema = list_subdir_files(options.shard_directories.front(), "dps");
  const auto dpv_schema = list_subdir_files(options.shard_directories.front(), "dpv");
  const auto groups_schema = list_subdir_files(options.shard_directories.front(), "groups");
  if (!dps_schema.empty()) {
    trx::fs::create_directories(output_dir + SEPARATOR + "dps", ec);
  }
  if (!dpv_schema.empty()) {
    trx::fs::create_directories(output_dir + SEPARATOR + "dpv", ec);
  }
  if (!groups_schema.empty()) {
    trx::fs::create_directories(output_dir + SEPARATOR + "groups", ec);
  }
  for (const auto &name : dps_schema) {
    std::ofstream clear_file(output_dir + SEPARATOR + "dps" + SEPARATOR + name, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!clear_file.is_open()) {
      throw TrxIOError("Failed to create merged dps file: " + name);
    }
  }
  for (const auto &name : dpv_schema) {
    std::ofstream clear_file(output_dir + SEPARATOR + "dpv" + SEPARATOR + name, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!clear_file.is_open()) {
      throw TrxIOError("Failed to create merged dpv file: " + name);
    }
  }
  for (const auto &name : groups_schema) {
    std::ofstream clear_file(output_dir + SEPARATOR + "groups" + SEPARATOR + name,
                             std::ios::binary | std::ios::out | std::ios::trunc);
    if (!clear_file.is_open()) {
      throw TrxIOError("Failed to create merged group file: " + name);
    }
  }

  uint64_t total_vertices = 0;
  uint64_t total_streamlines = 0;
  for (size_t i = 0; i < options.shard_directories.size(); ++i) {
    const std::string &shard_dir = options.shard_directories[i];
    ensure_schema_match("dps", dps_schema, shard_dir);
    ensure_schema_match("dpv", dpv_schema, shard_dir);
    ensure_schema_match("groups", groups_schema, shard_dir);

    const json shard_header = read_header(shard_dir);
    const uint64_t shard_vertices = static_cast<uint64_t>(shard_header["NB_VERTICES"].int_value());
    const uint64_t shard_streamlines = static_cast<uint64_t>(shard_header["NB_STREAMLINES"].int_value());

    const std::string shard_positions = find_file_with_prefix(shard_dir, "positions.");
    const std::string shard_offsets = find_file_with_prefix(shard_dir, "offsets.");
    if (shard_positions.empty() || shard_offsets.empty()) {
      throw TrxFormatError("Shard missing positions/offsets: " + shard_dir);
    }
    if (trx::fs::path(shard_positions).filename().string() != positions_filename) {
      throw TrxFormatError("Shard positions dtype mismatch: " + shard_dir);
    }
    if (trx::fs::path(shard_offsets).filename().string() != offsets_filename) {
      throw TrxFormatError("Shard offsets dtype mismatch: " + shard_dir);
    }

    append_binary(positions_out, shard_positions);
    append_offsets_with_base(offsets_out, shard_offsets, total_vertices, i != 0);

    for (const auto &name : dps_schema) {
      append_binary(output_dir + SEPARATOR + "dps" + SEPARATOR + name, shard_dir + SEPARATOR + "dps" + SEPARATOR + name);
    }
    for (const auto &name : dpv_schema) {
      append_binary(output_dir + SEPARATOR + "dpv" + SEPARATOR + name, shard_dir + SEPARATOR + "dpv" + SEPARATOR + name);
    }
    for (const auto &name : groups_schema) {
      if (total_streamlines > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw TrxFormatError("Group index offset exceeds uint32 range during merge");
      }
      append_group_indices_with_base(
          output_dir + SEPARATOR + "groups" + SEPARATOR + name,
          shard_dir + SEPARATOR + "groups" + SEPARATOR + name,
          static_cast<uint32_t>(total_streamlines));
    }

    total_vertices += shard_vertices;
    total_streamlines += shard_streamlines;
  }

  merged_header = _json_set(merged_header, "NB_VERTICES", static_cast<int>(total_vertices));
  merged_header = _json_set(merged_header, "NB_STREAMLINES", static_cast<int>(total_streamlines));
  {
    const std::string merged_header_path = output_dir + SEPARATOR + "header.json";
    std::ofstream out(merged_header_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      throw TrxIOError("Failed to write merged header: " + merged_header_path);
    }
    out << merged_header.dump() << std::endl;
  }

  if (options.output_directory) {
    return;
  }

  const trx::fs::path archive_path(options.output_path);
  if (archive_path.has_parent_path()) {
    mkdir_or_throw(archive_path.parent_path().string());
  }

  int errorp = 0;
  detail::ZipArchive zf(zip_open(options.output_path.c_str(), ZIP_CREATE + ZIP_TRUNCATE, &errorp));
  if (!zf) {
    throw TrxIOError("Could not open archive " + options.output_path + ": " + strerror(errorp));
  }
  zip_from_folder(zf.get(), output_dir, output_dir, options.compression_standard, nullptr);
  zf.commit(options.output_path);
  rm_dir(output_dir);
}
}; // namespace trx