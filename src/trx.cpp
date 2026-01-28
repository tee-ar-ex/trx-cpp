#include <algorithm>
#include <cstring>
#include <errno.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <trx/trx.h>
#include <tuple>
#include <vector>
#include <zip.h>

// #define ZIP_DD_SIG 0x08074b50
// #define ZIP_CD_SIG 0x06054b50
using namespace Eigen;
using namespace std;

namespace trxmmap {
namespace {
inline int sys_error() {
  return errno;
}

std::string normalize_slashes(std::string path) {
  std::replace(path.begin(), path.end(), '\\', '/');
  return path;
}

bool parse_positions_dtype(const std::string &filename, std::string &out_dtype) {
  const std::string normalized = normalize_slashes(filename);
  try {
    const auto tuple = trxmmap::_split_ext_with_dimensionality(normalized);
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
} // namespace

std::string detect_positions_dtype(const std::string &path) {
  const trx::fs::path input(path);
  if (!trx::fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + path);
  }

  std::error_code ec;
  if (trx::fs::is_directory(input, ec) && !ec) {
    std::map<std::string, std::tuple<long long, long long>> files;
    trxmmap::populate_fps(path.c_str(), files);
    for (const auto &kv : files) {
      std::string dtype;
      if (parse_positions_dtype(kv.first, dtype)) {
        return dtype;
      }
    }
    return "";
  }

  int err = 0;
  zip_t *zf = zip_open(path.c_str(), 0, &err);
  if (zf == nullptr) {
    throw std::runtime_error("Could not open zip file: " + path);
  }
  std::string dtype;
  const zip_int64_t count = zip_get_num_entries(zf, 0);
  for (zip_int64_t i = 0; i < count; ++i) {
  const char *name = zip_get_name(zf, i, 0);
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

void populate_fps(const char *name, std::map<std::string, std::tuple<long long, long long>> &files_pointer_size) {
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

    if (strcmp(ext.c_str(), "json") == 0) {
      continue;
    }

    if (!_is_dtype_valid(ext)) {
      throw std::invalid_argument(std::string("The dtype of ") + elem_filename + std::string(" is not supported"));
    }

    if (strcmp(ext.c_str(), "bit") == 0) {
      ext = "bool";
    }

    const int dtype_size = _sizeof_dtype(ext);
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
  std::string ext = "";
  std::string delimeter = ".";

  if (str.rfind(delimeter) < str.length() - 1) {
    ext = str.substr(str.rfind(delimeter) + 1);
  }
  return ext;
}

bool _is_path_within(const trx::fs::path &child, const trx::fs::path &parent) {
  std::string parent_str = parent.lexically_normal().string();
  std::string child_str = child.lexically_normal().string();
  if (parent_str.empty())
    return false;
  if (child_str.size() < parent_str.size())
    return false;
  if (child_str.compare(0, parent_str.size(), parent_str) != 0)
    return false;
  if (child_str.size() == parent_str.size())
    return true;
  char next = child_str[parent_str.size()];
  return next == '/' || next == '\\';
}
// TODO: check if there's a better way
int _sizeof_dtype(std::string dtype) {
  if (dtype == "bit")
    return 1;
  if (dtype == "uint8")
    return sizeof(uint8_t);
  // Treat "ushort" as an alias of uint16 for cross-platform consistency.
  if (dtype == "uint16" || dtype == "ushort")
    return sizeof(uint16_t);
  if (dtype == "uint32")
    return sizeof(uint32_t);
  if (dtype == "uint64")
    return sizeof(uint64_t);
  if (dtype == "int8")
    return sizeof(int8_t);
  if (dtype == "int16")
    return sizeof(int16_t);
  if (dtype == "int32")
    return sizeof(int32_t);
  if (dtype == "int64")
    return sizeof(int64_t);
  if (dtype == "float32")
    return sizeof(float);
  if (dtype == "float64")
    return sizeof(double);
  return sizeof(half); // setting this as default for now but a better solution is needed
}

std::string _get_dtype(std::string dtype) {
  char dt = dtype.back();
  switch (dt) {
  case 'b':
    return "bit";
  case 'h':
    return "uint8";
  case 't':
    return "uint16";
  case 'j':
    return "uint32";
  case 'm':
    return "uint64";
  case 'y': // unsigned long long (Itanium ABI)
    return "uint64";
  case 'a':
    return "int8";
  case 's':
    return "int16";
  case 'i':
    return "int32";
  case 'l':
    return "int64";
  case 'x': // long long (Itanium ABI)
    return "int64";
  case 'f':
    return "float32";
  case 'd':
    return "float64";
  default:
    return "float16"; // setting this as default for now but a better solution is needed
  }
}
std::tuple<std::string, int, std::string> _split_ext_with_dimensionality(const std::string filename) {
  std::string base = path_basename(filename);

  size_t num_splits = std::count(base.begin(), base.end(), '.');
  int dim;

  if (num_splits != 1 && num_splits != 2) {
    throw std::invalid_argument("Invalid filename");
  }

  std::string ext = get_ext(base);

  base = base.substr(0, base.length() - ext.length() - 1);

  if (num_splits == 1) {
    dim = 1;
  } else {
    size_t pos = base.find_last_of(".");
    dim = std::stoi(base.substr(pos + 1, base.size()));
    base = base.substr(0, pos);
  }

  bool is_valid = _is_dtype_valid(ext);

  if (is_valid == false) {
    // TODO: make formatted string and include provided extension name
    throw std::invalid_argument("Unsupported file extension");
  }

  std::tuple<std::string, int, std::string> output{base, dim, ext};

  return output;
}

bool _is_dtype_valid(std::string &ext) {
  if (ext.compare("bit") == 0)
    return true;
  if (std::find(trxmmap::dtypes.begin(), trxmmap::dtypes.end(), ext) != trxmmap::dtypes.end())
    return true;
  return false;
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
  int buff_len = 255 * sizeof(char);
  char *buffer = (char *)malloc(buff_len);

  std::string jstream = "";
  zip_int64_t nbytes;
  while ((nbytes = zip_fread(zh, buffer, buff_len - 1)) > 0) {
    if (buffer != NULL) {
      jstream += string(buffer, nbytes);
    }
  }

  zip_fclose(zh);
  free(buffer);

  // convert jstream data into Json.
  std::string err;
  auto root = json::parse(jstream, err);
  if (!err.empty()) {
    throw std::runtime_error("Failed to parse header.json: " + err);
  }
  return root;
}

void allocate_file(const std::string &path, const int size) {
  std::ofstream file(path);
  if (file.is_open()) {
    std::string s(size, '\0');
    file << s;
    file.flush();
    file.close();
  } else {
    std::cerr << "Failed to allocate file : " << sys_error() << std::endl;
  }
}

mio::shared_mmap_sink _create_memmap(
    std::string filename, std::tuple<int, int> &shape, std::string mode, std::string dtype, long long offset) {
  if (dtype.compare("bool") == 0) {
    std::string ext = "bit";
    filename.replace(filename.size() - 4, 3, ext);
    filename.pop_back();
  }

  long filesize = std::get<0>(shape) * std::get<1>(shape) * _sizeof_dtype(dtype);
  // if file does not exist, create and allocate it

  struct stat buffer;
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

json assignHeader(json root) {
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

void get_reference_info(std::string reference, const MatrixXf &affine, const RowVectorXi &dimensions) {
  // TODO: find a library to use for nifti and trk (MRtrix??)
  //  if (reference.find(".nii") != std::string::npos)
  //  {
  //  }
  if (reference.find(".trk") != std::string::npos) {
    // TODO: Create exception class
    std::cout << "Trk reference not implemented" << std::endl;
    std::exit(1);
  } else {
    // TODO: Create exception class
    std::cout << "Trk reference not implemented" << std::endl;
    std::exit(1);
  }
}

void copy_dir(const char *src, const char *dst) {
  trx::fs::path src_path(src);
  trx::fs::path dst_path(dst);
  std::error_code ec;
  if (!trx::fs::exists(src_path, ec) || !trx::fs::is_directory(src_path, ec)) {
    return;
  }

  if (!trx::fs::create_directories(dst_path, ec) && ec) {
    throw std::runtime_error(std::string("Could not create directory ") + dst);
  }

  ec.clear();
  for (trx::fs::recursive_directory_iterator it(src_path, ec), end; it != end; it.increment(ec)) {
    if (ec) {
      throw std::runtime_error("Failed to read directory: " + src_path.string());
    }
    const trx::fs::path current = it->path();
    const trx::fs::path rel = current.lexically_relative(src_path);
    const trx::fs::path target = dst_path / rel;
    std::error_code entry_ec;
    if (it->is_directory(entry_ec)) {
      trx::fs::create_directories(target, entry_ec);
      if (entry_ec) {
        throw std::runtime_error("Could not create directory " + target.string());
      }
      continue;
    }
    if (!it->is_regular_file(entry_ec)) {
      continue;
    }
    copy_file(current.string().c_str(), target.string().c_str());
  }
}

void copy_file(const char *src, const char *dst) {
  std::ifstream in(src, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error(std::string("Failed to open source file ") + src);
  }
  std::ofstream out(dst, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error(std::string("Failed to open destination file ") + dst);
  }

  char buffer[4096];
  while (in) {
    in.read(buffer, sizeof(buffer));
    std::streamsize n = in.gcount();
    if (n > 0) {
      out.write(buffer, n);
      if (!out) {
        throw std::runtime_error(std::string("Error writing to file ") + dst);
      }
    }
  }
  if (!in.eof()) {
    throw std::runtime_error(std::string("Error reading file ") + src);
  }
}
int rm_dir(const char *d) {
  std::error_code ec;
  trx::fs::remove_all(d, ec);
  return ec ? -1 : 0;
}

std::string make_temp_dir(const std::string &prefix) {
  const char *env_tmp = std::getenv("TRX_TMPDIR");
  std::string base_dir;

  if (env_tmp != nullptr) {
    std::string val(env_tmp);
    if (val == "use_working_dir") {
      base_dir = ".";
    } else {
      trx::fs::path env_path(val);
      std::error_code ec;
      if (trx::fs::exists(env_path, ec) && trx::fs::is_directory(env_path, ec)) {
        base_dir = env_path.string();
      }
    }
  }

  if (base_dir.empty()) {
    const char *candidates[] = {std::getenv("TMPDIR"), std::getenv("TEMP"), std::getenv("TMP")};
    for (const char *candidate : candidates) {
      if (candidate == nullptr || std::string(candidate).empty()) {
        continue;
      }
      trx::fs::path path(candidate);
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

  trx::fs::path base_path(base_dir);
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
    trx::fs::path candidate = base_path / (prefix + "_" + std::to_string(dist(rng)));
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
  std::string root_dir = make_temp_dir("trx_zip");
  trx::fs::path normalized_root = trx::fs::path(root_dir).lexically_normal();

  zip_int64_t num_entries = zip_get_num_entries(zfolder, ZIP_FL_UNCHANGED);
  for (zip_int64_t i = 0; i < num_entries; ++i) {
    const char *entry_name = zip_get_name(zfolder, i, ZIP_FL_UNCHANGED);
    if (entry_name == nullptr) {
      continue;
    }
    std::string entry(entry_name);

    trx::fs::path entry_path(entry);
    if (entry_path.is_absolute()) {
      throw std::runtime_error("Zip entry has absolute path: " + entry);
    }

    trx::fs::path normalized_entry = entry_path.lexically_normal();
    trx::fs::path out_path = normalized_root / normalized_entry;
    trx::fs::path normalized_out = out_path.lexically_normal();

    if (!_is_path_within(normalized_out, normalized_root)) {
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

    char buffer[4096];
    zip_int64_t nbytes = 0;
    while ((nbytes = zip_fread(zf, buffer, sizeof(buffer))) > 0) {
      out.write(buffer, nbytes);
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
                     const std::string root,
                     const std::string directory,
                     zip_uint32_t compression_standard) {
  std::error_code ec;
  for (trx::fs::recursive_directory_iterator it(directory, ec), end; it != end; it.increment(ec)) {
    if (ec) {
      throw std::runtime_error("Failed to read directory: " + directory);
    }
    const trx::fs::path current = it->path();
    std::string zip_fname = rm_root(root, current.string());
    std::error_code entry_ec;
    if (it->is_directory(entry_ec)) {
      zip_dir_add(zf, zip_fname.c_str(), ZIP_FL_ENC_GUESS);
      continue;
    }
    if (!it->is_regular_file(entry_ec)) {
      continue;
    }

    const std::string fullpath = current.string();
    zip_source_t *s;
    zip_int64_t file_idx = -1;
    if ((s = zip_source_file(zf, fullpath.c_str(), 0, 0)) == NULL ||
        (file_idx = zip_file_add(zf, zip_fname.c_str(), s, ZIP_FL_ENC_UTF_8)) < 0) {
      zip_source_free(s);
      throw std::runtime_error(std::string("Error adding file ") + zip_fname + ": " + zip_strerror(zf));
    } else if (zip_set_file_compression(zf, file_idx, compression_standard, 0) < 0) {
      throw std::runtime_error(std::string("Error setting compression for ") + zip_fname + ": " + zip_strerror(zf));
    }
  }
}

std::string rm_root(const std::string root, const std::string path) {

  std::size_t index;
  std::string stripped;

  index = path.find(root);
  if (index != std::string::npos) {
    stripped = path.substr(index + root.size() + 1, path.size() - index - root.size() - 1);
  }
  return stripped;
}
}; // namespace trxmmap