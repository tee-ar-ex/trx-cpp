#include <trx/trx.h>

namespace trxmmap {
namespace detail {

int _sizeof_dtype(const std::string &dtype) {
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
  return sizeof(std::uint16_t); // default to 16-bit float size
}

std::string _get_dtype(const std::string &dtype) {
  const char dt = dtype.back();
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
  case 'y': // unsigned long long (Itanium ABI)
    return "uint64";
  case 'a':
    return "int8";
  case 's':
    return "int16";
  case 'i':
    return "int32";
  case 'l':
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

bool _is_dtype_valid(const std::string &ext) {
  if (ext == "bit")
    return true;
  if (std::find(::trxmmap::dtypes.begin(), ::trxmmap::dtypes.end(), ext) != ::trxmmap::dtypes.end())
    return true;
  return false;
}

std::tuple<std::string, int, std::string> _split_ext_with_dimensionality(const std::string &filename) {
  std::string base = ::trxmmap::path_basename(filename);

  const size_t num_splits = std::count(base.begin(), base.end(), '.');
  int dim = 0;

  if (num_splits != 1 && num_splits != 2) {
    throw std::invalid_argument("Invalid filename");
  }

  const std::string ext = ::trxmmap::get_ext(base);

  base = base.substr(0, base.length() - ext.length() - 1);

  if (num_splits == 1) {
    dim = 1;
  } else {
    const size_t pos = base.find_last_of('.');
    dim = std::stoi(base.substr(pos + 1, base.size()));
    base = base.substr(0, pos);
  }

  const bool is_valid = _is_dtype_valid(ext);

  if (!is_valid) {
    // TODO: make formatted string and include provided extension name
    throw std::invalid_argument("Unsupported file extension");
  }

  std::tuple<std::string, int, std::string> output{base, dim, ext};

  return output;
}

} // namespace detail
} // namespace trxmmap
