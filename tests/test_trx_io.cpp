#include <filesystem>
#include <gtest/gtest.h>
#include <trx/trx.h>
#include <zip.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace Eigen;
using namespace trxmmap;
namespace fs = std::filesystem;

namespace {
std::string get_test_data_root() {
  const auto *env = std::getenv("TRX_TEST_DATA_DIR");
  if (env == nullptr || std::string(env).empty()) {
    return {};
  }
  return std::string(env);
}

fs::path resolve_gold_standard_dir(const std::string &root_dir) {
  fs::path root(root_dir);
  fs::path gs_dir = root / "gold_standard";
  if (fs::exists(gs_dir)) {
    return gs_dir;
  }
  return root;
}

fs::path require_gold_standard_dir() {
  const auto root = get_test_data_root();
  if (root.empty()) {
    throw std::runtime_error("TRX_TEST_DATA_DIR not set");
  }

  const auto gs_dir = resolve_gold_standard_dir(root);
  const fs::path gs_trx = gs_dir / "gs.trx";
  const fs::path gs_dir_trx = gs_dir / "gs_fldr.trx";
  const fs::path gs_coords = gs_dir / "gs_rasmm_space.txt";
  if (!fs::exists(gs_trx)) {
    throw std::runtime_error("Missing gold_standard gs.trx");
  }
  if (!fs::exists(gs_dir_trx)) {
    throw std::runtime_error("Missing gold_standard gs_fldr.trx");
  }
  if (!fs::exists(gs_coords)) {
    throw std::runtime_error("Missing gold_standard gs_rasmm_space.txt");
  }
  return gs_dir;
}

fs::path make_temp_test_dir(const std::string &prefix) {
  std::error_code ec;
  auto base = fs::temp_directory_path(ec);
  if (ec) {
    throw std::runtime_error("Failed to get temp directory: " + ec.message());
  }

  static std::mt19937_64 rng(std::random_device{}());
  std::uniform_int_distribution<uint64_t> dist;

  for (int attempt = 0; attempt < 100; ++attempt) {
    fs::path candidate = base / (prefix + "_" + std::to_string(dist(rng)));
    std::error_code dir_ec;
    if (fs::create_directory(candidate, dir_ec)) {
      return candidate;
    }
    if (dir_ec && dir_ec != std::errc::file_exists) {
      throw std::runtime_error("Failed to create temporary directory: " + dir_ec.message());
    }
  }
  throw std::runtime_error("Unable to create unique temporary directory");
}

std::string normalize_path(const std::string &path) {
  std::string out = path;
  std::replace(out.begin(), out.end(), '\\', '/');
  return out;
}

std::string normalize_path(const fs::path &path) { return normalize_path(path.string()); }

bool is_dir(const fs::path &path) {
  std::error_code ec;
  return fs::is_directory(path, ec) && !ec;
}

bool is_regular(const fs::path &path) {
  std::error_code ec;
  return fs::is_regular_file(path, ec) && !ec;
}

Matrix<float, Dynamic, Dynamic, RowMajor> load_rasmm_coords(const fs::path &path) {
  std::ifstream in(path.string());
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open coordinate file: " + path.string());
  }

  std::vector<float> values;
  float v;
  while (in >> v) {
    values.push_back(v);
  }
  if (values.size() % 3 != 0) {
    throw std::runtime_error("Coordinate file does not contain triples of floats.");
  }

  const Eigen::Index rows = static_cast<Eigen::Index>(values.size() / 3);
  Matrix<float, Dynamic, Dynamic, RowMajor> coords(rows, 3);
  for (Eigen::Index i = 0; i < rows; ++i) {
    for (Eigen::Index j = 0; j < 3; ++j) {
      coords(i, j) = values[static_cast<size_t>(i * 3 + j)];
    }
  }
  return coords;
}

void expect_allclose(const Matrix<float, Dynamic, Dynamic, RowMajor> &actual,
                     const Matrix<float, Dynamic, Dynamic, RowMajor> &expected,
                     float rtol = 1e-4f,
                     float atol = 1e-6f) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (Eigen::Index i = 0; i < actual.rows(); ++i) {
    for (Eigen::Index j = 0; j < actual.cols(); ++j) {
      const float a = actual(i, j);
      const float b = expected(i, j);
      const float tol = atol + rtol * std::abs(b);
      EXPECT_LE(std::abs(a - b), tol);
    }
  }
}

template <typename DT> std::unique_ptr<trxmmap::TrxFile<DT>> load_trx(const fs::path &path) {
  if (is_dir(path)) {
    return trxmmap::load_from_directory<DT>(path.string());
  }
  return trxmmap::load_from_zip<DT>(path.string());
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const std::string &name, const std::string &value) : name_(name) {
    const auto *existing = std::getenv(name_.c_str());
    if (existing != nullptr) {
      had_value_ = true;
      previous_ = existing;
    }
    set(value);
  }

  ~ScopedEnvVar() {
    if (had_value_) {
      set(previous_);
    } else {
      unset();
    }
  }

private:
  void set(const std::string &value) {
#if defined(_WIN32) || defined(_WIN64)
    _putenv_s(name_.c_str(), value.c_str());
#else
    setenv(name_.c_str(), value.c_str(), 1);
#endif
  }

  void unset() {
#if defined(_WIN32) || defined(_WIN64)
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }

  std::string name_;
  bool had_value_ = false;
  std::string previous_;
};

std::string get_current_working_dir() {
  std::error_code ec;
  auto cwd = fs::current_path(ec);
  if (ec) {
    throw std::runtime_error("Failed to get current working directory: " + ec.message());
  }
  return cwd.string();
}

bool wait_for_path_gone(const fs::path &path, int retries = 10, int delay_ms = 50) {
  for (int i = 0; i < retries; ++i) {
    if (!fs::exists(path)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }
  return !fs::exists(path);
}

void write_text_file(const fs::path &path, const std::string &contents) {
  std::ofstream out(path.string());
  if (!out.is_open()) {
    throw std::runtime_error("Failed to write text file: " + path.string());
  }
  out << contents;
  out.close();
}

void write_tsf_text_file(const fs::path &path, const std::string &data_stream) {
  std::ostringstream out;
  out << "mrtrix track scalars\n";
  out << "file: .\n";
  out << "datatype: Float32LE\n";
  out << "timestamp: 0\n";
  out << "END\n";
  out << data_stream;
  write_text_file(path, out.str());
}

std::string build_tsf_contents(const std::vector<std::vector<double>> &streamlines) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  const double nan_value = std::numeric_limits<double>::quiet_NaN();
  const double inf_value = std::numeric_limits<double>::infinity();
  bool first = true;
  for (size_t s = 0; s < streamlines.size(); ++s) {
    const auto &values = streamlines[s];
    for (size_t i = 0; i < values.size(); ++i) {
      if (!first) {
        out << ' ';
      }
      out << values[i];
      first = false;
    }
    if (s + 1 < streamlines.size()) {
      if (!first) {
        out << ' ';
      }
      out << nan_value;
      first = false;
    }
  }
  if (!first) {
    out << ' ';
  }
  out << inf_value;
  return out.str();
}

struct TsfHeader {
  size_t offset = 0;
  std::string timestamp;
  std::string datatype;
  size_t count = 0;
  size_t total_count = 0;
};

TsfHeader read_tsf_header(const fs::path &path) {
  std::ifstream in(path.string(), std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open TSF file: " + path.string());
  }

  TsfHeader header;
  std::string line;
  while (std::getline(in, line)) {
    if (line == "END") {
      break;
    }
    const auto pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, pos);
    const std::string value = line.substr(pos + 1);
    if (key == "timestamp") {
      header.timestamp = value;
      header.timestamp.erase(0, header.timestamp.find_first_not_of(' '));
    } else if (key == "datatype") {
      header.datatype = value;
      header.datatype.erase(0, header.datatype.find_first_not_of(' '));
    } else if (key == "file") {
      std::istringstream iss(value);
      std::string dot;
      iss >> dot >> header.offset;
    } else if (key == "count") {
      std::string trimmed = value;
      trimmed.erase(0, trimmed.find_first_not_of(' '));
      header.count = static_cast<size_t>(std::stoull(trimmed));
    } else if (key == "total_count") {
      std::string trimmed = value;
      trimmed.erase(0, trimmed.find_first_not_of(' '));
      header.total_count = static_cast<size_t>(std::stoull(trimmed));
    }
  }
  return header;
}

std::vector<float> read_tsf_float32_data(const fs::path &path, size_t offset, size_t count) {
  std::ifstream in(path.string(), std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open TSF file: " + path.string());
  }
  in.seekg(static_cast<std::streamoff>(offset));
  std::vector<float> values(count);
  in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
  return values;
}

class ScopedLocale {
public:
  explicit ScopedLocale(const std::locale &loc) : previous_(std::locale::global(loc)) {}

  ~ScopedLocale() { std::locale::global(previous_); }

private:
  std::locale previous_;
};

void set_streamline_lengths(ArraySequence<float> *streamlines, const std::vector<uint32_t> &lengths) {
  // Intentional white-box access: ArraySequence has no public API for setting
  // lengths/offsets in tests, and we need consistent metadata for fixtures.
  streamlines->_lengths.resize(static_cast<Eigen::Index>(lengths.size()));
  uint64_t offset = 0;
  for (Eigen::Index i = 0; i < streamlines->_lengths.size(); ++i) {
    streamlines->_lengths(i) = lengths[static_cast<size_t>(i)];
    streamlines->_offsets(i) = offset;
    offset += lengths[static_cast<size_t>(i)];
  }
  streamlines->_offsets(streamlines->_lengths.size()) = offset;
}
} // namespace

TEST(TrxFileIo, load_rasmm) {
  const auto gs_dir = require_gold_standard_dir();
  const auto coords = load_rasmm_coords(gs_dir / "gs_rasmm_space.txt");

  const std::vector<fs::path> inputs = {gs_dir / "gs.trx", gs_dir / "gs_fldr.trx"};
  for (const auto &input : inputs) {
    ASSERT_TRUE(fs::exists(input));
  auto trx = load_trx<float>(input);
    Matrix<float, Dynamic, Dynamic, RowMajor> actual = trx->streamlines->_data;
    expect_allclose(actual, coords);
    trx->close();
  }
}

TEST(TrxFileIo, multi_load_save_rasmm) {
  const auto gs_dir = require_gold_standard_dir();
  const auto coords = load_rasmm_coords(gs_dir / "gs_rasmm_space.txt");

  const std::vector<fs::path> inputs = {gs_dir / "gs.trx", gs_dir / "gs_fldr.trx"};
  for (const auto &input : inputs) {
    ASSERT_TRUE(fs::exists(input));
    fs::path tmp_dir = make_temp_test_dir("trx_gs");

    auto trx = load_trx<float>(input);
    const std::string input_str = normalize_path(input.string());
    const std::string basename = trxmmap::get_base("/", input_str);
    const std::string ext = trxmmap::get_ext(input_str);
    const std::string basename_no_ext = ext.empty() ? basename : basename.substr(0, basename.size() - ext.size() - 1);

    for (int i = 0; i < 3; ++i) {
      fs::path out_path = tmp_dir / (basename_no_ext + "_tmp" + std::to_string(i) + (ext.empty() ? "" : ("." + ext)));
      trxmmap::save(*trx, out_path.string());
      trx->close();
      trx = load_trx<float>(out_path);
    }

    Matrix<float, Dynamic, Dynamic, RowMajor> actual = trx->streamlines->_data;
    expect_allclose(actual, coords);
    trx->close();

    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
  }
}

TEST(TrxFileIo, delete_tmp_gs_dir_rasmm) {
  const auto gs_dir = require_gold_standard_dir();
  const auto coords = load_rasmm_coords(gs_dir / "gs_rasmm_space.txt");

  const std::vector<fs::path> inputs = {gs_dir / "gs.trx", gs_dir / "gs_fldr.trx"};
  for (const auto &input : inputs) {
    ASSERT_TRUE(fs::exists(input));
    auto trx = load_trx<float>(input);

    std::string tmp_dir = trx->_uncompressed_folder_handle;
    if (is_regular(input)) {
      ASSERT_FALSE(tmp_dir.empty());
      ASSERT_TRUE(fs::exists(tmp_dir));
    }

    Matrix<float, Dynamic, Dynamic, RowMajor> actual = trx->streamlines->_data;
    expect_allclose(actual, coords);
    trx->close();

    if (is_regular(input)) {
#if defined(_WIN32) || defined(_WIN64)
      // Windows can hold file handles briefly after close; avoid flaky removal assertions.
      (void)wait_for_path_gone(tmp_dir);
#else
      EXPECT_TRUE(wait_for_path_gone(tmp_dir));
#endif
    }

    trx = load_trx<float>(input);
    Matrix<float, Dynamic, Dynamic, RowMajor> actual2 = trx->streamlines->_data;
    expect_allclose(actual2, coords);
    trx->close();
  }
}

TEST(TrxFileIo, close_tmp_files) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path input = gs_dir / "gs.trx";
  ASSERT_TRUE(fs::exists(input));

  auto trx = load_trx<float>(input);
  const std::string tmp_dir = trx->_uncompressed_folder_handle;
  ASSERT_FALSE(tmp_dir.empty());
  ASSERT_TRUE(fs::exists(tmp_dir));

  const std::vector<fs::path> expected_paths = {"offsets.uint32",
                                                "positions.3.float32",
                                                "header.json",
                                                "dps/random_coord.3.float32",
                                                "dpv/color_y.float32",
                                                "dpv/color_x.float32",
                                                "dpv/color_z.float32"};

  for (const auto &rel_path : expected_paths) {
    EXPECT_TRUE(fs::exists(fs::path(tmp_dir) / rel_path));
  }

  trx->close();

#if defined(_WIN32) || defined(_WIN64)
  // Windows can hold file handles briefly after close; avoid flaky removal assertions.
  (void)wait_for_path_gone(tmp_dir);
#else
  EXPECT_TRUE(wait_for_path_gone(tmp_dir));
#endif
}

TEST(TrxFileIo, change_tmp_dir) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path input = gs_dir / "gs.trx";
  ASSERT_TRUE(fs::exists(input));

  const auto *home_env = std::getenv("HOME");
#if defined(_WIN32) || defined(_WIN64)
  if (home_env == nullptr || std::string(home_env).empty()) {
    home_env = std::getenv("USERPROFILE");
  }
#endif
  if (home_env == nullptr || std::string(home_env).empty()) {
    GTEST_SKIP() << "No HOME/USERPROFILE set for TRX_TMPDIR test";
  }

  {
    ScopedEnvVar env("TRX_TMPDIR", "use_working_dir");
    auto trx = load_trx<float>(input);
    fs::path tmp_dir = trx->_uncompressed_folder_handle;
    fs::path parent = tmp_dir.parent_path();
    fs::path expected = fs::path(get_current_working_dir());
    std::string parent_norm = normalize_path(parent.lexically_normal());
    if (parent_norm == ".") {
      parent_norm = normalize_path(expected.lexically_normal());
    }
    EXPECT_EQ(parent_norm, normalize_path(expected.lexically_normal()));
    trx->close();
  }

  {
    ScopedEnvVar env("TRX_TMPDIR", home_env);
    auto trx = load_trx<float>(input);
    fs::path tmp_dir = trx->_uncompressed_folder_handle;
    fs::path parent = tmp_dir.parent_path();
    fs::path expected = fs::path(std::string(home_env));
    EXPECT_EQ(normalize_path(parent.lexically_normal()), normalize_path(expected.lexically_normal()));
    trx->close();
  }
}

TEST(TrxFileIo, complete_dir_from_trx) {
  const auto gs_dir = require_gold_standard_dir();

  const std::set<std::string> expected_content = {"offsets.uint32",
                                                  "positions.3.float32",
                                                  "header.json",
                                                  "dps/random_coord.3.float32",
                                                  "dpv/color_y.float32",
                                                  "dpv/color_x.float32",
                                                  "dpv/color_z.float32"};

  const std::vector<fs::path> inputs = {gs_dir / "gs.trx", gs_dir / "gs_fldr.trx"};
  for (const auto &input : inputs) {
    ASSERT_TRUE(fs::exists(input));
    auto trx = load_trx<float>(input);
    fs::path dir_to_check =
        trx->_uncompressed_folder_handle.empty() ? input : fs::path(trx->_uncompressed_folder_handle);

    std::set<std::string> file_paths;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir_to_check, ec), end; it != end; it.increment(ec)) {
      if (ec) {
        break;
      }
      if (!it->is_regular_file(ec)) {
        ec.clear();
        continue;
      }
      fs::path rel = it->path().lexically_relative(dir_to_check);
      file_paths.insert(normalize_path(rel.string()));
    }

    EXPECT_EQ(file_paths, expected_content);
    trx->close();
  }
}

TEST(TrxFileIo, complete_zip_from_trx) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path input = gs_dir / "gs.trx";
  ASSERT_TRUE(fs::exists(input));

  const std::set<std::string> expected_content = {"offsets.uint32",
                                                  "positions.3.float32",
                                                  "header.json",
                                                  "dps/random_coord.3.float32",
                                                  "dpv/color_y.float32",
                                                  "dpv/color_x.float32",
                                                  "dpv/color_z.float32"};

  int errorp = 0;
  zip_t *zf = zip_open(input.string().c_str(), 0, &errorp);
  ASSERT_NE(zf, nullptr);

  std::set<std::string> zip_file_list;
  zip_int64_t num_entries = zip_get_num_entries(zf, ZIP_FL_UNCHANGED);
  for (zip_int64_t i = 0; i < num_entries; ++i) {
    const auto *entry_name = zip_get_name(zf, i, ZIP_FL_UNCHANGED);
    if (entry_name == nullptr) {
      continue;
    }
    std::string name(entry_name);
    if (!name.empty() && name.back() == '/') {
      continue;
    }
    zip_file_list.insert(name);
  }
  zip_close(zf);

  EXPECT_EQ(zip_file_list, expected_content);
}

TEST(TrxFileIo, add_dps_from_text_success) {
  trxmmap::TrxFile<float> trx(4, 2);
  set_streamline_lengths(trx.streamlines.get(), {2, 2});

  const fs::path tmp_dir = make_temp_test_dir("trx_dps_text");
  const fs::path input_path = tmp_dir / "dps.txt";
  write_text_file(input_path, "0.25 0.75");

  trxmmap::add_dps_from_text(trx, "weight", "float32", input_path.string());
  auto it = trx.data_per_streamline.find("weight");
  ASSERT_NE(it, trx.data_per_streamline.end());
  EXPECT_EQ(it->second->_matrix.rows(), 2);
  EXPECT_EQ(it->second->_matrix.cols(), 1);
  EXPECT_FLOAT_EQ(it->second->_matrix(0, 0), 0.25F);
  EXPECT_FLOAT_EQ(it->second->_matrix(1, 0), 0.75F);
}

TEST(TrxFileIo, add_dps_from_text_errors) {
  trxmmap::TrxFile<float> trx(4, 2);
  set_streamline_lengths(trx.streamlines.get(), {2, 2});

  const fs::path tmp_dir = make_temp_test_dir("trx_dps_text_err");
  const fs::path input_path = tmp_dir / "dps.txt";
  write_text_file(input_path, "1.0");

  EXPECT_THROW(trxmmap::add_dps_from_text(trx, "", "float32", input_path.string()), std::invalid_argument);
  EXPECT_THROW(trxmmap::add_dps_from_text(trx, "weight", "badtype", input_path.string()), std::invalid_argument);
  EXPECT_THROW(trxmmap::add_dps_from_text(trx, "weight", "int32", input_path.string()), std::invalid_argument);

  EXPECT_THROW(trxmmap::add_dps_from_text(trx, "weight", "float32", (tmp_dir / "missing.txt").string()),
               std::runtime_error);

  write_text_file(input_path, "1.0 abc");
  EXPECT_THROW(trxmmap::add_dps_from_text(trx, "weight", "float32", input_path.string()), std::runtime_error);

  write_text_file(input_path, "1.0");
  EXPECT_THROW(trxmmap::add_dps_from_text(trx, "weight", "float32", input_path.string()), std::runtime_error);

  trxmmap::TrxFile<float> empty;
  EXPECT_THROW(trxmmap::add_dps_from_text(empty, "weight", "float32", input_path.string()), std::runtime_error);
}

TEST(TrxFileIo, add_dpv_from_tsf_success) {
  ScopedLocale scoped_locale(std::locale::classic());
  trxmmap::TrxFile<float> source_trx(4, 2);
  set_streamline_lengths(source_trx.streamlines.get(), {2, 2});

  const fs::path tmp_dir = make_temp_test_dir("trx_dpv_tsf");
  const fs::path input_path = tmp_dir / "dpv_text.tsf";
  write_tsf_text_file(input_path, build_tsf_contents({{0.1, 0.2}, {0.3, 0.4}}));
  trxmmap::add_dpv_from_tsf(source_trx, "signal", "float32", input_path.string());

  const fs::path binary_path = tmp_dir / "dpv_binary.tsf";
  trxmmap::export_dpv_to_tsf(source_trx, "signal", binary_path.string(), "42");

  trxmmap::TrxFile<float> trx(4, 2);
  set_streamline_lengths(trx.streamlines.get(), {2, 2});
  trxmmap::add_dpv_from_tsf(trx, "signal", "float32", binary_path.string());
  auto it = trx.data_per_vertex.find("signal");
  ASSERT_NE(it, trx.data_per_vertex.end());
  EXPECT_EQ(it->second->_data.rows(), 4);
  EXPECT_EQ(it->second->_data.cols(), 1);
  EXPECT_FLOAT_EQ(it->second->_data(0, 0), 0.1F);
  EXPECT_FLOAT_EQ(it->second->_data(1, 0), 0.2F);
  EXPECT_FLOAT_EQ(it->second->_data(2, 0), 0.3F);
  EXPECT_FLOAT_EQ(it->second->_data(3, 0), 0.4F);
  EXPECT_EQ(it->second->_lengths, trx.streamlines->_lengths);
}

TEST(TrxFileIo, add_dpv_from_tsf_errors) {
  ScopedLocale scoped_locale(std::locale::classic());
  trxmmap::TrxFile<float> trx(4, 2);
  set_streamline_lengths(trx.streamlines.get(), {2, 2});

  const fs::path tmp_dir = make_temp_test_dir("trx_dpv_tsf_err");
  const fs::path input_path = tmp_dir / "dpv.tsf";
  write_tsf_text_file(input_path, build_tsf_contents({{0.1, 0.2}, {0.3}}));

  EXPECT_THROW(trxmmap::add_dpv_from_tsf(trx, "", "float32", input_path.string()), std::invalid_argument);
  EXPECT_THROW(trxmmap::add_dpv_from_tsf(trx, "signal", "badtype", input_path.string()), std::invalid_argument);
  EXPECT_THROW(trxmmap::add_dpv_from_tsf(trx, "signal", "int32", input_path.string()), std::invalid_argument);

  EXPECT_THROW(trxmmap::add_dpv_from_tsf(trx, "signal", "float32", (tmp_dir / "missing.tsf").string()),
               std::runtime_error);

  write_tsf_text_file(input_path, "0.1 0.2 abc");
  EXPECT_THROW(trxmmap::add_dpv_from_tsf(trx, "signal", "float32", input_path.string()), std::runtime_error);

  write_tsf_text_file(input_path, build_tsf_contents({{0.1}, {0.2, 0.3}}));
  EXPECT_THROW(trxmmap::add_dpv_from_tsf(trx, "signal", "float32", input_path.string()), std::runtime_error);

  write_tsf_text_file(input_path, build_tsf_contents({{0.1, 0.2}, {0.3}}));
  EXPECT_THROW(trxmmap::add_dpv_from_tsf(trx, "signal", "float32", input_path.string()), std::runtime_error);

  write_text_file(input_path, "mrtrix track scalars\nfile: . 0\ndatatype: Float32LE\ntimestamp: 0\n0.1 0.2");
  EXPECT_THROW(trxmmap::add_dpv_from_tsf(trx, "signal", "float32", input_path.string()), std::runtime_error);

  trxmmap::TrxFile<float> empty;
  EXPECT_THROW(trxmmap::add_dpv_from_tsf(empty, "signal", "float32", input_path.string()), std::runtime_error);

  trxmmap::TrxFile<float> no_dir(4, 2);
  set_streamline_lengths(no_dir.streamlines.get(), {2, 2});
  // Intentional white-box access: there is no public API to construct a TrxFile
  // with valid streamlines but without an uncompressed folder. This test verifies
  // that add_dpv_from_tsf fails in that specific internal state.
  no_dir._uncompressed_folder_handle.clear();
  EXPECT_THROW(trxmmap::add_dpv_from_tsf(no_dir, "signal", "float32", input_path.string()), std::runtime_error);
}

TEST(TrxFileIo, export_dpv_to_tsf_success) {
  ScopedLocale scoped_locale(std::locale::classic());
  trxmmap::TrxFile<float> trx(4, 2);
  set_streamline_lengths(trx.streamlines.get(), {2, 2});

  const fs::path tmp_dir = make_temp_test_dir("trx_export_tsf");
  const fs::path input_path = tmp_dir / "dpv_input.tsf";
  write_tsf_text_file(input_path, build_tsf_contents({{0.1, 0.2}, {0.3, 0.4}}));
  trxmmap::add_dpv_from_tsf(trx, "signal", "float32", input_path.string());

  const fs::path output_path = tmp_dir / "signal.tsf";
  const std::string timestamp = "1234.5678";
  trxmmap::export_dpv_to_tsf(trx, "signal", output_path.string(), timestamp);

  const TsfHeader header = read_tsf_header(output_path);
  EXPECT_EQ(header.timestamp, timestamp);
  const uint16_t endian_check = 1;
  const bool little_endian = *reinterpret_cast<const uint8_t *>(&endian_check) == 1;
  EXPECT_EQ(header.datatype, little_endian ? "Float32LE" : "Float32BE");
  EXPECT_EQ(header.count, 2U);
  EXPECT_EQ(header.total_count, 2U);

  const std::vector<float> values = read_tsf_float32_data(output_path, header.offset, 6);
  EXPECT_FLOAT_EQ(values[0], 0.1F);
  EXPECT_FLOAT_EQ(values[1], 0.2F);
  EXPECT_TRUE(std::isnan(values[2]));
  EXPECT_FLOAT_EQ(values[3], 0.3F);
  EXPECT_FLOAT_EQ(values[4], 0.4F);
  EXPECT_TRUE(std::isinf(values[5]));
}

TEST(TrxFileIo, export_dpv_to_tsf_errors) {
  trxmmap::TrxFile<float> trx(4, 2);
  set_streamline_lengths(trx.streamlines.get(), {2, 2});

  const fs::path tmp_dir = make_temp_test_dir("trx_export_tsf_err");
  const fs::path output_path = tmp_dir / "signal.tsf";

  EXPECT_THROW(trxmmap::export_dpv_to_tsf(trx, "", output_path.string(), "1"), std::invalid_argument);
  EXPECT_THROW(trxmmap::export_dpv_to_tsf(trx, "signal", output_path.string(), ""), std::invalid_argument);
  EXPECT_THROW(trxmmap::export_dpv_to_tsf(trx, "signal", output_path.string(), "1", "int32"), std::invalid_argument);
  EXPECT_THROW(trxmmap::export_dpv_to_tsf(trx, "missing", output_path.string(), "1"), std::runtime_error);

  trxmmap::TrxFile<float> empty;
  EXPECT_THROW(trxmmap::export_dpv_to_tsf(empty, "signal", output_path.string(), "1"), std::runtime_error);
}
