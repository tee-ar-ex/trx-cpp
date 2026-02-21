#include <filesystem>
#include <gtest/gtest.h>
#define private public
#include <trx/trx.h>
#undef private

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace trx;
namespace fs = std::filesystem;

namespace {
std::string get_test_data_root() {
  const auto *env = std::getenv("TRX_TEST_DATA_DIR"); // NOLINT(concurrency-mt-unsafe)
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
  if (!fs::exists(gs_dir / "gs.trx")) {
    throw std::runtime_error("Missing gold_standard gs.trx");
  }
  if (!fs::exists(gs_dir / "gs_fldr.trx")) {
    throw std::runtime_error("Missing gold_standard gs_fldr.trx");
  }
  return gs_dir;
}

fs::path make_temp_test_dir(const std::string &prefix) {
  std::error_code ec;
  auto base = fs::temp_directory_path(ec);
  if (ec) {
    throw std::runtime_error("Failed to get temp directory: " + ec.message());
  }

  thread_local std::mt19937_64 rng(std::random_device{}());
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

fs::path copy_gold_standard_dir(const fs::path &gs_dir, const std::string &prefix, fs::path &temp_root) {
  const fs::path source = gs_dir / "gs_fldr.trx";
  if (!fs::exists(source)) {
    throw std::runtime_error("Missing gold_standard gs_fldr.trx");
  }
  temp_root = make_temp_test_dir(prefix);
  fs::path dest = temp_root / source.filename();
  std::error_code ec;
  fs::copy(source, dest, fs::copy_options::recursive, ec);
  if (ec) {
    throw std::runtime_error("Failed to copy gold_standard directory: " + ec.message());
  }
  return dest;
}

fs::path find_file_with_prefix(const fs::path &dir, const std::string &prefix) {
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
    if (ec) {
      break;
    }
    if (!it->is_regular_file(ec)) {
      continue;
    }
    const std::string name = it->path().filename().string();
    if (name.rfind(prefix, 0) == 0) {
      return it->path();
    }
  }
  throw std::runtime_error("Failed to find file with prefix: " + prefix);
}

fs::path find_first_file_recursive(const fs::path &dir) {
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
    if (ec) {
      break;
    }
    if (it->is_regular_file(ec)) {
      return it->path();
    }
  }
  throw std::runtime_error("Failed to find file under: " + dir.string());
}

json read_header_file(const fs::path &dir) {
  const fs::path header_path = dir / "header.json";
  std::ifstream in(header_path.string());
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open header.json");
  }
  std::string jstream((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();
  std::string err;
  json header = json::parse(jstream, err);
  if (!err.empty()) {
    throw std::runtime_error("Failed to parse header.json: " + err);
  }
  return header;
}

void write_header_file(const fs::path &dir, const json &header) {
  const fs::path header_path = dir / "header.json";
  std::ofstream out(header_path.string());
  if (!out.is_open()) {
    throw std::runtime_error("Failed to write header.json");
  }
  out << header.dump() << '\n';
  out.close();
}

std::string pick_int_dtype_same_size(const std::string &ext) {
  const int size = trx::detail::_sizeof_dtype(ext);
  if (size == 8) {
    return "int64";
  }
  if (size == 4) {
    return "int32";
  }
  return "int16";
}

fs::path rename_with_new_dim(const fs::path &file_path, int new_dim) {
  const std::string filename = file_path.filename().string();
  auto parsed = trx::detail::_split_ext_with_dimensionality(filename);
  const std::string base = std::get<0>(parsed);
  const std::string ext = std::get<2>(parsed);
  std::string new_name = base + "." + std::to_string(new_dim) + "." + ext;
  fs::path new_path = file_path.parent_path() / new_name;
  fs::rename(file_path, new_path);
  return new_path;
}

fs::path rename_with_new_ext(const fs::path &file_path, const std::string &new_ext) {
  const std::string filename = file_path.filename().string();
  const std::string old_ext = get_ext(filename);
  const std::string new_name = filename.substr(0, filename.size() - old_ext.size()) + new_ext;
  fs::path new_path = file_path.parent_path() / new_name;
  fs::rename(file_path, new_path);
  return new_path;
}

void ensure_directory_exists(const fs::path &dir_path) {
  std::error_code ec;
  if (!fs::exists(dir_path, ec)) {
    fs::create_directories(dir_path, ec);
  }
  if (ec) {
    throw std::runtime_error("Failed to create directory: " + dir_path.string());
  }
}

void write_zero_filled_file(const fs::path &file_path, const std::string &dtype, size_t count) {
  const int dtype_size = trx::detail::_sizeof_dtype(dtype);
  const size_t total_bytes = count * static_cast<size_t>(dtype_size);
  std::vector<char> bytes(total_bytes, 0);
  std::ofstream out(file_path.string(), std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to write file: " + file_path.string());
  }
  if (!bytes.empty()) {
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  out.close();
}

bool has_regular_file_recursive(const fs::path &dir_path) {
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir_path, ec), end; it != end; it.increment(ec)) {
    if (ec) {
      break;
    }
    if (it->is_regular_file(ec)) {
      return true;
    }
  }
  return false;
}

void expect_basic_consistency(const AnyTrxFile &trx) {
  ASSERT_TRUE(trx.header["NB_STREAMLINES"].is_number());
  ASSERT_TRUE(trx.header["NB_VERTICES"].is_number());

  const auto nb_streamlines = static_cast<size_t>(trx.header["NB_STREAMLINES"].int_value());
  const auto nb_vertices = static_cast<size_t>(trx.header["NB_VERTICES"].int_value());

  EXPECT_EQ(trx.num_streamlines(), nb_streamlines);
  EXPECT_EQ(trx.num_vertices(), nb_vertices);

  ASSERT_EQ(trx.offsets_u64.size(), nb_streamlines + 1);
  EXPECT_EQ(trx.offsets_u64.back(), static_cast<uint64_t>(nb_vertices));

  EXPECT_EQ(trx.positions.cols, 3);
  EXPECT_EQ(trx.positions.rows, static_cast<int>(nb_vertices));
  EXPECT_FALSE(trx.positions.empty());

  const auto bytes = trx.positions.to_bytes();
  EXPECT_NE(bytes.data, nullptr);
  EXPECT_GT(bytes.size, 0U);
}

fs::path write_test_shard(const fs::path &root,
                          const std::string &name,
                          const std::vector<std::array<float, 3>> &points,
                          const std::vector<uint64_t> &offsets,
                          const std::vector<float> &dps_values,
                          const std::vector<float> &dpv_values,
                          const std::vector<uint32_t> &group_indices) {
  const fs::path shard_dir = root / name;
  std::error_code ec;
  fs::create_directories(shard_dir, ec);
  if (ec) {
    throw std::runtime_error("Failed to create shard directory: " + shard_dir.string());
  }

  json::object header_obj;
  header_obj["DIMENSIONS"] = json::array{1, 1, 1};
  header_obj["NB_STREAMLINES"] = static_cast<int>(offsets.size() - 1);
  header_obj["NB_VERTICES"] = static_cast<int>(points.size());
  header_obj["VOXEL_TO_RASMM"] = json::array{
      json::array{1.0, 0.0, 0.0, 0.0},
      json::array{0.0, 1.0, 0.0, 0.0},
      json::array{0.0, 0.0, 1.0, 0.0},
      json::array{0.0, 0.0, 0.0, 1.0},
  };
  write_header_file(shard_dir, json(header_obj));

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> positions(
      static_cast<Eigen::Index>(points.size()), 3);
  for (size_t i = 0; i < points.size(); ++i) {
    positions(static_cast<Eigen::Index>(i), 0) = points[i][0];
    positions(static_cast<Eigen::Index>(i), 1) = points[i][1];
    positions(static_cast<Eigen::Index>(i), 2) = points[i][2];
  }
  trx::write_binary((shard_dir / "positions.3.float32").string(), positions);

  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> offsets_mat(
      static_cast<Eigen::Index>(offsets.size()), 1);
  for (size_t i = 0; i < offsets.size(); ++i) {
    offsets_mat(static_cast<Eigen::Index>(i), 0) = offsets[i];
  }
  trx::write_binary((shard_dir / "offsets.uint64").string(), offsets_mat);

  fs::create_directories(shard_dir / "dps", ec);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dps_mat(
      static_cast<Eigen::Index>(dps_values.size()), 1);
  for (size_t i = 0; i < dps_values.size(); ++i) {
    dps_mat(static_cast<Eigen::Index>(i), 0) = dps_values[i];
  }
  trx::write_binary((shard_dir / "dps" / "weight.float32").string(), dps_mat);

  fs::create_directories(shard_dir / "dpv", ec);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dpv_mat(
      static_cast<Eigen::Index>(dpv_values.size()), 1);
  for (size_t i = 0; i < dpv_values.size(); ++i) {
    dpv_mat(static_cast<Eigen::Index>(i), 0) = dpv_values[i];
  }
  trx::write_binary((shard_dir / "dpv" / "signal.float32").string(), dpv_mat);

  fs::create_directories(shard_dir / "groups", ec);
  Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> groups_mat(
      static_cast<Eigen::Index>(group_indices.size()), 1);
  for (size_t i = 0; i < group_indices.size(); ++i) {
    groups_mat(static_cast<Eigen::Index>(i), 0) = group_indices[i];
  }
  trx::write_binary((shard_dir / "groups" / "Bundle.uint32").string(), groups_mat);

  return shard_dir;
}
} // namespace

TEST(AnyTrxFile, LoadZipAndValidate) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs.trx";
  auto trx = load_any(gs_trx.string());

  EXPECT_TRUE(trx.positions.dtype == "float16" || trx.positions.dtype == "float32" || trx.positions.dtype == "float64");
  expect_basic_consistency(trx);

  if (trx.positions.dtype == "float32") {
    auto positions = trx.positions.as_matrix<float>();
    EXPECT_EQ(positions.rows(), trx.positions.rows);
    EXPECT_EQ(positions.cols(), trx.positions.cols);
  }
  trx.close();
}

TEST(AnyTrxFile, LoadDirectoryAndValidate) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  EXPECT_TRUE(trx.positions.dtype == "float16" || trx.positions.dtype == "float32" || trx.positions.dtype == "float64");
  expect_basic_consistency(trx);
  trx.close();
}

TEST(AnyTrxFile, SaveUpdatesHeader) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs.trx";
  auto trx = load_any(gs_trx.string());

  auto header_obj = trx.header.object_items();
  header_obj["COMMENT"] = "saved by anytrxfile test";
  trx.header = json(header_obj);

  const auto temp_dir = make_temp_test_dir("trx_any_save");
  const fs::path out_path = temp_dir / "saved_copy.trx";
  trx.save(out_path.string(), ZIP_CM_STORE);
  trx.close();

  auto reloaded = load_any(out_path.string());
  EXPECT_EQ(reloaded.header["COMMENT"].string_value(), "saved by anytrxfile test");
  reloaded.close();
}

TEST(AnyTrxFile, MissingHeaderCountsThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_missing_counts", temp_root);

  auto header = read_header_file(corrupt_dir);
  auto header_obj = header.object_items();
  header_obj.erase("NB_VERTICES");
  write_header_file(corrupt_dir, json(header_obj));

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, WrongPositionsDimThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_pos_dim", temp_root);

  const fs::path positions = find_file_with_prefix(corrupt_dir, "positions");
  rename_with_new_dim(positions, 4);

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, UnsupportedPositionsDtypeThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_pos_dtype", temp_root);

  const fs::path positions = find_file_with_prefix(corrupt_dir, "positions");
  const std::string ext = get_ext(positions.string());
  rename_with_new_ext(positions, pick_int_dtype_same_size(ext));

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, WrongOffsetsDimThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_offsets_dim", temp_root);

  const fs::path offsets = find_file_with_prefix(corrupt_dir, "offsets");
  rename_with_new_dim(offsets, 2);

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, UnsupportedOffsetsDtypeThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_offsets_dtype", temp_root);

  const fs::path offsets = find_file_with_prefix(corrupt_dir, "offsets");
  const std::string ext = get_ext(offsets.string());
  rename_with_new_ext(offsets, pick_int_dtype_same_size(ext));

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, WrongDpsDimThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_dps_dim", temp_root);

  const fs::path dps_dir = corrupt_dir / "dps";
  if (!fs::exists(dps_dir) || !has_regular_file_recursive(dps_dir)) {
    ensure_directory_exists(dps_dir);
    const auto header = read_header_file(corrupt_dir);
    const auto nb_streamlines = static_cast<size_t>(header["NB_STREAMLINES"].int_value());
    const fs::path dps_file = dps_dir / "weight.float32";
    write_zero_filled_file(dps_file, "float32", nb_streamlines);
  }
  const fs::path dps_file = find_first_file_recursive(dps_dir);
  rename_with_new_dim(dps_file, 2);

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, WrongDpvDimThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_dpv_dim", temp_root);

  const fs::path dpv_dir = corrupt_dir / "dpv";
  if (!fs::exists(dpv_dir) || !has_regular_file_recursive(dpv_dir)) {
    ensure_directory_exists(dpv_dir);
    const auto header = read_header_file(corrupt_dir);
    const auto nb_vertices = static_cast<size_t>(header["NB_VERTICES"].int_value());
    const fs::path dpv_file = dpv_dir / "color.float32";
    write_zero_filled_file(dpv_file, "float32", nb_vertices);
  }
  const fs::path dpv_file = find_first_file_recursive(dpv_dir);
  rename_with_new_dim(dpv_file, 2);

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, WrongDpgDimThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_dpg_dim", temp_root);

  const fs::path dpg_dir = corrupt_dir / "dpg";
  if (!fs::exists(dpg_dir) || !has_regular_file_recursive(dpg_dir)) {
    const fs::path group_dir = dpg_dir / "GroupA";
    ensure_directory_exists(group_dir);
    const fs::path dpg_file = group_dir / "mean.float32";
    write_zero_filled_file(dpg_file, "float32", 1);
  }
  const fs::path dpg_file = find_first_file_recursive(dpg_dir);
  rename_with_new_dim(dpg_file, 2);

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, UnsupportedGroupDtypeThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_group_dtype", temp_root);

  const fs::path groups_dir = corrupt_dir / "groups";
  if (!fs::exists(groups_dir) || !has_regular_file_recursive(groups_dir)) {
    ensure_directory_exists(groups_dir);
    const auto header = read_header_file(corrupt_dir);
    const auto nb_streamlines = static_cast<size_t>(header["NB_STREAMLINES"].int_value());
    const fs::path group_file = groups_dir / "GroupA.uint32";
    write_zero_filled_file(group_file, "uint32", nb_streamlines);
  }
  const fs::path group_file = find_first_file_recursive(groups_dir);
  rename_with_new_ext(group_file, "int32");

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, InvalidEntryThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_invalid_entry", temp_root);

  const fs::path bogus = corrupt_dir / "bogus.float32";
  std::ofstream out(bogus.string(), std::ios::binary);
  float value = 1.0F;
  std::array<char, sizeof(value)> value_bytes{};
  std::memcpy(value_bytes.data(), &value, sizeof(value));
  out.write(value_bytes.data(), static_cast<std::streamsize>(value_bytes.size()));
  out.close();

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, MissingEssentialDataThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_missing_essential", temp_root);

  const fs::path positions = find_file_with_prefix(corrupt_dir, "positions");
  std::error_code ec;
  fs::remove(positions, ec);

  EXPECT_THROW(load_any(corrupt_dir.string()), std::invalid_argument);

  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, OffsetsOverflowThrows) {
  const auto gs_dir = require_gold_standard_dir();
  fs::path temp_root;
  const fs::path corrupt_dir = copy_gold_standard_dir(gs_dir, "trx_any_offsets_overflow", temp_root);

  const auto header = read_header_file(corrupt_dir);
  const int nb_streamlines = header["NB_STREAMLINES"].int_value();
  const fs::path offsets = find_file_with_prefix(corrupt_dir, "offsets");
  const std::string ext = get_ext(offsets.string());

  fs::path offsets_path = offsets;
  if (ext != "uint64") {
    offsets_path = rename_with_new_ext(offsets, "uint64");
  }

  std::vector<uint64_t> data(static_cast<size_t>(nb_streamlines) + 1, 0);
  if (data.size() >= 2) {
    data[1] = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    for (size_t i = 2; i < data.size(); ++i) {
      data[i] = data[i - 1] + 1;
    }
  }
  std::ofstream out(offsets_path.string(), std::ios::binary | std::ios::trunc);
  std::vector<char> data_bytes(data.size() * sizeof(uint64_t));
  std::memcpy(data_bytes.data(), data.data(), data_bytes.size());
  out.write(data_bytes.data(), static_cast<std::streamsize>(data_bytes.size()));
  out.close();

  EXPECT_THROW(load_any(corrupt_dir.string()), std::runtime_error);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, SaveRejectsUnsupportedExtension) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  const auto temp_dir = make_temp_test_dir("trx_any_save_badext");
  const fs::path out_path = temp_dir / "bad.txt";
  EXPECT_THROW(trx.save(out_path.string(), ZIP_CM_STORE), std::invalid_argument);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_dir, ec);
}

TEST(AnyTrxFile, SaveRejectsMissingOffsets) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  trx.offsets = TypedArray();
  const auto temp_dir = make_temp_test_dir("trx_any_save_no_offsets");
  const fs::path out_path = temp_dir / "missing_offsets.trx";
  EXPECT_THROW(trx.save(out_path.string(), ZIP_CM_STORE), std::runtime_error);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_dir, ec);
}

TEST(AnyTrxFile, SaveRejectsMissingDecodedOffsets) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  trx.offsets_u64.clear();
  const auto temp_dir = make_temp_test_dir("trx_any_save_no_offsets_u64");
  const fs::path out_path = temp_dir / "missing_offsets_u64.trx";
  EXPECT_THROW(trx.save(out_path.string(), ZIP_CM_STORE), std::runtime_error);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_dir, ec);
}

TEST(AnyTrxFile, SaveRejectsStreamlineCountMismatch) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  auto header_obj = trx.header.object_items();
  header_obj["NB_STREAMLINES"] = header_obj["NB_STREAMLINES"].int_value() + 1;
  trx.header = json(header_obj);

  const auto temp_dir = make_temp_test_dir("trx_any_save_bad_streamlines");
  const fs::path out_path = temp_dir / "bad_streamlines.trx";
  EXPECT_THROW(trx.save(out_path.string(), ZIP_CM_STORE), std::runtime_error);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_dir, ec);
}

TEST(AnyTrxFile, SaveRejectsVertexCountMismatch) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  auto header_obj = trx.header.object_items();
  header_obj["NB_VERTICES"] = static_cast<int>(trx.offsets_u64.back() + 1);
  trx.header = json(header_obj);

  const auto temp_dir = make_temp_test_dir("trx_any_save_bad_vertices");
  const fs::path out_path = temp_dir / "bad_vertices.trx";
  EXPECT_THROW(trx.save(out_path.string(), ZIP_CM_STORE), std::runtime_error);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_dir, ec);
}

TEST(AnyTrxFile, SaveRejectsNonMonotonicOffsets) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  ASSERT_GT(trx.offsets_u64.size(), 1U);
  trx.offsets_u64[0] = 1;
  trx.offsets_u64[1] = 0;

  const auto temp_dir = make_temp_test_dir("trx_any_save_non_mono");
  const fs::path out_path = temp_dir / "non_mono.trx";
  EXPECT_THROW(trx.save(out_path.string(), ZIP_CM_STORE), std::runtime_error);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_dir, ec);
}

TEST(AnyTrxFile, SaveRejectsPositionsRowMismatch) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  const uint64_t sentinel = trx.offsets_u64.back();
  ASSERT_GT(sentinel, 0U);
  trx.positions.rows = static_cast<int>(sentinel - 1);

  const auto temp_dir = make_temp_test_dir("trx_any_save_bad_positions");
  const fs::path out_path = temp_dir / "bad_positions.trx";
  EXPECT_THROW(trx.save(out_path.string(), ZIP_CM_STORE), std::runtime_error);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_dir, ec);
}

TEST(AnyTrxFile, SaveRejectsMissingBackingDirectory) {
  const auto gs_dir = require_gold_standard_dir();
  const fs::path gs_trx = gs_dir / "gs_fldr.trx";
  auto trx = load_any(gs_trx.string());

  trx._backing_directory.clear();
  trx._uncompressed_folder_handle.clear();

  const auto temp_dir = make_temp_test_dir("trx_any_save_no_backing");
  const fs::path out_path = temp_dir / "no_backing.trx";
  EXPECT_THROW(trx.save(out_path.string(), ZIP_CM_STORE), std::runtime_error);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_dir, ec);
}

TEST(AnyTrxFile, MergeTrxShardsDirectoryOutput) {
  const fs::path temp_root = make_temp_test_dir("trx_merge_shards");
  const fs::path shard1 = write_test_shard(temp_root,
                                           "shard1",
                                           {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
                                           {0, 2},
                                           {10.0F},
                                           {0.1F, 0.2F},
                                           {0});
  const fs::path shard2 = write_test_shard(temp_root,
                                           "shard2",
                                           {{2.0F, 0.0F, 0.0F}, {3.0F, 0.0F, 0.0F}, {4.0F, 0.0F, 0.0F}},
                                           {0, 1, 3},
                                           {20.0F, 30.0F},
                                           {0.3F, 0.4F, 0.5F},
                                           {1});

  const fs::path output_dir = temp_root / "merged";
  MergeTrxShardsOptions options;
  options.shard_directories = {shard1.string(), shard2.string()};
  options.output_path = output_dir.string();
  options.output_directory = true;
  merge_trx_shards(options);

  auto merged = load_any(output_dir.string());
  EXPECT_EQ(merged.num_streamlines(), 3U);
  EXPECT_EQ(merged.num_vertices(), 5U);
  ASSERT_EQ(merged.offsets_u64.size(), 4U);
  EXPECT_EQ(merged.offsets_u64[0], 0U);
  EXPECT_EQ(merged.offsets_u64[1], 2U);
  EXPECT_EQ(merged.offsets_u64[2], 3U);
  EXPECT_EQ(merged.offsets_u64[3], 5U);

  const auto pos = merged.positions.as_matrix<float>();
  EXPECT_FLOAT_EQ(pos(0, 0), 0.0F);
  EXPECT_FLOAT_EQ(pos(4, 0), 4.0F);

  auto dps_it = merged.data_per_streamline.find("weight");
  ASSERT_NE(dps_it, merged.data_per_streamline.end());
  auto dps = dps_it->second.as_matrix<float>();
  EXPECT_EQ(dps.rows(), 3);
  EXPECT_FLOAT_EQ(dps(0, 0), 10.0F);
  EXPECT_FLOAT_EQ(dps(2, 0), 30.0F);

  auto dpv_it = merged.data_per_vertex.find("signal");
  ASSERT_NE(dpv_it, merged.data_per_vertex.end());
  auto dpv = dpv_it->second.as_matrix<float>();
  EXPECT_EQ(dpv.rows(), 5);
  EXPECT_FLOAT_EQ(dpv(0, 0), 0.1F);
  EXPECT_FLOAT_EQ(dpv(4, 0), 0.5F);

  auto group_it = merged.groups.find("Bundle");
  ASSERT_NE(group_it, merged.groups.end());
  auto grp = group_it->second.as_matrix<uint32_t>();
  ASSERT_EQ(grp.rows(), 2);
  EXPECT_EQ(grp(0, 0), 0U);
  EXPECT_EQ(grp(1, 0), 2U);
  merged.close();

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, MergeTrxShardsSchemaMismatchThrows) {
  const fs::path temp_root = make_temp_test_dir("trx_merge_schema");
  const fs::path shard1 = write_test_shard(temp_root,
                                           "shard1",
                                           {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
                                           {0, 2},
                                           {10.0F},
                                           {0.1F, 0.2F},
                                           {0});
  const fs::path shard2 = write_test_shard(temp_root,
                                           "shard2",
                                           {{2.0F, 0.0F, 0.0F}},
                                           {0, 1},
                                           {20.0F},
                                           {0.3F},
                                           {0});

  std::error_code ec;
  fs::remove(shard2 / "dpv" / "signal.float32", ec);

  const fs::path output_dir = temp_root / "merged";
  MergeTrxShardsOptions options;
  options.shard_directories = {shard1.string(), shard2.string()};
  options.output_path = output_dir.string();
  options.output_directory = true;
  EXPECT_THROW(merge_trx_shards(options), std::runtime_error);

  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, PreparePositionsOutputCopiesMetadataAndOffsets) {
  const fs::path temp_root = make_temp_test_dir("trx_prepare_positions");
  const fs::path shard1 = write_test_shard(temp_root,
                                           "shard1",
                                           {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
                                           {0, 2},
                                           {10.0F},
                                           {0.1F, 0.2F},
                                           {0});
  auto input = load_any(shard1.string());

  const fs::path output_dir = temp_root / "prepared";
  PrepareOutputOptions options;
  options.overwrite_existing = true;
  const auto info = prepare_positions_output(input, output_dir.string(), options);

  EXPECT_EQ(info.directory, output_dir.string());
  EXPECT_EQ(info.dtype, "float32");
  EXPECT_EQ(info.points, 2U);
  EXPECT_EQ(fs::path(info.positions_path).filename().string(), "positions.3.float32");
  EXPECT_TRUE(fs::exists(output_dir / "header.json"));
  EXPECT_TRUE(fs::exists(output_dir / "offsets.uint64"));
  EXPECT_TRUE(fs::exists(output_dir / "dps" / "weight.float32"));
  EXPECT_TRUE(fs::exists(output_dir / "dpv" / "signal.float32"));
  EXPECT_TRUE(fs::exists(output_dir / "groups" / "Bundle.uint32"));
  EXPECT_FALSE(fs::exists(info.positions_path));

  input.close();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, PositionsChunkIterationAndMutation) {
  const fs::path temp_root = make_temp_test_dir("trx_positions_chunk");
  const fs::path shard1 = write_test_shard(temp_root,
                                           "shard1",
                                           {{0.0F, 0.0F, 0.0F}, {1.0F, 2.0F, 3.0F}},
                                           {0, 2},
                                           {10.0F},
                                           {0.1F, 0.2F},
                                           {0});
  auto trx = load_any(shard1.string());

  size_t total_points = 0;
  size_t callbacks = 0;
  trx.for_each_positions_chunk(12, [&](TrxScalarType dtype, const void *data, size_t point_offset, size_t point_count) {
    EXPECT_EQ(dtype, TrxScalarType::Float32);
    EXPECT_NE(data, nullptr);
    EXPECT_EQ(point_count, 1U);
    EXPECT_LT(point_offset, 2U);
    total_points += point_count;
    callbacks += 1;
  });
  EXPECT_EQ(total_points, 2U);
  EXPECT_EQ(callbacks, 2U);

  trx.for_each_positions_chunk_mutable(12, [&](TrxScalarType dtype, void *data, size_t, size_t point_count) {
    EXPECT_EQ(dtype, TrxScalarType::Float32);
    auto *vals = reinterpret_cast<float *>(data);
    for (size_t i = 0; i < point_count * 3; ++i) {
      vals[i] += 1.0F;
    }
  });

  const auto positions = trx.positions.as_matrix<float>();
  EXPECT_FLOAT_EQ(positions(0, 0), 1.0F);
  EXPECT_FLOAT_EQ(positions(1, 0), 2.0F);
  EXPECT_FLOAT_EQ(positions(1, 1), 3.0F);
  EXPECT_FLOAT_EQ(positions(1, 2), 4.0F);
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, MergeTrxShardsArchiveOutput) {
  const fs::path temp_root = make_temp_test_dir("trx_merge_shards_archive");
  const fs::path shard1 = write_test_shard(temp_root,
                                           "shard1",
                                           {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
                                           {0, 2},
                                           {10.0F},
                                           {0.1F, 0.2F},
                                           {0});
  const fs::path shard2 = write_test_shard(temp_root,
                                           "shard2",
                                           {{2.0F, 0.0F, 0.0F}, {3.0F, 0.0F, 0.0F}},
                                           {0, 2},
                                           {20.0F},
                                           {0.3F, 0.4F},
                                           {0});

  const fs::path output_archive = temp_root / "merged.trx";
  MergeTrxShardsOptions options;
  options.shard_directories = {shard1.string(), shard2.string()};
  options.output_path = output_archive.string();
  options.output_directory = false;
  merge_trx_shards(options);

  ASSERT_TRUE(fs::exists(output_archive));
  auto merged = load_any(output_archive.string());
  EXPECT_EQ(merged.num_streamlines(), 2U);
  EXPECT_EQ(merged.num_vertices(), 4U);
  merged.close();

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, MergeTrxShardsRejectsDpg) {
  const fs::path temp_root = make_temp_test_dir("trx_merge_shards_dpg");
  const fs::path shard1 = write_test_shard(temp_root,
                                           "shard1",
                                           {{0.0F, 0.0F, 0.0F}},
                                           {0, 1},
                                           {10.0F},
                                           {0.1F},
                                           {0});
  const fs::path shard2 = write_test_shard(temp_root,
                                           "shard2",
                                           {{1.0F, 0.0F, 0.0F}},
                                           {0, 1},
                                           {20.0F},
                                           {0.2F},
                                           {0});
  std::error_code ec;
  fs::create_directories(shard2 / "dpg" / "Bundle", ec);
  std::ofstream out((shard2 / "dpg" / "Bundle" / "mean.float32").string(), std::ios::binary | std::ios::trunc);
  float one = 1.0F;
  out.write(reinterpret_cast<const char *>(&one), sizeof(float));
  out.close();

  MergeTrxShardsOptions options;
  options.shard_directories = {shard1.string(), shard2.string()};
  options.output_path = (temp_root / "merged").string();
  options.output_directory = true;
  EXPECT_THROW(merge_trx_shards(options), std::runtime_error);

  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, PreparePositionsOutputOverwriteFalseThrows) {
  const fs::path temp_root = make_temp_test_dir("trx_prepare_positions_overwrite");
  const fs::path shard1 = write_test_shard(temp_root,
                                           "shard1",
                                           {{0.0F, 0.0F, 0.0F}},
                                           {0, 1},
                                           {10.0F},
                                           {0.1F},
                                           {0});
  auto input = load_any(shard1.string());
  const fs::path output_dir = temp_root / "prepared";
  std::error_code ec;
  fs::create_directories(output_dir, ec);
  ASSERT_TRUE(fs::exists(output_dir));

  PrepareOutputOptions options;
  options.overwrite_existing = false;
  EXPECT_THROW(prepare_positions_output(input, output_dir.string(), options), std::runtime_error);

  input.close();
  fs::remove_all(temp_root, ec);
}

TEST(AnyTrxFile, SaveRespectsExplicitMode) {
  const fs::path temp_root = make_temp_test_dir("trx_any_save_modes");
  const fs::path shard1 = write_test_shard(temp_root,
                                           "shard1",
                                           {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
                                           {0, 2},
                                           {10.0F},
                                           {0.1F, 0.2F},
                                           {0});
  auto trx = load_any(shard1.string());

  const fs::path dir_out = temp_root / "save_dir_mode";
  TrxSaveOptions dir_opts;
  dir_opts.mode = TrxSaveMode::Directory;
  trx.save(dir_out.string(), dir_opts);
  EXPECT_TRUE(fs::is_directory(dir_out));
  EXPECT_TRUE(fs::exists(dir_out / "header.json"));

  const fs::path archive_out = temp_root / "save_archive_mode.trx";
  TrxSaveOptions archive_opts;
  archive_opts.mode = TrxSaveMode::Archive;
  archive_opts.compression_standard = ZIP_CM_STORE;
  trx.save(archive_out.string(), archive_opts);
  EXPECT_TRUE(fs::is_regular_file(archive_out));

  auto dir_loaded = load_any(dir_out.string());
  auto arc_loaded = load_any(archive_out.string());
  EXPECT_EQ(dir_loaded.num_streamlines(), trx.num_streamlines());
  EXPECT_EQ(arc_loaded.num_vertices(), trx.num_vertices());
  dir_loaded.close();
  arc_loaded.close();
  trx.close();

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}
