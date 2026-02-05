#include <filesystem>
#include <gtest/gtest.h>
#include <trx/trx.h>

#include <cstdlib>
#include <random>
#include <string>

using namespace trx;
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
  EXPECT_GT(bytes.size, 0u);
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
