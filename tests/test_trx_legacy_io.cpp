#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <trx/legacy_io.h>
#include <trx/trx.h>

#include <array>
#include <cstdlib>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::string test_data_root() {
  const auto *env = std::getenv("TRX_TEST_DATA_DIR"); // NOLINT(concurrency-mt-unsafe)
  if (env != nullptr && !std::string(env).empty()) {
    fs::path dir = fs::path(env) / "gs";
    if (fs::exists(dir / "gs.trx"))
      return dir.string();
    if (fs::exists(fs::path(env) / "gs.trx"))
      return std::string(env);
  }
  fs::path repo_data = fs::path(__FILE__).parent_path() / "test_data" / "gs";
  if (fs::exists(repo_data / "gs.trx"))
    return repo_data.string();
  return {};
}

fs::path unique_temp_path(const std::string &stem, const std::string &ext) {
  std::error_code ec;
  const fs::path base = fs::temp_directory_path(ec);
  if (ec) {
    throw std::runtime_error("Failed to get temp directory: " + ec.message());
  }
  return base / (stem + "_" + std::to_string(std::rand()) + ext);
}

void expect_legacy_to_trx_round_trip(const fs::path &input, const std::string &ref_nifti, const std::string &stem) {
  trx::legacy::Tractogram tr;
  if (input.extension() == ".tck") {
    ASSERT_TRUE(trx::legacy::load_tck(input.string(), tr));
  } else if (input.extension() == ".trk") {
    ASSERT_TRUE(trx::legacy::load_trk(input.string(), tr));
  } else if (input.extension() == ".vtk") {
    ASSERT_TRUE(trx::legacy::load_vtk(input.string(), tr));
  } else {
    FAIL() << "unhandled input extension: " << input.string();
  }
  ASSERT_FALSE(tr.offsets.empty());

  const size_t expected_vertices = tr.pts.size() / 3;
  const size_t expected_streamlines = tr.offsets.size() - 1;

  const fs::path out = unique_temp_path(stem, ".trx");
  std::error_code ec;
  fs::remove(out, ec);
  ASSERT_TRUE(trx::legacy::save_trx(tr, out.string(), ref_nifti));

  auto loaded = trx::load_any(out.string());
  EXPECT_EQ(loaded.num_vertices(), expected_vertices);
  EXPECT_EQ(loaded.num_streamlines(), expected_streamlines);
  loaded.close();

  fs::remove(out, ec);
}

} // namespace

TEST(LegacyIo, TckToTrxRoundTripPreservesHeaderCounts) {
  const std::string root_str = test_data_root();
  if (root_str.empty())
    GTEST_SKIP() << "Test data not found";
  const fs::path root(root_str);
  const fs::path tck = root / "gs.tck";
  const fs::path nii = root / "gs.nii";
  if (!fs::exists(tck) || !fs::exists(nii)) {
    GTEST_SKIP() << "gs.tck / gs.nii not present in test data";
  }
  expect_legacy_to_trx_round_trip(tck, nii.string(), "trx_legacy_tck_roundtrip");
}

TEST(LegacyIo, TrkToTrxRoundTripPreservesHeaderCounts) {
  const std::string root_str = test_data_root();
  if (root_str.empty())
    GTEST_SKIP() << "Test data not found";
  const fs::path root(root_str);
  const fs::path trk = root / "gs.trk";
  if (!fs::exists(trk)) {
    GTEST_SKIP() << "gs.trk not present in test data";
  }
  expect_legacy_to_trx_round_trip(trk, "", "trx_legacy_trk_roundtrip");
}

TEST(LegacyIo, VtkToTrxRoundTripPreservesHeaderCounts) {
  const std::string root_str = test_data_root();
  if (root_str.empty())
    GTEST_SKIP() << "Test data not found";
  const fs::path root(root_str);
  const fs::path vtk = root / "gs.vtk";
  const fs::path nii = root / "gs.nii";
  if (!fs::exists(vtk) || !fs::exists(nii)) {
    GTEST_SKIP() << "gs.vtk / gs.nii not present in test data";
  }
  expect_legacy_to_trx_round_trip(vtk, nii.string(), "trx_legacy_vtk_roundtrip");
}

TEST(LegacyIo, LoadNiftiHeaderValid) {
  const std::string root_str = test_data_root();
  if (root_str.empty())
    GTEST_SKIP() << "Test data not found";
  const fs::path nii = fs::path(root_str) / "gs.nii";
  if (!fs::exists(nii)) {
    GTEST_SKIP() << "gs.nii not present in test data";
  }
  json11::Json header;
  ASSERT_TRUE(trx::legacy::load_nifti_header(nii.string(), header));
  EXPECT_TRUE(header["VOXEL_TO_RASMM"].is_array());
  EXPECT_EQ(header["VOXEL_TO_RASMM"].array_items().size(), 4u);
  EXPECT_TRUE(header["DIMENSIONS"].is_array());
  EXPECT_EQ(header["DIMENSIONS"].array_items().size(), 3u);
}

TEST(LegacyIo, VtkMalformedInputFailsGracefully) {
  trx::legacy::Tractogram tr;

  // Non-existent file
  EXPECT_FALSE(trx::legacy::load_vtk("/nonexistent/path/file.vtk", tr));

  // Empty file
  const fs::path empty_file = unique_temp_path("empty_vtk", ".vtk");
  {
    std::ofstream out(empty_file);
  }
  EXPECT_FALSE(trx::legacy::load_vtk(empty_file.string(), tr));
  fs::remove(empty_file);

  // File claiming huge points count with no data
  const fs::path huge_pts_file = unique_temp_path("huge_pts", ".vtk");
  {
    std::ofstream out(huge_pts_file);
    out << "# vtk DataFile Version 4.2\nvtk output\nBINARY\nDATASET POLYDATA\nPOINTS 18446744073709551600 float\n";
  }
  EXPECT_FALSE(trx::legacy::load_vtk(huge_pts_file.string(), tr));
  fs::remove(huge_pts_file);

  // File claiming lines but truncated
  const fs::path truncated_file = unique_temp_path("trunc_vtk", ".vtk");
  {
    std::ofstream out(truncated_file, std::ios::binary);
    out << "# vtk DataFile Version 4.2\nvtk output\nBINARY\nDATASET POLYDATA\nPOINTS 3 float\n";
    std::array<float, 3> pts = {1.0f, 2.0f, 3.0f};
    out.write(reinterpret_cast<const char *>(pts.data()), sizeof(float) * pts.size());
    out << "LINES 10 100\n";
  }
  EXPECT_FALSE(trx::legacy::load_vtk(truncated_file.string(), tr));
  fs::remove(truncated_file);

  // File with negative cell count
  const fs::path neg_cell_file = unique_temp_path("neg_cell_vtk", ".vtk");
  {
    std::ofstream out(neg_cell_file, std::ios::binary);
    out << "# vtk DataFile Version 4.2\nvtk output\nBINARY\nDATASET POLYDATA\nPOINTS 3 float\n";
    std::array<float, 3> pts = {1.0f, 2.0f, 3.0f};
    out.write(reinterpret_cast<const char *>(pts.data()), sizeof(float) * pts.size());
    out << "LINES 1 10\n";
    int32_t neg_count = -5;
    out.write(reinterpret_cast<const char *>(&neg_count), sizeof(neg_count));
  }
  EXPECT_FALSE(trx::legacy::load_vtk(neg_cell_file.string(), tr));
  fs::remove(neg_cell_file);
}

TEST(LegacyIo, InPlaceDirectorySavePreservesData) {
  const std::string root_str = test_data_root();
  if (root_str.empty())
    GTEST_SKIP() << "Test data not found";
  const fs::path root(root_str);
  const fs::path src_trx = root / "gs.trx";
  if (!fs::exists(src_trx)) {
    GTEST_SKIP() << "gs.trx not present in test data";
  }

  const fs::path temp_dir = unique_temp_path("inplace_dir_save", "_dir");
  std::error_code ec;
  fs::remove_all(temp_dir, ec);

  // Load archive and save as uncompressed directory
  {
    auto trx = trx::load_any(src_trx.string());
    trx::TrxSaveOptions opts;
    opts.mode = trx::TrxSaveMode::Directory;
    trx.save(temp_dir.string(), opts);
    trx.close();
  }

  ASSERT_TRUE(fs::exists(temp_dir / "header.json"));

  // Now load from the directory and save IN-PLACE with overwrite=true
  {
    auto dir_trx = trx::load_any(temp_dir.string());
    const size_t orig_v = dir_trx.num_vertices();
    const size_t orig_s = dir_trx.num_streamlines();

    trx::TrxSaveOptions opts;
    opts.mode = trx::TrxSaveMode::Directory;
    opts.overwrite_existing = true;
    EXPECT_NO_THROW(dir_trx.save(temp_dir.string(), opts));
    dir_trx.close();

    // Verify directory still exists and loads correctly
    auto reloaded = trx::load_any(temp_dir.string());
    EXPECT_EQ(reloaded.num_vertices(), orig_v);
    EXPECT_EQ(reloaded.num_streamlines(), orig_s);
    reloaded.close();
  }

  fs::remove_all(temp_dir, ec);
}

TEST(LegacyIo, OutOfRangeUint32GroupThrows) {
  const std::string root_str = test_data_root();
  if (root_str.empty())
    GTEST_SKIP() << "Test data not found";
  const fs::path root(root_str);
  const fs::path src_trx = root / "gs.trx";
  if (!fs::exists(src_trx)) {
    GTEST_SKIP() << "gs.trx not present in test data";
  }

  const fs::path temp_dir = unique_temp_path("bad_group_test", "_dir");
  std::error_code ec;
  fs::remove_all(temp_dir, ec);

  // Save as directory first
  {
    auto trx = trx::load_any(src_trx.string());
    trx::TrxSaveOptions opts;
    opts.mode = trx::TrxSaveMode::Directory;
    trx.save(temp_dir.string(), opts);
    trx.close();
  }

  // Inject a group with an invalid index (index >= NB_STREAMLINES)
  const fs::path groups_dir = temp_dir / "groups";
  fs::create_directories(groups_dir, ec);
  const fs::path bad_group_file = groups_dir / "bad_group.uint32";
  {
    std::ofstream out(bad_group_file, std::ios::binary);
    uint32_t invalid_idx = 999999;
    out.write(reinterpret_cast<const char *>(&invalid_idx), sizeof(invalid_idx));
  }

  // Loading from directory should now detect the out-of-range uint32 group index and throw
  EXPECT_THROW(trx::load_any(temp_dir.string()), trx::TrxFormatError);

  fs::remove_all(temp_dir, ec);
}
