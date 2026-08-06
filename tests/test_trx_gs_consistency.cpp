#include <gtest/gtest.h>
#include <trx/trx.h>
#include <trx/legacy_io.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
fs::path get_gs_data_dir() {
  const auto *env = std::getenv("TRX_TEST_DATA_DIR");
  if (env != nullptr && !std::string(env).empty()) {
    fs::path dir = fs::path(env) / "gs";
    if (fs::exists(dir / "gs.trx")) return dir;
    dir = fs::path(env) / "gold_standard";
    if (fs::exists(dir / "gs.trx")) return dir;
    if (fs::exists(fs::path(env) / "gs.trx")) return fs::path(env);
  }
  fs::path repo_data = fs::path(__FILE__).parent_path() / "test_data" / "gs";
  if (fs::exists(repo_data / "gs.trx")) return repo_data;
  return {};
}
} // namespace

TEST(GsConsistency, HeaderDataMetadataWithinEpsilon) {
  const fs::path gs_dir = get_gs_data_dir();
  ASSERT_FALSE(gs_dir.empty()) << "Gold standard test data directory not found";

  const fs::path trx_path = gs_dir / "gs.trx";
  const fs::path trk_path = gs_dir / "gs.trk";
  const fs::path tck_path = gs_dir / "gs.tck";
  const fs::path vtk_path = gs_dir / "gs.vtk";

  ASSERT_TRUE(fs::exists(trx_path)) << "Missing " << trx_path;
  ASSERT_TRUE(fs::exists(trk_path)) << "Missing " << trk_path;
  ASSERT_TRUE(fs::exists(tck_path)) << "Missing " << tck_path;
  ASSERT_TRUE(fs::exists(vtk_path)) << "Missing " << vtk_path;

  trx::legacy::Tractogram tr_trx, tr_trk, tr_tck, tr_vtk;
  ASSERT_TRUE(trx::legacy::load_trx(trx_path.string(), tr_trx)) << "Failed to load " << trx_path;
  ASSERT_TRUE(trx::legacy::load_trk(trk_path.string(), tr_trk)) << "Failed to load " << trk_path;
  ASSERT_TRUE(trx::legacy::load_tck(tck_path.string(), tr_tck)) << "Failed to load " << tck_path;
  ASSERT_TRUE(trx::legacy::load_vtk(vtk_path.string(), tr_vtk)) << "Failed to load " << vtk_path;

  // 1. Compare streamline count and vertex counts
  const size_t num_streamlines = tr_trx.offsets.size() > 0 ? tr_trx.offsets.size() - 1 : 0;
  EXPECT_GT(num_streamlines, 0u);
  EXPECT_EQ(tr_trk.offsets.size() - 1, num_streamlines);
  EXPECT_EQ(tr_tck.offsets.size() - 1, num_streamlines);
  EXPECT_EQ(tr_vtk.offsets.size() - 1, num_streamlines);

  for (size_t i = 0; i < tr_trx.offsets.size(); ++i) {
    EXPECT_EQ(tr_trk.offsets[i], tr_trx.offsets[i]);
    EXPECT_EQ(tr_tck.offsets[i], tr_trx.offsets[i]);
    EXPECT_EQ(tr_vtk.offsets[i], tr_trx.offsets[i]);
  }

  // 2. Compare vertex positions within small epsilon (1e-3)
  const size_t num_pts_values = tr_trx.pts.size();
  EXPECT_EQ(tr_trk.pts.size(), num_pts_values);
  EXPECT_EQ(tr_tck.pts.size(), num_pts_values);
  EXPECT_EQ(tr_vtk.pts.size(), num_pts_values);

  constexpr float kEpsilon = 1e-3f;

  for (size_t i = 0; i < num_pts_values; ++i) {
    EXPECT_NEAR(tr_trk.pts[i], tr_trx.pts[i], kEpsilon) << "TRK vs TRX mismatch at idx " << i;
    EXPECT_NEAR(tr_tck.pts[i], tr_trx.pts[i], kEpsilon) << "TCK vs TRX mismatch at idx " << i;
    EXPECT_NEAR(std::abs(tr_vtk.pts[i]), std::abs(tr_trx.pts[i]), kEpsilon) << "VTK vs TRX magnitude mismatch at idx " << i;
  }

  // 3. Compare Header / Affine Matrix (VOXEL_TO_RASMM) within epsilon
  const auto &hdr_trx = tr_trx.header;
  const auto &hdr_trk = tr_trk.header;

  if (!hdr_trx["DIMENSIONS"].is_null() && !hdr_trk["DIMENSIONS"].is_null()) {
    const auto &dim_trx = hdr_trx["DIMENSIONS"].array_items();
    const auto &dim_trk = hdr_trk["DIMENSIONS"].array_items();
    ASSERT_EQ(dim_trx.size(), dim_trk.size());
    for (size_t i = 0; i < dim_trx.size(); ++i) {
      EXPECT_EQ(dim_trx[i].int_value(), dim_trk[i].int_value());
    }
  }

  if (!hdr_trx["VOXEL_TO_RASMM"].is_null() && !hdr_trk["VOXEL_TO_RASMM"].is_null()) {
    const auto &vox_trx = hdr_trx["VOXEL_TO_RASMM"].array_items();
    const auto &vox_trk = hdr_trk["VOXEL_TO_RASMM"].array_items();
    ASSERT_EQ(vox_trx.size(), 4u);
    ASSERT_EQ(vox_trk.size(), 4u);
    for (size_t r = 0; r < 4; ++r) {
      const auto &row_trx = vox_trx[r].array_items();
      const auto &row_trk = vox_trk[r].array_items();
      ASSERT_EQ(row_trx.size(), 4u);
      ASSERT_EQ(row_trk.size(), 4u);
      for (size_t c = 0; c < 4; ++c) {
        EXPECT_NEAR(row_trx[c].number_value(), row_trk[c].number_value(), kEpsilon)
            << "VOXEL_TO_RASMM mismatch at (" << r << ", " << c << ")";
      }
    }
  }
}
