#include <filesystem>
#include <gtest/gtest.h>
#include <trx/legacy_io.h>
#include <trx/trx.h>

#include <cstdlib>
#include <string>

namespace fs = std::filesystem;

namespace {
std::string test_data_root() {
  const char *env = std::getenv("TRX_TEST_DATA_DIR"); // NOLINT(concurrency-mt-unsafe)
  if (env == nullptr || std::string(env).empty()) {
    throw std::runtime_error("TRX_TEST_DATA_DIR not set");
  }
  return std::string(env);
}

fs::path unique_temp_trx(const std::string &stem) {
  std::error_code ec;
  const fs::path base = fs::temp_directory_path(ec);
  if (ec) {
    throw std::runtime_error("Failed to get temp directory: " + ec.message());
  }
  return base / (stem + ".trx");
}

// Loads a legacy tractogram, saves it as TRX, and verifies the resulting TRX
// loads back with vertex/streamline counts matching the source. This guards the
// header-count regression where legacy -> TRX dropped NB_VERTICES /
// NB_STREAMLINES and produced an unreadable file.
void expect_legacy_to_trx_round_trip(const fs::path &input, const std::string &ref_nifti, const std::string &stem) {
  trx::legacy::Tractogram tr;
  if (input.extension() == ".tck") {
    ASSERT_TRUE(trx::legacy::load_tck(input.string(), tr));
  } else if (input.extension() == ".trk") {
    ASSERT_TRUE(trx::legacy::load_trk(input.string(), tr));
  } else {
    FAIL() << "unhandled input extension: " << input.string();
  }
  ASSERT_FALSE(tr.offsets.empty());

  const size_t expected_vertices = tr.pts.size() / 3;
  const size_t expected_streamlines = tr.offsets.size() - 1;

  const fs::path out = unique_temp_trx(stem);
  std::error_code ec;
  fs::remove(out, ec);
  ASSERT_TRUE(trx::legacy::save_trx(tr, out.string(), ref_nifti));

  // load_any requires NB_STREAMLINES/NB_VERTICES in header.json; before the fix
  // this threw "Missing NB_VERTICES or NB_STREAMLINES in header.json".
  auto loaded = trx::load_any(out.string());
  EXPECT_EQ(loaded.num_vertices(), expected_vertices);
  EXPECT_EQ(loaded.num_streamlines(), expected_streamlines);
  loaded.close();

  fs::remove(out, ec);
}

} // namespace

TEST(LegacyIo, TckToTrxRoundTripPreservesHeaderCounts) {
  const fs::path root = test_data_root();
  const fs::path tck = root / "gs.tck";
  const fs::path nii = root / "gs.nii";
  if (!fs::exists(tck) || !fs::exists(nii)) {
    GTEST_SKIP() << "gs.tck / gs.nii not present in test data";
  }
  expect_legacy_to_trx_round_trip(tck, nii.string(), "trx_legacy_tck_roundtrip");
}

TEST(LegacyIo, TrkToTrxRoundTripPreservesHeaderCounts) {
  const fs::path root = test_data_root();
  const fs::path trk = root / "gs.trk";
  if (!fs::exists(trk)) {
    GTEST_SKIP() << "gs.trk not present in test data";
  }
  // A TRK carries its own spatial header, so no reference NIfTI is required.
  expect_legacy_to_trx_round_trip(trk, "", "trx_legacy_trk_roundtrip");
}
