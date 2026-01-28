#include <any>
#include <sstream>

#define private public
#include <trx/trx.h>
#undef private

#include <filesystem>
#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <set>

using namespace Eigen;
using namespace trxmmap;
namespace fs = std::filesystem;

namespace {
template <typename DT> trxmmap::TrxFile<DT> *load_trx_dir(const fs::path &path) {
  return trxmmap::load_from_directory<DT>(path.string());
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

fs::path create_float_trx_dir() {
  fs::path root = make_temp_test_dir("trx_float");
  fs::path header_path = root / "header.json";

  json::object header_obj;
  header_obj["DIMENSIONS"] = json::array{1, 1, 1};
  header_obj["NB_STREAMLINES"] = 2;
  header_obj["NB_VERTICES"] = 4;
  header_obj["VOXEL_TO_RASMM"] = json::array{
      json::array{1.0, 0.0, 0.0, 0.0},
      json::array{0.0, 1.0, 0.0, 0.0},
      json::array{0.0, 0.0, 1.0, 0.0},
      json::array{0.0, 0.0, 0.0, 1.0},
  };
  json header(header_obj);

  std::ofstream header_out(header_path.string());
  if (!header_out.is_open()) {
    throw std::runtime_error("Failed to write header.json");
  }
  header_out << header.dump() << std::endl;
  header_out.close();

  Matrix<float, Dynamic, Dynamic, RowMajor> positions(4, 3);
  positions << 0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f;
  trxmmap::write_binary((root / "positions.3.float32").string().c_str(), positions);

  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> offsets(3, 1);
  offsets << 0, 2, 4;
  trxmmap::write_binary((root / "offsets.uint32").string().c_str(), offsets);

  fs::path dpv_dir = root / "dpv";
  std::error_code ec;
  fs::create_directories(dpv_dir, ec);
  Matrix<float, Dynamic, Dynamic, RowMajor> dpv(4, 1);
  dpv << 0.0f, 0.3f, 0.6f, 0.9f;
  trxmmap::write_binary((dpv_dir / "color.float32").string().c_str(), dpv);

  fs::path dps_dir = root / "dps";
  fs::create_directories(dps_dir, ec);
  Matrix<float, Dynamic, Dynamic, RowMajor> dps(2, 1);
  dps << 0.25f, 0.75f;
  trxmmap::write_binary((dps_dir / "weight.float32").string().c_str(), dps);

  fs::path groups_dir = root / "groups";
  fs::create_directories(groups_dir, ec);
  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> group_vals(2, 1);
  group_vals << 0, 1;
  trxmmap::write_binary((groups_dir / "GroupA.uint32").string().c_str(), group_vals);

  fs::path dpg_dir = root / "dpg" / "GroupA";
  fs::create_directories(dpg_dir, ec);
  Matrix<float, Dynamic, Dynamic, RowMajor> dpg(1, 1);
  dpg << 1.0f;
  trxmmap::write_binary((dpg_dir / "mean.float32").string().c_str(), dpg);

  return root;
}
} // namespace

TEST(TrxFileTpp, DeepcopyEmpty) {
  trxmmap::TrxFile<half> empty;
  trxmmap::TrxFile<half> *copy = empty.deepcopy();
  EXPECT_EQ(copy->header, empty.header);
  if (copy->streamlines != nullptr) {
    EXPECT_EQ(copy->streamlines->_data.size(), 0);
    EXPECT_EQ(copy->streamlines->_offsets.size(), 0);
    EXPECT_EQ(copy->streamlines->_lengths.size(), 0);
  }
  delete copy;
}

// Deepcopy preserves streamlines, dpv/dps, groups, and dpg shapes/content.
TEST(TrxFileTpp, DeepcopyWithGroupsDpgDpvDps) {
  const fs::path data_dir = create_float_trx_dir();

  trxmmap::TrxFile<float> *trx = load_trx_dir<float>(data_dir);
  trxmmap::TrxFile<float> *copy = trx->deepcopy();

  EXPECT_EQ(copy->header, trx->header);
  EXPECT_EQ(copy->streamlines->_data, trx->streamlines->_data);
  EXPECT_EQ(copy->streamlines->_offsets, trx->streamlines->_offsets);
  EXPECT_EQ(copy->streamlines->_lengths, trx->streamlines->_lengths);

  EXPECT_EQ(copy->data_per_vertex.size(), trx->data_per_vertex.size());
  for (const auto &kv : trx->data_per_vertex) {
    auto it = copy->data_per_vertex.find(kv.first);
    ASSERT_NE(it, copy->data_per_vertex.end());
    EXPECT_EQ(it->second->_data.rows(), kv.second->_data.rows());
    EXPECT_EQ(it->second->_data.cols(), kv.second->_data.cols());
  }

  EXPECT_EQ(copy->data_per_streamline.size(), trx->data_per_streamline.size());
  for (const auto &kv : trx->data_per_streamline) {
    auto it = copy->data_per_streamline.find(kv.first);
    ASSERT_NE(it, copy->data_per_streamline.end());
    EXPECT_EQ(it->second->_matrix.rows(), kv.second->_matrix.rows());
    EXPECT_EQ(it->second->_matrix.cols(), kv.second->_matrix.cols());
  }

  EXPECT_EQ(copy->groups.size(), trx->groups.size());
  for (const auto &kv : trx->groups) {
    auto it = copy->groups.find(kv.first);
    ASSERT_NE(it, copy->groups.end());
    EXPECT_EQ(it->second->_matrix.rows(), kv.second->_matrix.rows());
    EXPECT_EQ(it->second->_matrix.cols(), kv.second->_matrix.cols());
    EXPECT_EQ(it->second->_matrix, kv.second->_matrix);
  }

  EXPECT_EQ(copy->data_per_group.size(), trx->data_per_group.size());
  for (const auto &grp : trx->data_per_group) {
    auto it = copy->data_per_group.find(grp.first);
    ASSERT_NE(it, copy->data_per_group.end());
    EXPECT_EQ(it->second.size(), grp.second.size());
    for (const auto &dpg : grp.second) {
      auto it_dpg = it->second.find(dpg.first);
      ASSERT_NE(it_dpg, it->second.end());
      EXPECT_EQ(it_dpg->second->_matrix.rows(), dpg.second->_matrix.rows());
      EXPECT_EQ(it_dpg->second->_matrix.cols(), dpg.second->_matrix.cols());
    }
  }

  trx->close();
  copy->close();
  delete trx;
  delete copy;

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

// _copy_fixed_arrays_from copies streamlines + dpv/dps into a preallocated target.
TEST(TrxFileTpp, CopyFixedArraysFrom) {
  const fs::path data_dir = create_float_trx_dir();
  trxmmap::TrxFile<float> *src = load_trx_dir<float>(data_dir);
  const int nb_vertices = src->header["NB_VERTICES"].int_value();
  const int nb_streamlines = src->header["NB_STREAMLINES"].int_value();
  trxmmap::TrxFile<float> *dst = new trxmmap::TrxFile<float>(nb_vertices, nb_streamlines, src);

  dst->_copy_fixed_arrays_from(src, 0, 0, nb_streamlines);

  EXPECT_EQ(dst->streamlines->_data, src->streamlines->_data);
  EXPECT_EQ(dst->streamlines->_offsets, src->streamlines->_offsets);
  EXPECT_EQ(dst->streamlines->_lengths, src->streamlines->_lengths);

  EXPECT_EQ(dst->data_per_vertex.size(), src->data_per_vertex.size());
  for (const auto &kv : src->data_per_vertex) {
    auto it = dst->data_per_vertex.find(kv.first);
    ASSERT_NE(it, dst->data_per_vertex.end());
    EXPECT_EQ(it->second->_data, kv.second->_data);
  }

  EXPECT_EQ(dst->data_per_streamline.size(), src->data_per_streamline.size());
  for (const auto &kv : src->data_per_streamline) {
    auto it = dst->data_per_streamline.find(kv.first);
    ASSERT_NE(it, dst->data_per_streamline.end());
    EXPECT_EQ(it->second->_matrix, kv.second->_matrix);
  }

  src->close();
  dst->close();
  delete src;
  delete dst;

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

// resize() with default arguments is a no-op when sizes already match.
TEST(TrxFileTpp, ResizeNoChange) {
  const fs::path data_dir = create_float_trx_dir();
  trxmmap::TrxFile<float> *trx = load_trx_dir<float>(data_dir);
  json header_before = trx->header;
  trx->resize();
  EXPECT_EQ(trx->header, header_before);
  trx->close();
  delete trx;

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

// resize(..., delete_dpg=true) forces close(): clears header/streamlines and drops
// dpv/dps/groups/dpg.
TEST(TrxFileTpp, ResizeDeleteDpgCloses) {
  const fs::path data_dir = create_float_trx_dir();
  trxmmap::TrxFile<float> *trx = load_trx_dir<float>(data_dir);
  trx->resize(1, -1, true);

  EXPECT_EQ(trx->header["NB_STREAMLINES"].int_value(), 0);
  EXPECT_EQ(trx->header["NB_VERTICES"].int_value(), 0);
  EXPECT_EQ(trx->data_per_group.size(), 0u);
  EXPECT_EQ(trx->groups.size(), 0u);
  EXPECT_EQ(trx->data_per_vertex.size(), 0u);
  EXPECT_EQ(trx->data_per_streamline.size(), 0u);

  delete trx;

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}
