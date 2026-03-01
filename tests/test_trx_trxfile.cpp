#include <any>
#include <memory>
#include <sstream>

#include <trx/trx.h>

#include <filesystem>
#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <set>

using namespace Eigen;
using namespace trx;
namespace fs = std::filesystem;

namespace {
template <typename DT> trx::TrxReader<DT> load_trx_dir(const fs::path &path) {
  return trx::TrxReader<DT>(path.string());
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
  trx::write_binary((root / "positions.3.float32").string(), positions);

  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> offsets(3, 1);
  offsets << 0, 2, 4;
  trx::write_binary((root / "offsets.uint32").string(), offsets);

  fs::path dpv_dir = root / "dpv";
  std::error_code ec;
  fs::create_directories(dpv_dir, ec);
  Matrix<float, Dynamic, Dynamic, RowMajor> dpv(4, 1);
  dpv << 0.0f, 0.3f, 0.6f, 0.9f;
  trx::write_binary((dpv_dir / "color.float32").string(), dpv);

  fs::path dps_dir = root / "dps";
  fs::create_directories(dps_dir, ec);
  Matrix<float, Dynamic, Dynamic, RowMajor> dps(2, 1);
  dps << 0.25f, 0.75f;
  trx::write_binary((dps_dir / "weight.float32").string(), dps);

  fs::path groups_dir = root / "groups";
  fs::create_directories(groups_dir, ec);
  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> group_vals(2, 1);
  group_vals << 0, 1;
  trx::write_binary((groups_dir / "GroupA.uint32").string(), group_vals);

  fs::path dpg_dir = root / "dpg" / "GroupA";
  fs::create_directories(dpg_dir, ec);
  Matrix<float, Dynamic, Dynamic, RowMajor> dpg(1, 1);
  dpg << 1.0f;
  trx::write_binary((dpg_dir / "mean.float32").string(), dpg);

  return root;
}

fs::path create_float_trx_dir_missing_sentinel() {
  fs::path root = make_temp_test_dir("trx_float_missing_sentinel");
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
  trx::write_binary((root / "positions.3.float32").string(), positions);

  // Offsets without sentinel (size == NB_STREAMLINES).
  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> offsets(2, 1);
  offsets << 0, 2;
  trx::write_binary((root / "offsets.uint32").string(), offsets);

  return root;
}
} // namespace

TEST(TrxFileTpp, DeepcopyEmpty) {
  trx::TrxFile<half> empty;
  auto copy = empty.deepcopy();
  EXPECT_EQ(copy->header, empty.header);
  if (copy->streamlines != nullptr) {
    EXPECT_EQ(copy->streamlines->_data.size(), 0);
    EXPECT_EQ(copy->streamlines->_offsets.size(), 0);
    EXPECT_EQ(copy->streamlines->_lengths.size(), 0);
  }
}

// Deepcopy preserves streamlines, dpv/dps, groups, and dpg shapes/content.
TEST(TrxFileTpp, DeepcopyWithGroupsDpgDpvDps) {
  const fs::path data_dir = create_float_trx_dir();

  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();
  auto copy = trx->deepcopy();

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

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

TEST(TrxFileTpp, LoadOffsetsMissingSentinel) {
  const fs::path data_dir = create_float_trx_dir_missing_sentinel();

  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();

  ASSERT_NE(trx->streamlines, nullptr);
  EXPECT_EQ(trx->streamlines->_offsets.size(), 3);
  EXPECT_EQ(trx->streamlines->_offsets(2, 0), 4u);
  EXPECT_EQ(trx->num_streamlines(), 2u);
  EXPECT_EQ(trx->num_vertices(), 4u);

  trx->close();
  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

// _copy_fixed_arrays_from copies streamlines + dpv/dps into a preallocated target.
TEST(TrxFileTpp, CopyFixedArraysFrom) {
  const fs::path data_dir = create_float_trx_dir();
  auto src_reader = load_trx_dir<float>(data_dir);
  auto *src = src_reader.get();
  const int nb_vertices = src->header["NB_VERTICES"].int_value();
  const int nb_streamlines = src->header["NB_STREAMLINES"].int_value();
  auto dst = std::make_unique<trx::TrxFile<float>>(nb_vertices, nb_streamlines, src);

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

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

TEST(TrxFileTpp, AddGroupAndDpvFromVector) {
  const int nb_vertices = 5;
  const int nb_streamlines = 3;
  trx::TrxFile<float> trx(nb_vertices, nb_streamlines);

  trx.streamlines->_offsets(0, 0) = 0;
  trx.streamlines->_offsets(1, 0) = 2;
  trx.streamlines->_offsets(2, 0) = 4;
  trx.streamlines->_offsets(3, 0) = 5;
  trx.streamlines->_lengths(0) = 2;
  trx.streamlines->_lengths(1) = 2;
  trx.streamlines->_lengths(2) = 1;

  const std::vector<uint32_t> group_indices = {0, 2};
  trx.add_group_from_indices("GroupA", group_indices);
  ASSERT_EQ(trx.groups.size(), 1u);
  auto group_it = trx.groups.find("GroupA");
  ASSERT_NE(group_it, trx.groups.end());
  EXPECT_EQ(group_it->second->_matrix.rows(), 2);
  EXPECT_EQ(group_it->second->_matrix.cols(), 1);
  EXPECT_EQ(group_it->second->_matrix(0, 0), 0u);
  EXPECT_EQ(group_it->second->_matrix(1, 0), 2u);

  const std::vector<float> dpv_values = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  trx.add_dpv_from_vector("quality", "float32", dpv_values);
  ASSERT_EQ(trx.data_per_vertex.size(), 1u);
  auto dpv_it = trx.data_per_vertex.find("quality");
  ASSERT_NE(dpv_it, trx.data_per_vertex.end());
  EXPECT_EQ(dpv_it->second->_data.rows(), nb_vertices);
  EXPECT_EQ(dpv_it->second->_data.cols(), 1);
  for (int i = 0; i < nb_vertices; ++i) {
    EXPECT_FLOAT_EQ(dpv_it->second->_data(i, 0), dpv_values[static_cast<size_t>(i)]);
  }
}

TEST(TrxFileTpp, TrxStreamFinalize) {
  auto tmp_dir = make_temp_test_dir("trx_proto");
  const fs::path out_path = tmp_dir / "proto.trx";

  TrxStream proto;
  std::vector<float> sl1 = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> sl2 = {2.0f, 0.0f, 0.0f, 2.0f, 1.0f, 0.0f, 2.0f, 2.0f, 0.0f};
  proto.push_streamline(sl1);
  proto.push_streamline(sl2);
  proto.push_group_from_indices("GroupA", {0, 1});
  proto.push_dps_from_vector("weight", "float32", std::vector<float>{0.5f, 1.5f});
  proto.push_dpv_from_vector("score", "float32", std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  proto.finalize<float>(out_path.string(), ZIP_CM_STORE);

  auto trx = load_any(out_path.string());
  EXPECT_EQ(trx.num_streamlines(), 2u);
  EXPECT_EQ(trx.num_vertices(), 5u);

  auto grp_it = trx.groups.find("GroupA");
  ASSERT_NE(grp_it, trx.groups.end());
  auto grp_mat = grp_it->second.as_matrix<uint32_t>();
  EXPECT_EQ(grp_mat.rows(), 2);
  EXPECT_EQ(grp_mat.cols(), 1);
  EXPECT_EQ(grp_mat(0, 0), 0u);
  EXPECT_EQ(grp_mat(1, 0), 1u);

  auto dps_it = trx.data_per_streamline.find("weight");
  ASSERT_NE(dps_it, trx.data_per_streamline.end());
  auto dps_mat = dps_it->second.as_matrix<float>();
  EXPECT_EQ(dps_mat.rows(), 2);
  EXPECT_EQ(dps_mat.cols(), 1);
  EXPECT_FLOAT_EQ(dps_mat(0, 0), 0.5f);
  EXPECT_FLOAT_EQ(dps_mat(1, 0), 1.5f);

  auto dpv_it = trx.data_per_vertex.find("score");
  ASSERT_NE(dpv_it, trx.data_per_vertex.end());
  auto dpv_mat = dpv_it->second.as_matrix<float>();
  EXPECT_EQ(dpv_mat.rows(), 5);
  EXPECT_EQ(dpv_mat.cols(), 1);
  EXPECT_FLOAT_EQ(dpv_mat(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(dpv_mat(4, 0), 5.0f);

  trx.close();
  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
}

TEST(TrxFileTpp, TrxStreamOnDiskMetadataAllDtypes) {
  auto tmp_dir = make_temp_test_dir("trx_ondisk_dtypes");
  const fs::path out_path = tmp_dir / "ondisk.trx";

  TrxStream proto;
  proto.set_metadata_mode(TrxStream::MetadataMode::OnDisk);

  std::vector<float> sl1 = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> sl2 = {2.0f, 0.0f, 0.0f};
  proto.push_streamline(sl1);
  proto.push_streamline(sl2);

  proto.push_dps_from_vector("w_f16", "float16", std::vector<float>{0.5f, 1.5f});
  proto.push_dps_from_vector("w_f32", "float32", std::vector<float>{0.5f, 1.5f});
  proto.push_dps_from_vector("w_f64", "float64", std::vector<double>{0.5, 1.5});

  proto.push_dpv_from_vector("s_f16", "float16", std::vector<float>{1.0f, 2.0f, 3.0f});
  proto.push_dpv_from_vector("s_f32", "float32", std::vector<float>{1.0f, 2.0f, 3.0f});
  proto.push_dpv_from_vector("s_f64", "float64", std::vector<double>{1.0, 2.0, 3.0});

  proto.finalize<float>(out_path.string(), ZIP_CM_STORE);

  auto trx = load_any(out_path.string());
  EXPECT_EQ(trx.num_streamlines(), 2u);
  EXPECT_EQ(trx.num_vertices(), 3u);

  for (const auto &key : {"w_f16", "w_f32", "w_f64"}) {
    auto it = trx.data_per_streamline.find(key);
    ASSERT_NE(it, trx.data_per_streamline.end()) << "missing dps key: " << key;
    EXPECT_EQ(it->second.rows, 2);
  }
  for (const auto &key : {"s_f16", "s_f32", "s_f64"}) {
    auto it = trx.data_per_vertex.find(key);
    ASSERT_NE(it, trx.data_per_vertex.end()) << "missing dpv key: " << key;
    EXPECT_EQ(it->second.rows, 3);
  }

  trx.close();
  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
}

TEST(TrxFileTpp, QueryAabbCounts) {
  constexpr int kStreamlineCount = 1000;
  constexpr int kInsideCount = 250;
  constexpr int kPointsPerStreamline = 5;

  const int nb_vertices = kStreamlineCount * kPointsPerStreamline;
  trx::TrxFile<float> trx(nb_vertices, kStreamlineCount);

  trx.streamlines->_offsets(0, 0) = 0;
  for (int i = 0; i < kStreamlineCount; ++i) {
    trx.streamlines->_lengths(i) = kPointsPerStreamline;
    trx.streamlines->_offsets(i + 1, 0) = (i + 1) * kPointsPerStreamline;
  }

  int cursor = 0;
  for (int i = 0; i < kStreamlineCount; ++i) {
    const bool inside = i < kInsideCount;
    for (int p = 0; p < kPointsPerStreamline; ++p, ++cursor) {
      if (inside) {
        trx.streamlines->_data(cursor, 0) = -0.8f + 0.05f * static_cast<float>(p);
        trx.streamlines->_data(cursor, 1) = 0.3f + 0.1f * static_cast<float>(p);
        trx.streamlines->_data(cursor, 2) = 0.1f + 0.05f * static_cast<float>(p);
      } else {
        trx.streamlines->_data(cursor, 0) = 0.0f;
        trx.streamlines->_data(cursor, 1) = 0.0f;
        trx.streamlines->_data(cursor, 2) = -1000.0f - static_cast<float>(i);
      }
    }
  }

  const std::array<float, 3> min_corner{ -0.9f, 0.2f, 0.05f };
  const std::array<float, 3> max_corner{ -0.1f, 1.1f, 0.55f };

  auto subset = trx.query_aabb(min_corner, max_corner);
  EXPECT_EQ(subset->num_streamlines(), static_cast<size_t>(kInsideCount));
  EXPECT_EQ(subset->num_vertices(), static_cast<size_t>(kInsideCount * kPointsPerStreamline));
  subset->close();
}

// resize() with default arguments is a no-op when sizes already match.
TEST(TrxFileTpp, ResizeNoChange) {
  const fs::path data_dir = create_float_trx_dir();
  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();
  json header_before = trx->header;
  trx->resize();
  EXPECT_EQ(trx->header, header_before);
  trx->close();

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

// resize(..., delete_dpg=true) forces close(): clears header/streamlines and drops
// dpv/dps/groups/dpg.
TEST(TrxFileTpp, ResizeDeleteDpgCloses) {
  const fs::path data_dir = create_float_trx_dir();
  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();
  trx->resize(1, -1, true);

  EXPECT_EQ(trx->header["NB_STREAMLINES"].int_value(), 0);
  EXPECT_EQ(trx->header["NB_VERTICES"].int_value(), 0);
  EXPECT_EQ(trx->data_per_group.size(), 0u);
  EXPECT_EQ(trx->groups.size(), 0u);
  EXPECT_EQ(trx->data_per_vertex.size(), 0u);
  EXPECT_EQ(trx->data_per_streamline.size(), 0u);

  trx->close();

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

TEST(TrxFileTpp, NormalizeForSaveRejectsNonMonotonicOffsets) {
  const fs::path data_dir = create_float_trx_dir();
  auto src_reader = load_trx_dir<float>(data_dir);
  auto *src = src_reader.get();

  ASSERT_GE(src->streamlines->_offsets.size(), 3);
  src->streamlines->_offsets(1) = 5;
  src->streamlines->_offsets(2) = 4;

  EXPECT_THROW(src->normalize_for_save(), trx::TrxError);

  src->close();

  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

TEST(TrxFileTpp, NormalizeForSaveRecomputesLengthsAndHeader) {
  const fs::path data_dir = create_float_trx_dir();
  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();

  ASSERT_EQ(trx->streamlines->_offsets.size(), 3);
  trx->streamlines->_lengths(0) = 99;
  trx->streamlines->_lengths(1) = 99;
  trx->header = _json_set(trx->header, "NB_STREAMLINES", 123);
  trx->header = _json_set(trx->header, "NB_VERTICES", 456);

  trx->normalize_for_save();

  EXPECT_EQ(trx->streamlines->_lengths(0), 2u);
  EXPECT_EQ(trx->streamlines->_lengths(1), 2u);
  EXPECT_EQ(trx->header["NB_STREAMLINES"].int_value(), 2);
  EXPECT_EQ(trx->header["NB_VERTICES"].int_value(), 4);

  trx->close();
  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

TEST(TrxFileTpp, LoadFromDirectoryMissingHeader) {
  // Directory exists and has files in it, but no header.json.
  // Covers the detailed error-diagnostic branch in load_from_directory (lines 980-1006).
  auto tmp_dir = make_temp_test_dir("trx_no_header");
  const fs::path dummy = tmp_dir / "positions.3.float32";
  std::ofstream f(dummy.string(), std::ios::binary);
  f.close();

  EXPECT_THROW(TrxFile<float>::load_from_directory(tmp_dir.string()), trx::TrxError);

  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
}

TEST(TrxFileTpp, TrxStreamFloat16Unbuffered) {
  // TrxStream("float16") with default unbuffered mode.
  // Covers the float16 unbuffered write path in push_streamline (lines 1642-1650)
  // and the float16 read-back loop in finalize (lines 1958-1966).
  auto tmp_dir = make_temp_test_dir("trx_f16_unbuf");
  const fs::path out_path = tmp_dir / "f16.trx";

  TrxStream proto("float16");
  std::vector<float> sl1 = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> sl2 = {2.0f, 0.0f, 0.0f};
  proto.push_streamline(sl1);
  proto.push_streamline(sl2);
  proto.finalize<float>(out_path.string(), ZIP_CM_STORE);

  auto trx = load_any(out_path.string());
  EXPECT_EQ(trx.num_streamlines(), 2u);
  EXPECT_EQ(trx.num_vertices(), 3u);
  trx.close();

  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
}

TEST(TrxFileTpp, TrxStreamFloat16Buffered) {
  // TrxStream("float16") with a small position buffer.
  // Pushing two single-point streamlines fills the buffer (6 half-values >= max 6)
  // and triggers flush_positions_buffer mid-stream (lines 1592-1603, 1660-1673).
  // finalize then calls flush_positions_buffer again with an empty buffer,
  // hitting the early-return path (lines 1592-1593).
  auto tmp_dir = make_temp_test_dir("trx_f16_buf");
  const fs::path out_path = tmp_dir / "f16_buf.trx";

  TrxStream proto("float16");
  // 12 bytes / 2 bytes per half = 6 half entries = 2 xyz triplets → flush after 2 points
  proto.set_positions_buffer_max_bytes(12);
  std::vector<float> pt = {1.0f, 2.0f, 3.0f};
  proto.push_streamline(pt);  // buffer: 3 halves
  proto.push_streamline(pt);  // buffer: 6 halves >= 6 → flush
  proto.push_streamline(pt);  // buffer: 3 halves (after flush)
  proto.finalize<float>(out_path.string(), ZIP_CM_STORE);  // flush remainder, then early-return

  auto trx = load_any(out_path.string());
  EXPECT_EQ(trx.num_streamlines(), 3u);
  EXPECT_EQ(trx.num_vertices(), 3u);
  trx.close();

  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
}
