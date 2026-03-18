#include <any>
#include <memory>
#include <sstream>

#include <trx/trx.h>

#include <filesystem>
#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/resource.h>
#endif

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

fs::path create_float64_trx_dir() {
  fs::path root = make_temp_test_dir("trx_float64");
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

  Matrix<double, Dynamic, Dynamic, RowMajor> positions(4, 3);
  positions << 0.125, 1.5, 2.875, 3.25, 4.625, 5.75, 6.0, 7.125, 8.5, 9.875, 10.25, 11.625;
  trx::write_binary((root / "positions.3.float64").string(), positions);

  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> offsets(3, 1);
  offsets << 0, 2, 4;
  trx::write_binary((root / "offsets.uint32").string(), offsets);

  return root;
}

fs::path create_connectivity_fixture_trx() {
  fs::path root = make_temp_test_dir("trx_connectivity_fixture");
  const fs::path out_path = root / "connectivity.trx";

  TrxStream stream;
  for (int i = 0; i < 5; ++i) {
    stream.push_streamline(std::vector<float>{0.0f, 0.0f, 0.0f});
  }

  // Streamline memberships:
  // s0 -> A
  // s1 -> B
  // s2 -> C
  // s3 -> A,B
  // s4 -> B,C
  // Include a duplicate index in A to verify de-duplication per streamline.
  stream.push_group_from_indices("A", {0, 3, 3});
  stream.push_group_from_indices("B", {1, 3, 4});
  stream.push_group_from_indices("C", {2, 4});
  stream.push_dps_from_vector("weight", "float32", std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  stream.finalize<float>(out_path.string(), ZIP_CM_STORE);

  return out_path;
}

fs::path create_many_groups_fixture_trx(size_t group_count, size_t streamline_count = 16) {
  fs::path root = make_temp_test_dir("trx_many_groups_fixture");
  const fs::path out_path = root / "many_groups.trx";

  TrxStream stream;
  for (size_t i = 0; i < streamline_count; ++i) {
    stream.push_streamline(std::vector<float>{0.0f, 0.0f, 0.0f});
  }

  for (size_t g = 0; g < group_count; ++g) {
    const std::string name = "G" + std::to_string(g);
    const uint32_t sid = static_cast<uint32_t>(g % streamline_count);
    stream.push_group_from_indices(name, {sid});
  }
  stream.finalize<float>(out_path.string(), ZIP_CM_STORE);

  return out_path;
}

fs::path create_connectivity_bad_dps_fixture_dir() {
  fs::path root = make_temp_test_dir("trx_connectivity_bad_dps_fixture");
  fs::path header_path = root / "header.json";

  json::object header_obj;
  header_obj["DIMENSIONS"] = json::array{1, 1, 1};
  header_obj["NB_STREAMLINES"] = 2;
  header_obj["NB_VERTICES"] = 2;
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

  Matrix<float, Dynamic, Dynamic, RowMajor> positions(2, 3);
  positions << 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f;
  trx::write_binary((root / "positions.3.float32").string(), positions);

  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> offsets(3, 1);
  offsets << 0, 1, 2;
  trx::write_binary((root / "offsets.uint32").string(), offsets);

  fs::path groups_dir = root / "groups";
  std::error_code ec;
  fs::create_directories(groups_dir, ec);
  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> group_vals(2, 1);
  group_vals << 0, 1;
  trx::write_binary((groups_dir / "A.uint32").string(), group_vals);

  fs::path dps_dir = root / "dps";
  fs::create_directories(dps_dir, ec);
  Matrix<float, Dynamic, Dynamic, RowMajor> dps(2, 2);
  dps << 1.0f, 2.0f, 3.0f, 4.0f;
  trx::write_binary((dps_dir / "weights2d.2.float32").string(), dps);

  return root;
}

fs::path create_connectivity_no_groups_fixture_trx(size_t streamline_count = 3) {
  fs::path root = make_temp_test_dir("trx_connectivity_nogroups");
  const fs::path out_path = root / "nogroups.trx";

  TrxStream stream;
  for (size_t i = 0; i < streamline_count; ++i) {
    stream.push_streamline(std::vector<float>{0.0f, 0.0f, 0.0f});
  }
  stream.finalize<float>(out_path.string(), ZIP_CM_STORE);
  return out_path;
}

fs::path create_connectivity_empty_group_fixture_dir() {
  fs::path root = make_temp_test_dir("trx_connectivity_empty_group");
  fs::path header_path = root / "header.json";

  json::object header_obj;
  header_obj["DIMENSIONS"] = json::array{1, 1, 1};
  header_obj["NB_STREAMLINES"] = 3;
  header_obj["NB_VERTICES"] = 3;
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

  Matrix<float, Dynamic, Dynamic, RowMajor> positions(3, 3);
  positions << 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f;
  trx::write_binary((root / "positions.3.float32").string(), positions);

  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> offsets(4, 1);
  offsets << 0, 1, 2, 3;
  trx::write_binary((root / "offsets.uint32").string(), offsets);

  std::error_code ec;
  const fs::path groups_dir = root / "groups";
  fs::create_directories(groups_dir, ec);
  // Empty file -> valid empty group membership list.
  std::ofstream((groups_dir / "Empty.uint32").string(), std::ios::binary | std::ios::trunc).close();

  return root;
}

fs::path create_connectivity_out_of_range_group_fixture_dir() {
  fs::path root = make_temp_test_dir("trx_connectivity_out_of_range");
  fs::path header_path = root / "header.json";

  json::object header_obj;
  header_obj["DIMENSIONS"] = json::array{1, 1, 1};
  header_obj["NB_STREAMLINES"] = 3;
  header_obj["NB_VERTICES"] = 3;
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

  Matrix<float, Dynamic, Dynamic, RowMajor> positions(3, 3);
  positions << 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f;
  trx::write_binary((root / "positions.3.float32").string(), positions);

  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> offsets(4, 1);
  offsets << 0, 1, 2, 3;
  trx::write_binary((root / "offsets.uint32").string(), offsets);

  std::error_code ec;
  const fs::path groups_dir = root / "groups";
  fs::create_directories(groups_dir, ec);
  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> group_vals(3, 1);
  // Index 99 is out-of-range and should be ignored by connectivity builder.
  group_vals << 0, 2, 99;
  trx::write_binary((groups_dir / "A.uint32").string(), group_vals);

  return root;
}

fs::path create_connectivity_large_lazy_group_fixture_dir(size_t streamline_count = 100000,
                                                          size_t duplicate_factor = 5) {
  fs::path root = make_temp_test_dir("trx_connectivity_large_lazy_group");
  fs::path header_path = root / "header.json";

  json::object header_obj;
  header_obj["DIMENSIONS"] = json::array{1, 1, 1};
  header_obj["NB_STREAMLINES"] = static_cast<int>(streamline_count);
  header_obj["NB_VERTICES"] = static_cast<int>(streamline_count);
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

  Matrix<float, Dynamic, Dynamic, RowMajor> positions(static_cast<Eigen::Index>(streamline_count), 3);
  for (size_t i = 0; i < streamline_count; ++i) {
    const auto row = static_cast<Eigen::Index>(i);
    positions(row, 0) = static_cast<float>(i);
    positions(row, 1) = 0.0f;
    positions(row, 2) = 0.0f;
  }
  trx::write_binary((root / "positions.3.float32").string(), positions);

  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> offsets(static_cast<Eigen::Index>(streamline_count + 1), 1);
  for (size_t i = 0; i <= streamline_count; ++i) {
    offsets(static_cast<Eigen::Index>(i), 0) = static_cast<uint32_t>(i);
  }
  trx::write_binary((root / "offsets.uint32").string(), offsets);

  std::error_code ec;
  const fs::path groups_dir = root / "groups";
  fs::create_directories(groups_dir, ec);
  const size_t n_ids = streamline_count * duplicate_factor;
  Matrix<uint32_t, Dynamic, Dynamic, RowMajor> group_vals(static_cast<Eigen::Index>(n_ids), 1);
  for (size_t i = 0; i < n_ids; ++i) {
    group_vals(static_cast<Eigen::Index>(i), 0) = static_cast<uint32_t>(i % streamline_count);
  }
  trx::write_binary((groups_dir / "All.uint32").string(), group_vals);

  return root;
}

#if !defined(_WIN32) && !defined(_WIN64)
struct NoFileLimitRestoreGuard {
  rlimit old_limit{};
  bool active = false;

  ~NoFileLimitRestoreGuard() {
    if (active) {
      static_cast<void>(setrlimit(RLIMIT_NOFILE, &old_limit));
    }
  }
};
#endif
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
    const auto *src_group = trx->get_group_members(kv.first);
    const auto *copy_group = copy->get_group_members(kv.first);
    ASSERT_NE(src_group, nullptr);
    ASSERT_NE(copy_group, nullptr);
    EXPECT_EQ(copy_group->_matrix.rows(), src_group->_matrix.rows());
    EXPECT_EQ(copy_group->_matrix.cols(), src_group->_matrix.cols());
    EXPECT_EQ(copy_group->_matrix, src_group->_matrix);
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

TEST(TrxFileTpp, MetadataArraysAreUnmappedAfterLoad) {
  const fs::path data_dir = create_float_trx_dir();
  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();

  for (const auto &kv : trx->groups) {
    EXPECT_EQ(kv.second, nullptr);
    const auto *group = trx->get_group_members(kv.first);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->mmap.file_handle(), mio::invalid_handle);
    EXPECT_FALSE(group->_matrix_owned.empty());
  }
  for (const auto &kv : trx->data_per_streamline) {
    ASSERT_NE(kv.second, nullptr);
    EXPECT_EQ(kv.second->mmap.file_handle(), mio::invalid_handle);
    EXPECT_FALSE(kv.second->_matrix_owned.empty());
  }
  for (const auto &kv : trx->data_per_vertex) {
    ASSERT_NE(kv.second, nullptr);
    EXPECT_NE(kv.second->mmap_pos.file_handle(), mio::invalid_handle);
    EXPECT_TRUE(kv.second->_data_owned.empty());
  }
  for (const auto &group_kv : trx->data_per_group) {
    for (const auto &field_kv : group_kv.second) {
      ASSERT_NE(field_kv.second, nullptr);
      EXPECT_EQ(field_kv.second->mmap.file_handle(), mio::invalid_handle);
      EXPECT_FALSE(field_kv.second->_matrix_owned.empty());
    }
  }

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

TEST(TrxFileTpp, ComputeGroupConnectivityFromSyntheticFileStreamlineCount) {
  const fs::path trx_path = create_connectivity_fixture_trx();
  auto reader = trx::TrxReader<float>(trx_path.string());
  auto *trx = reader.get();

  const auto result = trx->compute_group_connectivity(ConnectivityMeasure::StreamlineCount);
  ASSERT_EQ(result.group_names.size(), 3u);
  EXPECT_EQ(result.group_names[0], "A");
  EXPECT_EQ(result.group_names[1], "B");
  EXPECT_EQ(result.group_names[2], "C");

  // Packed upper triangle for groups [A, B, C]:
  // [AA, AB, AC, BB, BC, CC]
  const std::vector<uint64_t> expected_counts = {2, 1, 0, 3, 1, 2};
  const std::vector<double> expected_values = {2.0, 1.0, 0.0, 3.0, 1.0, 2.0};
  EXPECT_EQ(result.streamline_count_upper, expected_counts);
  EXPECT_EQ(result.value_upper.size(), expected_values.size());
  for (size_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_DOUBLE_EQ(result.value_upper[i], expected_values[i]);
  }

  trx->close();
  std::error_code ec;
  fs::remove_all(trx_path.parent_path(), ec);
}

TEST(TrxFileTpp, ComputeGroupConnectivityFromSyntheticFileDpsWeighted) {
  const fs::path trx_path = create_connectivity_fixture_trx();
  auto reader = trx::TrxReader<float>(trx_path.string());
  auto *trx = reader.get();

  const auto result = trx->compute_group_connectivity(ConnectivityMeasure::DpsSum, "weight");
  ASSERT_EQ(result.group_names.size(), 3u);
  EXPECT_EQ(result.group_names[0], "A");
  EXPECT_EQ(result.group_names[1], "B");
  EXPECT_EQ(result.group_names[2], "C");

  // Same connectivity pattern as the count test, weighted by DPS:
  // weights [1,2,3,4,5] for streamlines [s0..s4]
  // Packed [AA, AB, AC, BB, BC, CC] -> [5, 4, 0, 11, 5, 8]
  const std::vector<uint64_t> expected_counts = {2, 1, 0, 3, 1, 2};
  const std::vector<double> expected_values = {5.0, 4.0, 0.0, 11.0, 5.0, 8.0};
  EXPECT_EQ(result.streamline_count_upper, expected_counts);
  EXPECT_EQ(result.value_upper.size(), expected_values.size());
  for (size_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_DOUBLE_EQ(result.value_upper[i], expected_values[i]);
  }

  trx->close();
  std::error_code ec;
  fs::remove_all(trx_path.parent_path(), ec);
}

TEST(TrxFileTpp, ComputeGroupConnectivityDpsSumRequiresFieldName) {
  const fs::path trx_path = create_connectivity_fixture_trx();
  auto reader = trx::TrxReader<float>(trx_path.string());
  auto *trx = reader.get();

  EXPECT_THROW(static_cast<void>(trx->compute_group_connectivity(ConnectivityMeasure::DpsSum, "")),
               trx::TrxArgumentError);

  trx->close();
  std::error_code ec;
  fs::remove_all(trx_path.parent_path(), ec);
}

TEST(TrxFileTpp, ComputeGroupConnectivityDpsSumRejectsNonScalarDps) {
  const fs::path data_dir = create_connectivity_bad_dps_fixture_dir();
  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();
  EXPECT_THROW(static_cast<void>(trx->compute_group_connectivity(ConnectivityMeasure::DpsSum, "weights2d")),
               trx::TrxFormatError);

  trx->close();
  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

TEST(TrxFileTpp, ComputeGroupConnectivityNoGroupsAndNoMemberships) {
  {
    const fs::path trx_path = create_connectivity_no_groups_fixture_trx(3);
    auto reader = trx::TrxReader<float>(trx_path.string());
    auto *trx = reader.get();

    const auto result = trx->compute_group_connectivity(ConnectivityMeasure::StreamlineCount);
    EXPECT_TRUE(result.group_names.empty());
    EXPECT_TRUE(result.streamline_count_upper.empty());
    EXPECT_TRUE(result.value_upper.empty());

    trx->close();
    std::error_code ec;
    fs::remove_all(trx_path.parent_path(), ec);
  }

  {
    const fs::path data_dir = create_connectivity_empty_group_fixture_dir();
    auto reader = load_trx_dir<float>(data_dir);
    auto *trx = reader.get();

    const auto result = trx->compute_group_connectivity(ConnectivityMeasure::StreamlineCount);
    ASSERT_EQ(result.group_names.size(), 1u);
    EXPECT_EQ(result.group_names[0], "Empty");
    ASSERT_EQ(result.streamline_count_upper.size(), 1u);
    ASSERT_EQ(result.value_upper.size(), 1u);
    EXPECT_EQ(result.streamline_count_upper[0], 0u);
    EXPECT_DOUBLE_EQ(result.value_upper[0], 0.0);

    trx->close();
    std::error_code ec;
    fs::remove_all(data_dir, ec);
  }
}

TEST(TrxFileTpp, ComputeGroupConnectivityIgnoresOutOfRangeGroupIds) {
  const fs::path data_dir = create_connectivity_out_of_range_group_fixture_dir();
  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();

  const auto result = trx->compute_group_connectivity(ConnectivityMeasure::StreamlineCount);
  ASSERT_EQ(result.group_names.size(), 1u);
  EXPECT_EQ(result.group_names[0], "A");
  ASSERT_EQ(result.streamline_count_upper.size(), 1u);
  ASSERT_EQ(result.value_upper.size(), 1u);
  // Only ids {0,2} are valid for NB_STREAMLINES=3; id 99 is ignored.
  EXPECT_EQ(result.streamline_count_upper[0], 2u);
  EXPECT_DOUBLE_EQ(result.value_upper[0], 2.0);

  trx->close();
  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

TEST(TrxFileTpp, ComputeGroupConnectivityLargeLazyGroupFallbackPath) {
  constexpr size_t kStreamlines = 100000;
  constexpr size_t kDuplicateFactor = 5;

  const fs::path data_dir = create_connectivity_large_lazy_group_fixture_dir(kStreamlines, kDuplicateFactor);
  auto reader = load_trx_dir<float>(data_dir);
  auto *trx = reader.get();

  auto it = trx->groups.find("All");
  ASSERT_NE(it, trx->groups.end());
  ASSERT_EQ(it->second, nullptr) << "Expected lazy group backing before compute.";

  const auto result = trx->compute_group_connectivity(ConnectivityMeasure::StreamlineCount);
  ASSERT_EQ(result.group_names.size(), 1u);
  EXPECT_EQ(result.group_names[0], "All");
  ASSERT_EQ(result.streamline_count_upper.size(), 1u);
  ASSERT_EQ(result.value_upper.size(), 1u);
  // Duplicate ids must be de-duplicated per-streamline before pair accumulation.
  EXPECT_EQ(result.streamline_count_upper[0], kStreamlines);
  EXPECT_DOUBLE_EQ(result.value_upper[0], static_cast<double>(kStreamlines));

  // compute_group_connectivity should be able to consume lazy group backing directly.
  EXPECT_EQ(trx->groups["All"], nullptr);

  trx->close();
  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

#if !defined(_WIN32) && !defined(_WIN64)
TEST(TrxFileTpp, LazyLoadManyGroupsUnderTightFileDescriptorLimit) {
  const fs::path trx_path = create_many_groups_fixture_trx(400, 32);
  auto reader = trx::TrxReader<float>(trx_path.string());
  auto *trx = reader.get();
  ASSERT_EQ(trx->groups.size(), 400u);

  NoFileLimitRestoreGuard guard;
  if (getrlimit(RLIMIT_NOFILE, &guard.old_limit) != 0) {
    GTEST_SKIP() << "getrlimit(RLIMIT_NOFILE) failed: errno=" << errno << " (" << std::strerror(errno) << ")";
  }

  const rlim_t requested_soft_limit = 64;
  if (guard.old_limit.rlim_cur <= requested_soft_limit) {
    GTEST_SKIP() << "Current soft RLIMIT_NOFILE is already tight (" << guard.old_limit.rlim_cur << ")";
  }
  if (guard.old_limit.rlim_max < requested_soft_limit) {
    GTEST_SKIP() << "Hard RLIMIT_NOFILE too low for this test setup: " << guard.old_limit.rlim_max;
  }

  rlimit tight = guard.old_limit;
  tight.rlim_cur = requested_soft_limit;
  if (setrlimit(RLIMIT_NOFILE, &tight) != 0) {
    GTEST_SKIP() << "setrlimit(RLIMIT_NOFILE) failed: errno=" << errno << " (" << std::strerror(errno) << ")";
  }
  guard.active = true;

  for (const auto &kv : trx->groups) {
    const auto *group = trx->get_group_members(kv.first);
    ASSERT_NE(group, nullptr);
    // Regression guard: lazy load should materialize and close each backing file.
    EXPECT_EQ(group->mmap.file_handle(), mio::invalid_handle);
    EXPECT_FALSE(group->_matrix_owned.empty());
  }

  // Also exercise the estimator over many groups in the same tight-fd regime.
  const auto connectivity = trx->compute_group_connectivity(ConnectivityMeasure::StreamlineCount);
  EXPECT_EQ(connectivity.group_names.size(), 400u);
  EXPECT_EQ(connectivity.streamline_count_upper.size(), (400u * 401u) / 2u);

  trx->close();
  std::error_code ec;
  fs::remove_all(trx_path.parent_path(), ec);
}
#endif

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

TEST(TrxFileTpp, LoadFloat32PositionsFromFloat16Chunked) {
  auto tmp_dir = make_temp_test_dir("trx_load_f32_chunked");
  const fs::path out_path = tmp_dir / "f16_chunked.trx";

  TrxStream proto("float16");
  proto.push_streamline(std::vector<float>{0.1f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f});
  proto.push_streamline(std::vector<float>{6.7f, 7.8f, 8.9f});
  proto.finalize<float>(out_path.string(), ZIP_CM_STORE);

  LoadFloat32Options options;
  options.chunk_rows = 1; // force chunked loop execution
  auto trx_f32 = load_float32_positions(out_path.string(), options);
  ASSERT_TRUE(trx_f32);
  ASSERT_TRUE(trx_f32->streamlines);
  ASSERT_EQ(trx_f32->num_streamlines(), 2u);
  ASSERT_EQ(trx_f32->num_vertices(), 3u);

  EXPECT_NEAR(trx_f32->streamlines->_data(0, 0), 0.1f, 2e-3f);
  EXPECT_NEAR(trx_f32->streamlines->_data(1, 0), 3.4f, 2e-3f);
  EXPECT_NEAR(trx_f32->streamlines->_data(2, 2), 8.9f, 2e-3f);

  trx_f32->close();
  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
}

TEST(TrxFileTpp, LoadFloat32PositionsPassthroughFromFloat32) {
  auto tmp_dir = make_temp_test_dir("trx_load_f32_passthrough");
  const fs::path out_path = tmp_dir / "f32_passthrough.trx";

  TrxStream proto("float32");
  proto.push_streamline(std::vector<float>{0.25f, 1.25f, 2.25f, 3.25f, 4.25f, 5.25f});
  proto.push_streamline(std::vector<float>{6.25f, 7.25f, 8.25f});
  proto.finalize<float>(out_path.string(), ZIP_CM_STORE);

  auto via_api = load_float32_positions(out_path.string());
  auto direct = load<float>(out_path.string());
  ASSERT_TRUE(via_api);
  ASSERT_TRUE(direct);
  ASSERT_TRUE(via_api->streamlines);
  ASSERT_TRUE(direct->streamlines);

  EXPECT_EQ(via_api->num_streamlines(), direct->num_streamlines());
  EXPECT_EQ(via_api->num_vertices(), direct->num_vertices());
  EXPECT_EQ(via_api->streamlines->_data.rows(), direct->streamlines->_data.rows());
  EXPECT_EQ(via_api->streamlines->_data.cols(), direct->streamlines->_data.cols());
  EXPECT_EQ(via_api->streamlines->_data, direct->streamlines->_data);
  EXPECT_EQ(via_api->streamlines->_offsets, direct->streamlines->_offsets);
  EXPECT_EQ(via_api->streamlines->_lengths, direct->streamlines->_lengths);

  via_api->close();
  direct->close();
  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
}

TEST(TrxFileTpp, LoadFloat32PositionsFromFloat64ChunkRowsZeroClamped) {
  const fs::path data_dir = create_float64_trx_dir();

  LoadFloat32Options options;
  options.chunk_rows = 0; // should clamp to 1 and still convert all vertices
  auto trx_f32 = load_float32_positions(data_dir.string(), options);
  ASSERT_TRUE(trx_f32);
  ASSERT_TRUE(trx_f32->streamlines);
  ASSERT_EQ(trx_f32->num_streamlines(), 2u);
  ASSERT_EQ(trx_f32->num_vertices(), 4u);

  EXPECT_FLOAT_EQ(trx_f32->streamlines->_data(0, 0), 0.125f);
  EXPECT_FLOAT_EQ(trx_f32->streamlines->_data(1, 1), 4.625f);
  EXPECT_FLOAT_EQ(trx_f32->streamlines->_data(2, 2), 8.5f);
  EXPECT_FLOAT_EQ(trx_f32->streamlines->_data(3, 0), 9.875f);
  EXPECT_EQ(trx_f32->streamlines->_offsets(0, 0), 0u);
  EXPECT_EQ(trx_f32->streamlines->_offsets(1, 0), 2u);
  EXPECT_EQ(trx_f32->streamlines->_offsets(2, 0), 4u);

  trx_f32->close();
  std::error_code ec;
  fs::remove_all(data_dir, ec);
}

// Regression test for add_dps_from_vector / add_dpv_from_vector when the requested
// on-disk dtype differs from the TrxFile template parameter DT.
//
// Before the fix, TrxFile<half>::add_dps_from_vector(..., "float32", ...) would:
//   1. Allocate a mmap sized for N*4 bytes (float32)
//   2. Map an Eigen::Matrix<half> over it (only covers N*2 bytes)
//   3. Write N half values into the first N*2 bytes, leaving N*2 bytes as zeros
// The on-disk file then contained [half-encoded values][zeros], which was garbage
// when read back as float32 (each float32 element spanned one half + two zero bytes,
// producing near-zero denormals instead of the original values).
//
// The same bug affected add_dpv_from_vector.
TEST(TrxFileTpp, AddDpsFromVectorCrossDtypeHalfFileFloat32Data) {
  // Build a minimal TrxFile<half> with a backing directory, then add float32 DPS.
  const int nb_streamlines = 4;
  const int nb_vertices = 4;
  trx::TrxFile<half> trx(nb_vertices, nb_streamlines);

  trx.streamlines->_offsets(0, 0) = 0;
  trx.streamlines->_offsets(1, 0) = 1;
  trx.streamlines->_offsets(2, 0) = 2;
  trx.streamlines->_offsets(3, 0) = 3;
  trx.streamlines->_offsets(4, 0) = 4;
  for (int i = 0; i < nb_streamlines; ++i)
    trx.streamlines->_lengths(i) = 1;

  // Typical tractography weights: mix of small and large float32 values that
  // cannot be represented exactly as float16 (to catch byte-level misinterpretation).
  const std::vector<float> dps_values = {1.5f, 26880.0f, 0.03125f, 1024.5f};

  trx.add_dps_from_vector("weight", "float32", dps_values);

  ASSERT_EQ(trx.data_per_streamline.size(), 1u);
  auto it = trx.data_per_streamline.find("weight");
  ASSERT_NE(it, trx.data_per_streamline.end());

  // In-memory matrix is DT=half; values are cross-cast from float32 → half.
  // Check that each value is close to the original (within float16 relative precision ~0.1%).
  const auto &mat = it->second->_matrix;
  ASSERT_EQ(mat.rows(), nb_streamlines);
  for (int i = 0; i < nb_streamlines; ++i) {
    const float expected = dps_values[static_cast<size_t>(i)];
    const float actual = static_cast<float>(mat(i, 0));
    EXPECT_NEAR(actual, expected, std::abs(expected) * 0.002f)
        << "DPS value at index " << i << " is corrupted: expected " << expected
        << " got " << actual;
  }

  trx.close();
}

TEST(TrxFileTpp, AddDpvFromVectorCrossDtypeHalfFileFloat32Data) {
  const int nb_streamlines = 2;
  const int nb_vertices = 4;
  trx::TrxFile<half> trx(nb_vertices, nb_streamlines);

  trx.streamlines->_offsets(0, 0) = 0;
  trx.streamlines->_offsets(1, 0) = 2;
  trx.streamlines->_offsets(2, 0) = 4;
  trx.streamlines->_lengths(0) = 2;
  trx.streamlines->_lengths(1) = 2;

  const std::vector<float> dpv_values = {1.5f, 26880.0f, 0.03125f, 1024.5f};

  trx.add_dpv_from_vector("score", "float32", dpv_values);

  ASSERT_EQ(trx.data_per_vertex.size(), 1u);
  auto it = trx.data_per_vertex.find("score");
  ASSERT_NE(it, trx.data_per_vertex.end());

  const auto &mat = it->second->_data;
  ASSERT_EQ(mat.rows(), nb_vertices);
  for (int i = 0; i < nb_vertices; ++i) {
    const float expected = dpv_values[static_cast<size_t>(i)];
    const float actual = static_cast<float>(mat(i, 0));
    EXPECT_NEAR(actual, expected, std::abs(expected) * 0.002f)
        << "DPV value at index " << i << " is corrupted: expected " << expected
        << " got " << actual;
  }

  trx.close();
}

// Regression test: TrxStream("float16") InMemory mode with float32 DPS/DPV.
//
// Before the fix, finalize<half>() routed InMemory DPS through
// TrxFile<half>::add_dps_from_vector(..., "float32", ...) which stored half-precision
// bytes in a file labelled float32.  On reload the float32 reader would misinterpret
// those bytes as pairs of zero-padded halves, giving near-zero denormals instead of
// the original values.
TEST(TrxFileTpp, TrxStreamFloat16InMemoryFloat32DpsRoundtrip) {
  auto tmp_dir = make_temp_test_dir("trx_f16_dps_roundtrip");
  const fs::path out_path = tmp_dir / "f16_dps.trx";

  // 3 single-point streamlines, 3 vertices total.
  TrxStream proto("float16");
  proto.push_streamline(std::vector<float>{0.0f, 0.0f, 0.0f});
  proto.push_streamline(std::vector<float>{1.0f, 0.0f, 0.0f});
  proto.push_streamline(std::vector<float>{2.0f, 0.0f, 0.0f});

  // DPS: float32 values that are not exactly representable as float16, so any
  // naive byte-reinterpretation (the old bug) produces obviously wrong results.
  const std::vector<float> dps_in = {1.5f, 26880.0f, 0.03125f};
  proto.push_dps_from_vector("weight", "float32", dps_in);

  // DPV: same values, one per vertex.
  const std::vector<float> dpv_in = {1.5f, 26880.0f, 0.03125f};
  proto.push_dpv_from_vector("score", "float32", dpv_in);

  proto.finalize<half>(out_path.string(), ZIP_CM_STORE);

  // Load via AnyTrxFile so as_matrix<float>() reads the on-disk dtype directly.
  auto trx = load_any(out_path.string());
  EXPECT_EQ(trx.num_streamlines(), 3u);
  EXPECT_EQ(trx.num_vertices(), 3u);

  // DPS round-trip.
  auto dps_it = trx.data_per_streamline.find("weight");
  ASSERT_NE(dps_it, trx.data_per_streamline.end());
  auto dps_mat = dps_it->second.as_matrix<float>();
  ASSERT_EQ(dps_mat.rows(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(dps_mat(i, 0), dps_in[static_cast<size_t>(i)])
        << "DPS value at index " << i << " corrupted after float16-stream round-trip";
  }

  // DPV round-trip.
  auto dpv_it = trx.data_per_vertex.find("score");
  ASSERT_NE(dpv_it, trx.data_per_vertex.end());
  auto dpv_mat = dpv_it->second.as_matrix<float>();
  ASSERT_EQ(dpv_mat.rows(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(dpv_mat(i, 0), dpv_in[static_cast<size_t>(i)])
        << "DPV value at index " << i << " corrupted after float16-stream round-trip";
  }

  trx.close();
  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
}
