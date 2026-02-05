#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <zip.h>
#include <trx/trx.h>
#include <typeinfo>

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

fs::path resolve_memmap_test_data_dir(const std::string &root_dir) {
  fs::path root(root_dir);
  fs::path memmap_dir = root / "memmap_test_data";
  if (fs::exists(memmap_dir)) {
    return memmap_dir;
  }
  return root;
}

struct TestTrxFixture {
  fs::path root_dir;
  std::string path;
  std::string dir_path;
  json expected_header;
  int nb_vertices;
  int nb_streamlines;

  TestTrxFixture() = default;
  TestTrxFixture(const TestTrxFixture &) = delete;
  TestTrxFixture &operator=(const TestTrxFixture &) = delete;

  TestTrxFixture(TestTrxFixture &&other) noexcept
      : root_dir(std::move(other.root_dir)),
        path(std::move(other.path)),
        dir_path(std::move(other.dir_path)),
        expected_header(std::move(other.expected_header)),
        nb_vertices(other.nb_vertices),
        nb_streamlines(other.nb_streamlines) {
    other.root_dir.clear();
  }

  TestTrxFixture &operator=(TestTrxFixture &&other) noexcept {
    if (this != &other) {
      cleanup();
      root_dir = std::move(other.root_dir);
      path = std::move(other.path);
      dir_path = std::move(other.dir_path);
      expected_header = std::move(other.expected_header);
      nb_vertices = other.nb_vertices;
      nb_streamlines = other.nb_streamlines;
      other.root_dir.clear();
    }
    return *this;
  }

  ~TestTrxFixture() { cleanup(); }

  void cleanup() {
    std::error_code ec;
    if (!root_dir.empty()) {
      fs::remove_all(root_dir, ec);
      if (ec) {
        std::cerr << "Failed to clean up test directory " << root_dir.string() << ": " << ec.message() << std::endl;
      }
      root_dir.clear();
    }
  }
};

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

TestTrxFixture create_fixture() {

  TestTrxFixture fixture;
  fs::path root_dir = make_temp_test_dir("trx_test");
  fs::path trx_dir = root_dir / "trx_data";
  std::error_code ec;
  if (!fs::create_directory(trx_dir, ec) && ec) {
    throw std::runtime_error("Failed to create trx data directory: " + ec.message());
  }

  fixture.root_dir = root_dir;
  fixture.path = (root_dir / "small.trx").string();
  fixture.dir_path = trx_dir.string();
  fixture.nb_vertices = 12;
  fixture.nb_streamlines = 4;

  json::object header_obj;
  header_obj["DIMENSIONS"] = json::array{117, 151, 115};
  header_obj["NB_STREAMLINES"] = fixture.nb_streamlines;
  header_obj["NB_VERTICES"] = fixture.nb_vertices;
  header_obj["VOXEL_TO_RASMM"] = json::array{
      json::array{-1.25, 0.0, 0.0, 72.5},
      json::array{0.0, 1.25, 0.0, -109.75},
      json::array{0.0, 0.0, 1.25, -64.5},
      json::array{0.0, 0.0, 0.0, 1.0},
  };
  fixture.expected_header = json(header_obj);

  // Write header.json
  fs::path header_path = trx_dir / "header.json";
  std::ofstream header_out(header_path.string());
  if (!header_out.is_open()) {
    throw std::runtime_error("Failed to write header.json");
  }
  header_out << fixture.expected_header.dump() << std::endl;
  header_out.close();

  // Write positions (float16)
  Matrix<half, Dynamic, Dynamic, RowMajor> positions(fixture.nb_vertices, 3);
  positions.setZero();
  fs::path positions_path = trx_dir / "positions.3.float16";
  trxmmap::write_binary(positions_path.string(), positions);
  struct stat sb;
  if (stat(positions_path.string().c_str(), &sb) != 0) {
    throw std::runtime_error("Failed to stat positions file");
  }
  const size_t expected_positions_bytes = fixture.nb_vertices * 3 * sizeof(half);
  if (static_cast<size_t>(sb.st_size) != expected_positions_bytes) {
    throw std::runtime_error("Positions file size mismatch");
  }

  // Write offsets (uint64) with sentinel (NB_STREAMLINES + 1)
  Matrix<uint64_t, Dynamic, Dynamic, RowMajor> offsets(fixture.nb_streamlines + 1, 1);
  for (int i = 0; i < fixture.nb_streamlines; ++i) {
    offsets(i, 0) = static_cast<uint64_t>(i * (fixture.nb_vertices / fixture.nb_streamlines));
  }
  offsets(fixture.nb_streamlines, 0) = static_cast<uint64_t>(fixture.nb_vertices);

  fs::path offsets_path = trx_dir / "offsets.uint64";
  trxmmap::write_binary(offsets_path.string(), offsets);
  if (stat(offsets_path.string().c_str(), &sb) != 0) {
    throw std::runtime_error("Failed to stat offsets file");
  }
  const size_t expected_offsets_bytes = (fixture.nb_streamlines + 1) * sizeof(uint64_t);
  if (static_cast<size_t>(sb.st_size) != expected_offsets_bytes) {
    throw std::runtime_error("Offsets file size mismatch");
  }

  // Zip the directory into a trx file without compression
  int errorp = 0;
  zip_t *zf = zip_open(fixture.path.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &errorp);
  if (zf == nullptr) {
    throw std::runtime_error("Failed to create trx zip file");
  }
  trxmmap::zip_from_folder(zf, trx_dir.string(), trx_dir.string(), ZIP_CM_STORE);
  if (zip_close(zf) != 0) {
    throw std::runtime_error("Failed to close trx zip file");
  }

  // Validate zip entry sizes
  int zip_err = 0;
  zip_t *verify_zip = trxmmap::open_zip_for_read(fixture.path, zip_err);
  if (verify_zip == nullptr) {
    throw std::runtime_error("Failed to reopen trx zip file");
  }
  zip_stat_t stat_buf;
  if (zip_stat(verify_zip, "offsets.uint64", ZIP_FL_UNCHANGED, &stat_buf) != 0 ||
      static_cast<size_t>(stat_buf.size) != expected_offsets_bytes) {
    zip_close(verify_zip);
    throw std::runtime_error("Zip offsets entry size mismatch");
  }
  if (zip_stat(verify_zip, "positions.3.float16", ZIP_FL_UNCHANGED, &stat_buf) != 0 ||
      static_cast<size_t>(stat_buf.size) != expected_positions_bytes) {
    zip_close(verify_zip);
    throw std::runtime_error("Zip positions entry size mismatch");
  }
  zip_close(verify_zip);

  return fixture;
}

const TestTrxFixture &get_fixture() {
  static TestTrxFixture fixture = create_fixture();
  return fixture;
}
} // namespace

// TODO: Test null filenames. Maybe use MatrixBase instead of ArrayBase
// TODO: try to update test case to use GTest parameterization
// Mirrors trx/tests/test_memmap.py::test__generate_filename_from_data.
TEST(TrxFileMemmap, __generate_filename_from_data) {
  std::string filename = "mean_fa.bit";
  std::string output_fn;

  Matrix<int16_t, 5, 4> arr1;
  std::string exp_1 = "mean_fa.4.int16";

  output_fn = _generate_filename_from_data(arr1, filename);
  EXPECT_STREQ(output_fn.c_str(), exp_1.c_str());
  output_fn.clear();

  Matrix<double, 5, 4> arr2;
  std::string exp_2 = "mean_fa.4.float64";

  output_fn = _generate_filename_from_data(arr2, filename);
  EXPECT_STREQ(output_fn.c_str(), exp_2.c_str());
  output_fn.clear();

  Matrix<double, 5, 1> arr3;
  std::string exp_3 = "mean_fa.float64";

  output_fn = _generate_filename_from_data(arr3, filename);
  EXPECT_STREQ(output_fn.c_str(), exp_3.c_str());
  output_fn.clear();
}

TEST(TrxFileMemmap, detect_positions_dtype_normalizes_slashes) {
  const fs::path root_dir = make_temp_test_dir("trx_norm_slash");
  const fs::path weird_dir = root_dir / "subdir\\nested";
  std::error_code ec;
  fs::create_directories(weird_dir, ec);
  ASSERT_FALSE(ec);

  const fs::path positions_path = weird_dir / "positions.3.float64";
  std::ofstream out(positions_path.string());
  ASSERT_TRUE(out.is_open());
  out.close();

  EXPECT_EQ(trxmmap::detect_positions_dtype(root_dir.string()), "float64");
}

TEST(TrxFileMemmap, detect_positions_scalar_type_directory) {
  auto make_dir_with_positions = [](const std::string &suffix) {
    const fs::path root_dir = make_temp_test_dir("trx_scalar_type");
    const fs::path positions_path = root_dir / ("positions.3." + suffix);
    std::ofstream out(positions_path.string());
    if (!out.is_open()) {
      throw std::runtime_error("Failed to write positions file");
    }
    out.close();
    return root_dir;
  };

  const fs::path float16_dir = make_dir_with_positions("float16");
  const fs::path float32_dir = make_dir_with_positions("float32");
  const fs::path float64_dir = make_dir_with_positions("float64");

  EXPECT_EQ(trxmmap::detect_positions_scalar_type(float16_dir.string(), TrxScalarType::Float64),
            TrxScalarType::Float16);
  EXPECT_EQ(trxmmap::detect_positions_scalar_type(float32_dir.string(), TrxScalarType::Float16),
            TrxScalarType::Float32);
  EXPECT_EQ(trxmmap::detect_positions_scalar_type(float64_dir.string(), TrxScalarType::Float32),
            TrxScalarType::Float64);
}

TEST(TrxFileMemmap, detect_positions_scalar_type_fallback) {
  const fs::path empty_dir = make_temp_test_dir("trx_scalar_empty");
  EXPECT_EQ(trxmmap::detect_positions_scalar_type(empty_dir.string(), TrxScalarType::Float16),
            TrxScalarType::Float16);

  const fs::path invalid_dir = make_temp_test_dir("trx_scalar_invalid");
  const fs::path invalid_positions = invalid_dir / "positions.3.txt";
  std::ofstream out(invalid_positions.string());
  ASSERT_TRUE(out.is_open());
  out.close();

  EXPECT_THROW(trxmmap::detect_positions_scalar_type(invalid_dir.string(), TrxScalarType::Float64),
               std::invalid_argument);
}

TEST(TrxFileMemmap, detect_positions_scalar_type_missing_path) {
  const fs::path missing = fs::path(make_temp_test_dir("trx_scalar_missing")) / "nope";
  EXPECT_THROW(trxmmap::detect_positions_scalar_type(missing.string(), TrxScalarType::Float32), std::runtime_error);
}

TEST(TrxFileMemmap, open_zip_for_read_generic_fallback) {
#if defined(_WIN32) || defined(_WIN64)
  const fs::path root_dir = make_temp_test_dir("trx_zip_generic");
  const fs::path zip_path = root_dir / "sample.trx";

  int errorp = 0;
  zip_t *zf = zip_open(zip_path.string().c_str(), ZIP_CREATE | ZIP_TRUNCATE, &errorp);
  ASSERT_NE(zf, nullptr);
  const char payload[] = "data";
  zip_source_t *source = zip_source_buffer(zf, payload, sizeof(payload) - 1, 0);
  ASSERT_NE(source, nullptr);
  ASSERT_GE(zip_file_add(zf, "dummy.txt", source, ZIP_FL_OVERWRITE), 0);
  ASSERT_EQ(zip_close(zf), 0);

  std::string alt_path = zip_path.string();
  std::replace(alt_path.begin(), alt_path.end(), '/', '\\');
  const std::string generic = fs::path(alt_path).generic_string();
  if (generic == alt_path) {
    GTEST_SKIP() << "Generic string did not change on this platform";
  }

  errorp = 0;
  zip_t *direct = zip_open(alt_path.c_str(), 0, &errorp);
  if (direct != nullptr) {
    zip_close(direct);
    GTEST_SKIP() << "libzip accepts backslash paths; fallback not exercised";
  }

  errorp = 0;
  zip_t *fallback = trxmmap::open_zip_for_read(alt_path, errorp);
  ASSERT_NE(fallback, nullptr);
  zip_close(fallback);
#else
  GTEST_SKIP() << "Generic path fallback is Windows-only";
#endif
}

// Mirrors trx/tests/test_memmap.py::test__split_ext_with_dimensionality.
TEST(TrxFileMemmap, __split_ext_with_dimensionality) {
  std::tuple<std::string, int, std::string> output;
  const std::string fn1 = "mean_fa.float64";
  std::tuple<std::string, int, std::string> exp1("mean_fa", 1, "float64");
  output = _split_ext_with_dimensionality(fn1);
  EXPECT_TRUE(output == exp1);

  const std::string fn2 = "mean_fa.5.int32";
  std::tuple<std::string, int, std::string> exp2("mean_fa", 5, "int32");
  output = _split_ext_with_dimensionality(fn2);
  // std::cout << std::get<0>(output) << " TEST " << std::get<1>(output) << " " <<
  // std::get<2>(output) << std::endl;
  EXPECT_TRUE(output == exp2);

  const std::string fn3 = "mean_fa";
  EXPECT_THROW(
      {
        try {
          output = _split_ext_with_dimensionality(fn3);
        } catch (const std::invalid_argument &e) {
          EXPECT_STREQ("Invalid filename", e.what());
          throw;
        }
      },
      std::invalid_argument);

  const std::string fn4 = "mean_fa.5.4.int32";
  EXPECT_THROW(
      {
        try {
          output = _split_ext_with_dimensionality(fn4);
        } catch (const std::invalid_argument &e) {
          EXPECT_STREQ("Invalid filename", e.what());
          throw;
        }
      },
      std::invalid_argument);

  const std::string fn5 = "mean_fa.fa";
  EXPECT_THROW(
      {
        try {
          output = _split_ext_with_dimensionality(fn5);
        } catch (const std::invalid_argument &e) {
          EXPECT_STREQ("Unsupported file extension", e.what());
          throw;
        }
      },
      std::invalid_argument);
}

// Mirrors trx/tests/test_memmap.py::test__compute_lengths.
TEST(TrxFileMemmap, __compute_lengths) {
  Matrix<uint64_t, 5, 1> offsets{uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{4}};
  Matrix<uint32_t, 4, 1> lengths(trxmmap::_compute_lengths(offsets, 4));
  Matrix<uint32_t, 4, 1> result{uint32_t{1}, uint32_t{1}, uint32_t{1}, uint32_t{1}};

  EXPECT_EQ(lengths, result);

  Matrix<uint64_t, 5, 1> offsets2{uint64_t{0}, uint64_t{1}, uint64_t{1}, uint64_t{3}, uint64_t{4}};
  Matrix<uint32_t, 4, 1> lengths2(trxmmap::_compute_lengths(offsets2, 4));
  Matrix<uint32_t, 4, 1> result2{uint32_t{1}, uint32_t{0}, uint32_t{2}, uint32_t{1}};

  EXPECT_EQ(lengths2, result2);

  Matrix<uint64_t, 4, 1> offsets3{uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{4}};
  Matrix<uint32_t, 3, 1> lengths3(trxmmap::_compute_lengths(offsets3, 4));
  Matrix<uint32_t, 3, 1> result3{uint32_t{1}, uint32_t{1}, uint32_t{2}};

  EXPECT_EQ(lengths3, result3);

  Matrix<uint64_t, 2, 1> offsets4;
  offsets4 << uint64_t{0}, uint64_t{2};
  Matrix<uint32_t, 1, 1> lengths4(trxmmap::_compute_lengths(offsets4, 2));
  Matrix<uint32_t, 1, 1> result4(uint32_t{2});

  EXPECT_EQ(lengths4, result4);

  Matrix<uint64_t, 0, 0> offsets5;
  Matrix<uint32_t, 0, 1> lengths5(trxmmap::_compute_lengths(offsets5, 2));
  EXPECT_EQ(lengths5.size(), 0);

  Matrix<int16_t, 5, 1> offsets6{int16_t(0), int16_t(1), int16_t(2), int16_t(3), int16_t(4)};
  Matrix<uint32_t, 4, 1> lengths6(trxmmap::_compute_lengths(offsets6, 4));
  Matrix<uint32_t, 4, 1> result6{uint32_t{1}, uint32_t{1}, uint32_t{1}, uint32_t{1}};
  EXPECT_EQ(lengths6, result6);

  Matrix<int32_t, 5, 1> offsets7{int32_t(0), int32_t(1), int32_t(1), int32_t(3), int32_t(4)};
  Matrix<uint32_t, 4, 1> lengths7(trxmmap::_compute_lengths(offsets7, 4));
  Matrix<uint32_t, 4, 1> result7{uint32_t{1}, uint32_t{0}, uint32_t{2}, uint32_t{1}};
  EXPECT_EQ(lengths7, result7);
}

// Mirrors trx/tests/test_memmap.py::test__is_dtype_valid.
TEST(TrxFileMemmap, __is_dtype_valid) {
  std::string ext = "bit";
  EXPECT_TRUE(_is_dtype_valid(ext));

  std::string ext2 = "int16";
  EXPECT_TRUE(_is_dtype_valid(ext2));

  std::string ext3 = "float32";
  EXPECT_TRUE(_is_dtype_valid(ext3));

  std::string ext4 = "uint8";
  EXPECT_TRUE(_is_dtype_valid(ext4));

  std::string ext5 = "ushort";
  EXPECT_TRUE(_is_dtype_valid(ext5));

  std::string ext6 = "txt";
  EXPECT_FALSE(_is_dtype_valid(ext6));
}

// asserts C++ dtype alias behavior.
TEST(TrxFileMemmap, __sizeof_dtype_ushort_alias) {
  EXPECT_EQ(trxmmap::_sizeof_dtype("ushort"), sizeof(uint16_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("ushort"), trxmmap::_sizeof_dtype("uint16"));
}

// asserts dtype size mapping and default.
TEST(TrxFileMemmap, __sizeof_dtype_values) {
  EXPECT_EQ(trxmmap::_sizeof_dtype("bit"), 1);
  EXPECT_EQ(trxmmap::_sizeof_dtype("uint8"), sizeof(uint8_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("uint16"), sizeof(uint16_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("uint32"), sizeof(uint32_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("uint64"), sizeof(uint64_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("int8"), sizeof(int8_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("int16"), sizeof(int16_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("int32"), sizeof(int32_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("int64"), sizeof(int64_t));
  EXPECT_EQ(trxmmap::_sizeof_dtype("float32"), sizeof(float));
  EXPECT_EQ(trxmmap::_sizeof_dtype("float64"), sizeof(double));
  EXPECT_EQ(trxmmap::_sizeof_dtype("unknown"), sizeof(uint16_t));
}

// asserts dtype code mapping.
TEST(TrxFileMemmap, __get_dtype_codes) {
  EXPECT_EQ(trxmmap::_get_dtype("b"), "bit");
  EXPECT_EQ(trxmmap::_get_dtype("h"), "uint8");
  EXPECT_EQ(trxmmap::_get_dtype("t"), "uint16");
  EXPECT_EQ(trxmmap::_get_dtype("j"), "uint32");
  EXPECT_EQ(trxmmap::_get_dtype("m"), "uint64");
  EXPECT_EQ(trxmmap::_get_dtype("y"), "uint64");
  EXPECT_EQ(trxmmap::_get_dtype("a"), "int8");
  EXPECT_EQ(trxmmap::_get_dtype("s"), "int16");
  EXPECT_EQ(trxmmap::_get_dtype("i"), "int32");
  EXPECT_EQ(trxmmap::_get_dtype("l"), "int64");
  EXPECT_EQ(trxmmap::_get_dtype("x"), "int64");
  EXPECT_EQ(trxmmap::_get_dtype("f"), "float32");
  EXPECT_EQ(trxmmap::_get_dtype("d"), "float64");
  EXPECT_EQ(trxmmap::_get_dtype("z"), "float16");
  EXPECT_EQ(trxmmap::_get_dtype("foo"), "float16");
}

// Mirrors trx/tests/test_memmap.py::test__dichotomic_search.
TEST(TrxFileMemmap, __dichotomic_search) {
  Matrix<int, 1, 5> m{0, 1, 2, 3, 4};
  int result = trxmmap::_dichotomic_search(m);
  EXPECT_EQ(result, 4);

  Matrix<int, 1, 5> m2{0, 1, 0, 3, 4};
  int result2 = trxmmap::_dichotomic_search(m2);
  EXPECT_EQ(result2, 1);

  Matrix<int, 1, 5> m3{0, 1, 2, 0, 4};
  int result3 = trxmmap::_dichotomic_search(m3);
  EXPECT_EQ(result3, 2);

  Matrix<int, 1, 5> m4{0, 1, 2, 3, 4};
  int result4 = trxmmap::_dichotomic_search(m4, 1, 2);
  EXPECT_EQ(result4, 2);

  Matrix<int, 1, 5> m5{0, 1, 2, 3, 4};
  int result5 = trxmmap::_dichotomic_search(m5, 3, 3);
  EXPECT_EQ(result5, 3);

  Matrix<int, 1, 5> m6{0, 0, 0, 0, 0};
  int result6 = trxmmap::_dichotomic_search(m6, 3, 3);
  EXPECT_EQ(result6, -1);
}

// Mirrors trx/tests/test_memmap.py::test__create_memmap (create path).
TEST(TrxFileMemmap, __create_memmap) {

  fs::path dir = make_temp_test_dir("trx_memmap");
  fs::path path = dir / "offsets.int16";

  std::tuple<int, int> shape = std::make_tuple(3, 4);

  // Test 1: create file and allocate space assert that correct data is filled
  mio::shared_mmap_sink empty_mmap = trxmmap::_create_memmap(path.string(), shape);
  Map<Matrix<half, 3, 4>> expected_m(reinterpret_cast<half *>(empty_mmap.data()));
  Matrix<half, 3, 4> zero_filled{
      {half(0), half(0), half(0), half(0)}, {half(0), half(0), half(0), half(0)}, {half(0), half(0), half(0), half(0)}};

  EXPECT_EQ(expected_m, zero_filled);

  // Test 2: edit data and compare mapped values with new mmap
  for (int i = 0; i < expected_m.size(); i++) {
    expected_m(i) = half(i);
  }

  mio::shared_mmap_sink filled_mmap = trxmmap::_create_memmap(path.string(), shape);
  Map<Matrix<half, 3, 4>> real_m(reinterpret_cast<half *>(filled_mmap.data()), std::get<0>(shape), std::get<1>(shape));

  EXPECT_EQ(expected_m, real_m);

  std::error_code ec;
  fs::remove_all(dir, ec);
}

// Mirrors trx/tests/test_memmap.py::test__create_memmap (non-create path).
TEST(TrxFileMemmap, __create_memmap_empty) {
  fs::path dir = make_temp_test_dir("trx_memmap_empty");
  fs::path path = dir / "empty.float32";

  std::tuple<int, int> shape = std::make_tuple(0, 1);
  mio::shared_mmap_sink empty_mmap = trxmmap::_create_memmap(path.string(), shape);

  struct stat sb;
  ASSERT_EQ(stat(path.string().c_str(), &sb), 0);
  EXPECT_EQ(sb.st_size, 0);
  EXPECT_EQ(empty_mmap.size(), 0u);

  std::error_code ec;
  fs::remove_all(dir, ec);
}

// validates header.json parsing in C++.
TEST(TrxFileMemmap, load_header) {
  const auto &fixture = get_fixture();
  int errorp = 0;
  zip_t *zf = trxmmap::open_zip_for_read(fixture.path, errorp);
  json root = trxmmap::load_header(zf);

  EXPECT_EQ(root, fixture.expected_header);
  EXPECT_EQ(root.dump(), fixture.expected_header.dump());

  zip_close(zf);
}

// TEST(TrxFileMemmap, _load)
// {
// }

// TEST(TrxFileMemmap, _load_zip)
// {
// }

// TEST(TrxFileMemmap, initialize_empty_trx)
// {
// }

// TEST(TrxFileMemmap, _create_trx_from_pointer)
// {
// }

// Mirrors trx/tests/test_memmap.py::test_load (small.trx via zip path).
TEST(TrxFileMemmap, load_zip) {
  const auto &fixture = get_fixture();
  auto trx = trxmmap::load_from_zip<half>(fixture.path);
  EXPECT_GT(trx->streamlines->_data.size(), 0);
}

// Mirrors trx/tests/test_memmap.py::test_load for small.trx and small_compressed.trx.
TEST(TrxFileMemmap, load_zip_test_data) {
  const auto root = get_test_data_root();
  if (root.empty()) {
    GTEST_SKIP() << "TRX_TEST_DATA_DIR not set";
  }
  const auto memmap_dir = resolve_memmap_test_data_dir(root);

  const fs::path small_trx = memmap_dir / "small.trx";
  ASSERT_TRUE(fs::exists(small_trx));
  auto trx_small = trxmmap::load_from_zip<half>(small_trx.string());
  EXPECT_GT(trx_small->streamlines->_data.size(), 0);

  const fs::path small_compressed = memmap_dir / "small_compressed.trx";
  ASSERT_TRUE(fs::exists(small_compressed));
  auto trx_compressed = trxmmap::load_from_zip<half>(small_compressed.string());
  EXPECT_GT(trx_compressed->streamlines->_data.size(), 0);
}

// Mirrors trx/tests/test_memmap.py::test_load (small_fldr.trx via directory path).
TEST(TrxFileMemmap, load_directory) {
  const auto &fixture = get_fixture();
  auto trx = trxmmap::load_from_directory<half>(fixture.dir_path);
  EXPECT_GT(trx->streamlines->_data.size(), 0);
}

// Mirrors trx/tests/test_memmap.py::test_load_directory.
TEST(TrxFileMemmap, load_directory_test_data) {
  const auto root = get_test_data_root();
  if (root.empty()) {
    GTEST_SKIP() << "TRX_TEST_DATA_DIR not set";
  }
  const auto memmap_dir = resolve_memmap_test_data_dir(root);

  const fs::path small_dir = memmap_dir / "small_fldr.trx";
  ASSERT_TRUE(fs::exists(small_dir));
  auto trx = trxmmap::load_from_directory<half>(small_dir.string());
  EXPECT_GT(trx->streamlines->_data.size(), 0);
}

// Mirrors trx/tests/test_memmap.py::test_load with missing path raising.
TEST(TrxFileMemmap, load_missing_trx_throws) {
  const auto root = get_test_data_root();
  if (root.empty()) {
    GTEST_SKIP() << "TRX_TEST_DATA_DIR not set";
  }
  const auto memmap_dir = resolve_memmap_test_data_dir(root);

  const fs::path missing_trx = memmap_dir / "dontexist.trx";
  EXPECT_THROW(trxmmap::load_from_zip<half>(missing_trx.string()), std::runtime_error);
}

// validates C++ TrxFile initialization.
TEST(TrxFileMemmap, TrxFile) {
  auto trx = std::make_unique<TrxFile<half>>();

  // expected header
  json::object expected_obj;
  expected_obj["DIMENSIONS"] = json::array{1, 1, 1};
  expected_obj["NB_STREAMLINES"] = 0;
  expected_obj["NB_VERTICES"] = 0;
  expected_obj["VOXEL_TO_RASMM"] = json::array{
      json::array{1.0, 0.0, 0.0, 0.0},
      json::array{0.0, 1.0, 0.0, 0.0},
      json::array{0.0, 0.0, 1.0, 0.0},
      json::array{0.0, 0.0, 0.0, 1.0},
  };
  json expected(expected_obj);

  EXPECT_EQ(trx->header, expected);

  const auto &fixture = get_fixture();
  int errorp = 0;
  zip_t *zf = trxmmap::open_zip_for_read(fixture.path, errorp);
  json root = trxmmap::load_header(zf);
  auto root_init = std::make_unique<TrxFile<half>>();
  root_init->header = root;
  zip_close(zf);

  // TODO: test for now..

  auto trx_init = std::make_unique<TrxFile<half>>(fixture.nb_vertices, fixture.nb_streamlines, root_init.get());
  json::object init_as_obj;
  init_as_obj["DIMENSIONS"] = json::array{117, 151, 115};
  init_as_obj["NB_STREAMLINES"] = fixture.nb_streamlines;
  init_as_obj["NB_VERTICES"] = fixture.nb_vertices;
  init_as_obj["VOXEL_TO_RASMM"] = json::array{
      json::array{-1.25, 0.0, 0.0, 72.5},
      json::array{0.0, 1.25, 0.0, -109.75},
      json::array{0.0, 0.0, 1.25, -64.5},
      json::array{0.0, 0.0, 0.0, 1.0},
  };
  json init_as(init_as_obj);

  EXPECT_EQ(root_init->header, init_as);
  EXPECT_EQ(trx_init->streamlines->_data.size(), fixture.nb_vertices * 3);
  EXPECT_EQ(trx_init->streamlines->_offsets.size(), fixture.nb_streamlines + 1);
  EXPECT_EQ(trx_init->streamlines->_lengths.size(), fixture.nb_streamlines);
}

// validates C++ deepcopy.
TEST(TrxFileMemmap, deepcopy) {
  const auto &fixture = get_fixture();
  auto trx = trxmmap::load_from_zip<half>(fixture.path);
  auto copy = trx->deepcopy();

  EXPECT_EQ(trx->header, copy->header);
  EXPECT_EQ(trx->streamlines->_data, trx->streamlines->_data);
  EXPECT_EQ(trx->streamlines->_offsets, trx->streamlines->_offsets);
  EXPECT_EQ(trx->streamlines->_lengths, trx->streamlines->_lengths);
}

// Mirrors trx/tests/test_memmap.py::test_resize.
TEST(TrxFileMemmap, resize) {
  const auto &fixture = get_fixture();
  auto trx = trxmmap::load_from_zip<half>(fixture.path);
  trx->resize();
  trx->resize(10);
}
// exercises save paths in C++.
TEST(TrxFileMemmap, save) {
  const auto &fixture = get_fixture();
  auto trx = trxmmap::load_from_zip<half>(fixture.path);
  trxmmap::save(*trx, (std::string) "testsave");
  trxmmap::save(*trx, (std::string) "testsave.trx");

  // trxmmap::TrxFile<half> *saved = trxmmap::load_from_zip<half>("testsave.trx");
  //  EXPECT_EQ(saved->data_per_vertex["color_x.float16"]->_data,
  //  trx->data_per_vertex["color_x.float16"]->_data);
}

int main(int argc, char **argv) { // check_syntax off
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
