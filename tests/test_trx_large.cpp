#include <filesystem>
#include <gtest/gtest.h>
#include <trx/trx.h>

#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trx;
namespace fs = std::filesystem;

namespace {
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

void write_header_file(const fs::path &dir, const json &header) {
  std::ofstream out((dir / "header.json").string());
  if (!out.is_open()) {
    throw std::runtime_error("Failed to write header.json");
  }
  out << header.dump() << '\n';
}

}

TEST(Uint64Offsets, LoadVertexCountAboveInt32Max) {
  const uint64_t base = 2147483647ULL + 100ULL; // just past INT32_MAX
  const uint64_t nb_vertices = base + 5;

  const fs::path dir = make_temp_test_dir("trx_issue50");

  json header = json::object{
      {"DIMENSIONS", json::array{1, 1, 1}},
      {"NB_STREAMLINES", 2.0},
      {"NB_VERTICES", static_cast<double>(nb_vertices)},
      {"VOXEL_TO_RASMM", json::array{json::array{1, 0, 0, 0}, json::array{0, 1, 0, 0},
                                     json::array{0, 0, 1, 0}, json::array{0, 0, 0, 1}}},
  };
  write_header_file(dir, header);

  std::ofstream pos((dir / "positions.3.float32").string(), std::ios::binary);
  pos.seekp(static_cast<std::streamoff>(nb_vertices * 3ULL * sizeof(float) - 1));
  pos.put('\0');
  pos.close();

  const uint64_t offsets[3] = {base, base + 2, nb_vertices};
  std::ofstream((dir / "offsets.uint64").string(), std::ios::binary)
      .write(reinterpret_cast<const char *>(offsets), sizeof(offsets));

  auto trx = AnyTrxFile::load(dir.string()); // threw before the fix
  EXPECT_EQ(trx.num_vertices(), static_cast<size_t>(nb_vertices));
  trx.close();

  std::error_code ec;
  fs::remove_all(dir, ec);
}
