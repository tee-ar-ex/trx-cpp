#include <trx/trx.h>

#include <cstdint>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace fs = std::filesystem;

TEST(Uint64Offsets, LoadVertexCountAboveInt32Max) {
  const uint64_t base = 2147483647ULL + 100ULL; // just past INT32_MAX
  const uint64_t nb_vertices = base + 5;

  const fs::path dir = fs::temp_directory_path() / "trx_issue50_mini";
  fs::remove_all(dir);
  fs::create_directories(dir);

  std::ofstream(dir / "header.json")
      << R"({"DIMENSIONS":[1,1,1],"NB_STREAMLINES":2,"NB_VERTICES":)" << nb_vertices
      << R"(,"VOXEL_TO_RASMM":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})";

  // Sparse positions.3.float32: seek to the end and write one byte.
  std::ofstream pos(dir / "positions.3.float32", std::ios::binary);
  pos.seekp(static_cast<std::streamoff>(nb_vertices * 3ULL * sizeof(float) - 1));
  pos.put('\0');
  pos.close();

  const uint64_t offsets[3] = {base, base + 2, nb_vertices};
  std::ofstream(dir / "offsets.uint64", std::ios::binary)
      .write(reinterpret_cast<const char *>(offsets), sizeof(offsets));

  auto trx = trx::AnyTrxFile::load(dir.string()); // threw before the fix
  EXPECT_EQ(trx.num_vertices(), static_cast<size_t>(nb_vertices));
  trx.close();

  fs::remove_all(dir);
}
