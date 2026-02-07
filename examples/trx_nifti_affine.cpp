#include <trx/nifti_io.h>
#include <trx/trx.h>

#include <iostream>

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: trx-update-affine <input.trx> <reference.nii[.gz]> <output.trx>\n";
    return 1;
  }

  const std::string input_trx = argv[1];
  const std::string nifti_path = argv[2];
  const std::string output_trx = argv[3];

  Eigen::Matrix4f affine = trx::read_nifti_voxel_to_rasmm(nifti_path);

  return trx::with_trx_reader(input_trx, [&](auto &reader, trx::TrxScalarType) {
    auto &trx_file = *reader;
    trx_file.set_voxel_to_rasmm(affine);
    trx_file.save(output_trx);
    trx_file.close();
    return 0;
  });
}
