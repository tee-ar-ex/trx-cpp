#ifndef TRX_NIFTI_IO_H
#define TRX_NIFTI_IO_H

#include <Eigen/Core>
#include <string>

namespace trx {

/**
 * @brief Read VOXEL_TO_RASMM from a NIfTI header on disk.
 *
 * Prefers qform when present. If qform is missing but sform is present,
 * the sform is orthogonalized to a qform-equivalent matrix.
 */
Eigen::Matrix4f read_nifti_voxel_to_rasmm(const std::string &path);

} // namespace trx

#endif
