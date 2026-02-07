#ifndef TRX_NIFTI_IO_H
#define TRX_NIFTI_IO_H

#include <Eigen/Core>
#include <string>

namespace trx {

/**
 * @brief Read VOXEL_TO_RASMM from a NIfTI header on disk.
 *
 * Implementation notes:
 * - The qform/sform handling follows a direct translation of nibabel's
 *   NIfTI header logic (see nibabel/nifti1.py). The translation is adapted
 *   to C++ and Eigen, and avoids depending on nibabel at runtime.
 * - Prefers qform when present. If qform is missing but sform is present,
 *   the sform is orthogonalized to a qform-equivalent matrix.
 *
 * Licensing:
 * - nibabel is MIT-licensed; see third_party/nibabel/LICENSE for details.
 */
Eigen::Matrix4f read_nifti_voxel_to_rasmm(const std::string &path);

} // namespace trx

#endif
