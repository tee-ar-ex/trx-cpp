NIfTI Header Support
====================

When trx-cpp is built with ``TRX_ENABLE_NIFTI=ON``, the optional NIfTI I/O
module can read qform/sform affines from ``.nii``, ``.hdr``, or ``.nii.gz``
files and embed them in the TRX ``VOXEL_TO_RASMM`` header field.

This is primarily useful when the TRX file must interoperate with the
``.trk`` (TrackVis) format, which stores coordinates in voxel space and
relies on the NIfTI header for the voxel-to-world transform.

Attach a NIfTI affine to a TRX file
-------------------------------------

.. code-block:: cpp

   #include <trx/nifti_io.h>
   #include <trx/trx.h>

   Eigen::Matrix4f affine = trx::read_nifti_voxel_to_rasmm("reference.nii.gz");

   auto trx = trx::load<float>("tracks.trx");
   trx->set_voxel_to_rasmm(affine);
   trx->save("tracks_with_ref.trx");
   trx->close();

Notes
-----

- The qform is preferred when present. If only the sform is available, it is
  orthogonalized to a qform-equivalent matrix, consistent with ITK's NIfTI
  handling.
- The qform/sform logic is adapted from nibabel's MIT-licensed implementation
  (see ``third_party/nibabel/LICENSE``).
- zlib must be discoverable by CMake to build ``.nii.gz`` decompression
  support.

Enable at build time
---------------------

.. code-block:: bash

   cmake -S . -B build -DTRX_ENABLE_NIFTI=ON
   cmake --build build
