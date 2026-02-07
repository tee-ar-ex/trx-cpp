Usage
=====

AnyTrxFile vs TrxFile
---------------------

``AnyTrxFile`` is the runtime-typed API. It reads the dtype from the file and
exposes arrays as ``TypedArray`` with a ``dtype`` string. This is the simplest
entry point when you only have a TRX path.

``TrxFile<DT>`` is the typed API. It is templated on the positions dtype
(``half``, ``float``, or ``double``) and maps data directly into Eigen matrices of
that type. It provides stronger compile-time guarantees but requires knowing
the dtype at compile time or doing manual dispatch. The recommended typed entry
point is :func:`trx::with_trx_reader`, which performs dtype detection and
dispatches to the matching ``TrxReader<DT>``.

See the API reference for details: :class:`trx::AnyTrxFile` and
:class:`trx::TrxFile`.

Read a TRX zip and inspect data
-------------------------------

.. code-block:: cpp

   #include <trx/trx.h>

   using namespace trx;

   const std::string path = "/path/to/tracks.trx";

   auto trx = load_any(path);

   std::cout << "dtype: " << trx.positions.dtype << "\n";
   std::cout << "Vertices: " << trx.num_vertices() << "\n";
   std::cout << "Streamlines: " << trx.num_streamlines() << "\n";

   trx.close();


Write a TRX file
----------------

.. code-block:: cpp

   auto trx = load_any("tracks.trx");
   auto header_obj = trx.header.object_items();
   header_obj["COMMENT"] = "saved by trx-cpp";
   trx.header = json(header_obj);

   trx.save("tracks_copy.trx", ZIP_CM_STORE);
   trx.close();

Optional NIfTI header support
-----------------------------

If downstream software will have to interact with ``trk`` format data, the NIfTI header
is going to be essential to go back and forth between ``trk`` and ``trx``.

When built with ``TRX_ENABLE_NIFTI=ON``, or by default if you're building the examples,
you can read a NIfTI header (``.nii``, ``.hdr``, optionally ``.nii.gz``) and populate
``VOXEL_TO_RASMM`` in a TRX header. The qform is preferred; if missing, the
sform is orthogonalized to a qform-equivalent matrix (consistent with ITK's handling of nifti).
If using this feature, you will also need zlib available. The qform/sform logic here is
translated from nibabel's MIT-licensed implementation (see ``third_party/nibabel/LICENSE``).

.. code-block:: cpp

   #include <trx/nifti_io.h>
   #include <trx/trx.h>

   Eigen::Matrix4f affine = trx::read_nifti_voxel_to_rasmm("reference.nii.gz");
   auto trx = trx::load<float>("tracks.trx");
   trx->set_voxel_to_rasmm(affine);
   trx->save("tracks_with_ref.trx");
   trx->close();

Build and query AABBs
---------------------

TRX can build per-streamline axis-aligned bounding boxes (AABB) and use them to
extract a subset of streamlines intersecting a rectangular region. AABBs are
stored in ``float16`` for memory efficiency, while comparisons are done in
``float32``.

.. code-block:: cpp

   #include <trx/trx.h>

   auto trx = trx::load<float>("/path/to/tracks.trx");

   // Query an axis-aligned box (min/max corners in RAS+ world coordinates).
   std::array<float, 3> min_corner{-10.0f, -10.0f, -10.0f};
   std::array<float, 3> max_corner{10.0f, 10.0f, 10.0f};

   auto subset = trx->query_aabb(min_corner, max_corner);
   // Or precompute and pass the AABB cache explicitly:
   // auto aabbs = trx->build_streamline_aabbs();
   // auto subset = trx->query_aabb(min_corner, max_corner, &aabbs);
   // Optionally build cache for the result:
   // auto subset = trx->query_aabb(min_corner, max_corner, &aabbs, true);
   subset->save("subset.trx", ZIP_CM_STORE);
   subset->close();

Subset by streamline IDs
------------------------

If you already have a list of streamline indices (for example, from a clustering
step or a spatial query), you can create a new TrxFile directly from those
indices.

.. code-block:: cpp

   #include <trx/trx.h>

   auto trx = trx::load<float>("/path/to/tracks.trx");

   std::vector<uint32_t> ids{0, 4, 42, 99};
   auto subset = trx->subset_streamlines(ids);

   subset->save("subset_by_id.trx", ZIP_CM_STORE);
   subset->close();
