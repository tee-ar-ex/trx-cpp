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

Writing MRtrix-style streamlines to TRX
---------------------------------------

MRtrix3 streamlines are handled by ``MR::DWI::Tractography::Tracking::Exec``,
which appends points to a per-streamline container as tracking progresses.
That container is ``MR::DWI::Tractography::Tracking::GeneratedTrack``, which is
simply a ``std::vector<Eigen::Vector3f>`` with a couple of tracking metadata
fields (seed index and status). During tracking, each call that advances the
streamline pushes the current position into this vector, so the resulting
streamline is just an ordered list of 3D points.

In practice:

- Streamline points are stored as ``Eigen::Vector3f`` entries in
  ``GeneratedTrack``.
- The tracking code appends ``method.pos`` into that vector as each step
  completes (seed point first, then subsequent vertices).
- The final output is a list of accepted ``GeneratedTrack`` instances, each
  representing one streamline.

TRX stores streamlines as a single ``positions`` array of shape ``(NB_VERTICES, 3)``
and an ``offsets`` array of length ``(NB_STREAMLINES + 1)`` that provides the
prefix-sum offsets of each streamline in ``positions``. The example below shows
how to convert a list of MRtrix ``GeneratedTrack`` streamlines into a TRX file.

.. code-block:: cpp

   #include <trx/trx.h>
   #include "dwi/tractography/tracking/generated_track.h"

   using MR::DWI::Tractography::Tracking::GeneratedTrack;

   void write_trx_from_mrtrix(const std::vector<GeneratedTrack> &tracks,
                              const std::string &out_path) {
     // Count accepted streamlines and total vertices.
     std::vector<const GeneratedTrack *> accepted;
     accepted.reserve(tracks.size());
     size_t total_vertices = 0;
     for (const auto &tck : tracks) {
       if (tck.get_status() != GeneratedTrack::status_t::ACCEPTED) {
         continue;
       }
       accepted.push_back(&tck);
       total_vertices += tck.size();
     }

     const size_t nb_streamlines = accepted.size();
     const size_t nb_vertices = total_vertices;

     // Allocate a TRX file (float positions) with the desired sizes.
     trx::TrxFile<float> trx(nb_vertices, nb_streamlines);

     auto &positions = trx.streamlines->_data;    // (NB_VERTICES, 3)
     auto &offsets = trx.streamlines->_offsets;   // (NB_STREAMLINES + 1, 1)
     auto &lengths = trx.streamlines->_lengths;   // (NB_STREAMLINES, 1)

     size_t cursor = 0;
     offsets(0) = 0;
     for (size_t i = 0; i < nb_streamlines; ++i) {
       const auto &tck = *accepted[i];
       lengths(i) = static_cast<uint32_t>(tck.size());
       offsets(i + 1) = offsets(i) + tck.size();

       for (size_t j = 0; j < tck.size(); ++j, ++cursor) {
         positions(cursor, 0) = tck[j].x();
         positions(cursor, 1) = tck[j].y();
         positions(cursor, 2) = tck[j].z();
       }
     }

     trx.save(out_path, ZIP_CM_STORE);
     trx.close();
   }

Using TRX in a tckedit-style workflow
-------------------------------------

``tckedit`` works on MRtrix ``Streamline<>`` objects, which are just vectors of
3D points with optional per-streamline metadata (e.g., weight). The editing
pipeline in ``tckedit.cpp`` loads streamlines into ``Streamline<>`` batches,
applies inclusion/exclusion and length rules, and writes the edited set.

To use TRX in the same workflow, the process is:

- Load a TRX file and expand it into ``Streamline<>`` objects using the TRX
  ``positions`` and ``offsets`` arrays.
- Apply editing logic (either by reusing MRtrix editing components or by
  reproducing the same ROI/length rules in your own code).
- Write the edited streamlines back to TRX by rebuilding ``positions`` and
  ``offsets`` (the number of output streamlines can change after edits).

Below is a minimal conversion sketch showing TRX -> ``Streamline<>`` -> TRX.
It omits the actual editing step, which would occur in the middle.

.. code-block:: cpp

   #include <trx/trx.h>
   #include "dwi/tractography/streamline.h"

   using MR::DWI::Tractography::Streamline;

   std::vector<Streamline<float>> trx_to_streamlines(const std::string &trx_path) {
     auto trx = trx::load_any(trx_path);

     const auto positions = trx.positions.as_matrix<float>(); // (NB_VERTICES, 3)
     const auto offsets = trx.offsets.as_matrix<uint64_t>();   // (NB_STREAMLINES + 1, 1)

     std::vector<Streamline<float>> out;
     out.reserve(trx.num_streamlines());

     for (size_t i = 0; i < trx.num_streamlines(); ++i) {
       const size_t start = static_cast<size_t>(offsets(i, 0));
       const size_t end = static_cast<size_t>(offsets(i + 1, 0));
       Streamline<float> tck(end - start);
       for (size_t j = start; j < end; ++j) {
         tck[j - start] = Streamline<float>::point_type(
             positions(j, 0), positions(j, 1), positions(j, 2));
       }
       out.push_back(std::move(tck));
     }

     trx.close();
     return out;
   }

   void streamlines_to_trx(const std::vector<Streamline<float>> &tracks,
                           const std::string &out_path) {
     size_t total_vertices = 0;
     for (const auto &tck : tracks) {
       total_vertices += tck.size();
     }

     trx::TrxFile<float> trx(total_vertices, tracks.size());
     auto &positions = trx.streamlines->_data;
     auto &offsets = trx.streamlines->_offsets;
     auto &lengths = trx.streamlines->_lengths;

     size_t cursor = 0;
     offsets(0) = 0;
     for (size_t i = 0; i < tracks.size(); ++i) {
       const auto &tck = tracks[i];
       lengths(i) = static_cast<uint32_t>(tck.size());
       offsets(i + 1) = offsets(i) + tck.size();
       for (size_t j = 0; j < tck.size(); ++j, ++cursor) {
         positions(cursor, 0) = tck[j].x();
         positions(cursor, 1) = tck[j].y();
         positions(cursor, 2) = tck[j].z();
       }
     }

     trx.save(out_path, ZIP_CM_STORE);
     trx.close();
   }

Streaming TRX from MRtrix tckgen
--------------------------------

MRtrix ``tckgen`` writes streamlines as they are generated. To stream into TRX
without buffering all streamlines in memory, use ``trx::TrxStream`` to append
streamlines and finalize once tracking completes. This mirrors the tck writer
pattern but produces TRX output.

.. code-block:: cpp

   #include <trx/trx.h>
   #include "dwi/tractography/tracking/generated_track.h"

   using MR::DWI::Tractography::Tracking::GeneratedTrack;

   trx::TrxStream trx_stream;

   // Called for each accepted streamline.
   void on_streamline(const GeneratedTrack &tck) {
     std::vector<float> xyz;
     xyz.reserve(tck.size() * 3);
     for (const auto &pt : tck) {
       xyz.push_back(pt[0]);
       xyz.push_back(pt[1]);
       xyz.push_back(pt[2]);
     }
     trx_stream.push_streamline(xyz);
   }

   trx_stream.finalize<float>("tracks.trx", ZIP_CM_STORE);

Using TRX in DSI Studio
-----------------------

DSI Studio stores tractography in ``tract_model.cpp`` as a list of per-tract
point arrays and optional cluster assignments. The TRX format maps cleanly onto
this representation:

- DSI Studio cluster assignments map to TRX ``groups/`` files. Each cluster is a
  group containing the indices of streamlines that belong to that cluster.
- Per-streamline values (e.g., DSI's loaded scalar values) map to TRX DPS
  (``data_per_streamline``) arrays.
- Per-vertex values (e.g., along-tract scalars) map to TRX DPV
  (``data_per_vertex``) arrays.

This means a TRX file can carry the tract geometry, cluster membership, and
both per-streamline and per-vertex metrics in a single archive, and DSI Studio
can round-trip these fields without custom sidecars.

Usage sketch (DSI Studio)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // DSI Studio uses TractModel for import/export.
   // Saving a TRX will populate:
   // - streamlines -> positions/offsets
   // - tract_cluster -> groups/
   // - loaded_values -> dps/dsi_loaded_values

   TractModel tracts(handle->dim, handle->vs, handle->trans_to_mni);
   tracts.add_tracts(streamlines);         // std::vector<std::vector<float>>
   tracts.tract_cluster = cluster_ids;     // optional cluster labels
   tracts.save_tracts_to_file("out.trx");  // writes TRX

Using TRX with nibrary (dmriTrekker)
------------------------------------

nibrary (used by dmriTrekker for tractogram handling) could provide TRX
reading and writing support in its tractography I/O layer. TRX fits this
pipeline well because it exposes the same primitives that nibrary uses
internally: a list of streamlines (each a list of 3D points) plus optional
per-streamline and per-vertex fields.

Coordinate systems:

- TRX ``positions`` are stored in world space (RASMM), which is RAS+ and matches
  the coordinate system used by MRtrix3's ``.tck`` format. nibrary's internal
  streamline points are the same coordinates written out by its TCK writer, so
  those points map directly to TRX ``positions`` when using the same reference
  space.
- TRX header fields ``VOXEL_TO_RASMM`` and ``DIMENSIONS`` should be populated from
  the reference image used by dmriTrekker/nibrary so downstream tools interpret
  coordinates consistently.

Usage sketch (nibrary)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   using namespace NIBR;

   TractogramReader reader("tracks.trx");
   auto batch = reader.getNextStreamlineBatch(1000);

   TRXWriter writer("tracks_copy.trx");
   writer.open();
   writer.writeBatch(batch);
   long streamlines = 0, points = 0;
   writer.close(streamlines, points);



Loading MITK Diffusion streamlines into TRX
-------------------------------------------

MITK Diffusion stores its streamline output in ``StreamlineTrackingFilter`` as a
``BundleType``, which is a ``std::vector`` of ``FiberType`` objects. Each ``FiberType``
is a ``std::deque<itk::Point<float>>``, i.e., an ordered list of 3D points in
physical space. Converting this to TRX follows the same pattern as any other
list-of-points representation: flatten all points into the TRX ``positions``
array and build a prefix-sum ``offsets`` array.

Note on physical space, headers, and affines:

- MITK streamlines are in physical space (millimeters) using ITK's LPS+
  convention by default. TRX ``positions`` are expected to be in RASMM, so you
  should flip the x and y axes when writing TRX (and flip back when reading).


The sketch below shows how to write a ``BundleType`` to TRX and how to reconstruct
it from a TRX file if needed.

.. code-block:: cpp

   #include <trx/trx.h>
   #include <deque>
   #include <vector>
   #include <itkPoint.h>

   using FiberType = std::deque<itk::Point<float>>;
   using BundleType = std::vector<FiberType>;

   void mitk_bundle_to_trx(const BundleType &bundle, const std::string &out_path) {
     size_t total_vertices = 0;
     for (const auto &fiber : bundle) {
       total_vertices += fiber.size();
     }

     trx::TrxFile<float> trx(total_vertices, bundle.size());
     auto &positions = trx.streamlines->_data;
     auto &offsets = trx.streamlines->_offsets;
     auto &lengths = trx.streamlines->_lengths;

     size_t cursor = 0;
     offsets(0) = 0;
     for (size_t i = 0; i < bundle.size(); ++i) {
       const auto &fiber = bundle[i];
       lengths(i) = static_cast<uint32_t>(fiber.size());
       offsets(i + 1) = offsets(i) + fiber.size();
       size_t j = 0;
       for (const auto &pt : fiber) {
         // LPS (MITK/ITK) -> RAS (TRX)
         positions(cursor, 0) = -pt[0];
         positions(cursor, 1) = -pt[1];
         positions(cursor, 2) = pt[2];
         ++cursor;
         ++j;
       }
     }

     trx.save(out_path, ZIP_CM_STORE);
     trx.close();
   }

   BundleType trx_to_mitk_bundle(const std::string &trx_path) {
     auto trx = trx::load_any(trx_path);
     const auto positions = trx.positions.as_matrix<float>(); // (NB_VERTICES, 3)
     const auto offsets = trx.offsets.as_matrix<uint64_t>();   // (NB_STREAMLINES + 1, 1)

     BundleType bundle;
     bundle.reserve(trx.num_streamlines());

     for (size_t i = 0; i < trx.num_streamlines(); ++i) {
       const size_t start = static_cast<size_t>(offsets(i, 0));
       const size_t end = static_cast<size_t>(offsets(i + 1, 0));
       FiberType fiber;
       fiber.resize(end - start);
       for (size_t j = start; j < end; ++j) {
         // RAS (TRX) -> LPS (MITK/ITK)
         fiber[j - start][0] = -positions(j, 0);
         fiber[j - start][1] = -positions(j, 1);
         fiber[j - start][2] = positions(j, 2);
       }
       bundle.push_back(std::move(fiber));
     }

     trx.close();
     return bundle;
   }


