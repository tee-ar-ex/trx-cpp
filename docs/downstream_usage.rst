Downstream Usage
================

Writing MRtrix-style streamlines to TRX
---------------------------------------

MRtrix3 streamlines are created by ``MR::DWI::Tractography::Tracking::Exec``,
which appends points to a per-streamline container as tracking progresses.
That container is ``MR::DWI::Tractography::Tracking::GeneratedTrack``, which is
a ``std::vector<Eigen::Vector3f>`` with some tracking metadata
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
one idea on how to convert a list of MRtrix ``GeneratedTrack`` streamlines into a TRX file.

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


Streaming TRX from MRtrix tckgen
--------------------------------

MRtrix ``tckgen`` writes streamlines as they are generated. To stream into TRX
without buffering all streamlines in memory, use ``trx::TrxStream`` to append
streamlines and finalize once tracking completes. This mirrors how the tck
writer works.

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

   // DSI Studio tract data is stored as std::vector<std::vector<float>>
   // (see tract_model.cpp), with each streamline as an interleaved xyz list.
   // Coordinates are in DSI Studio's internal voxel space; convert to RASMM
   // as needed (e.g., multiply by voxel size and apply transforms).
   std::vector<std::vector<float>> streamlines = /* DSI Studio tract_data */;

   // Optional per-streamline cluster labels or group membership.
   std::vector<uint32_t> cluster_ids = /* same length as streamlines */;

   // Convert to TRX positions/offsets.
   size_t total_vertices = 0;
   for (const auto &sl : streamlines) {
     total_vertices += sl.size() / 3;
   }

   trx::TrxFile<float> trx(static_cast<int>(total_vertices),
                           static_cast<int>(streamlines.size()));
   auto &positions = trx.streamlines->_data;
   auto &offsets = trx.streamlines->_offsets;
   auto &lengths = trx.streamlines->_lengths;

   size_t cursor = 0;
   offsets(0) = 0;
   for (size_t i = 0; i < streamlines.size(); ++i) {
     const auto &sl = streamlines[i];
     const size_t points = sl.size() / 3;
     lengths(i) = static_cast<uint32_t>(points);
     offsets(i + 1) = offsets(i) + points;
     for (size_t p = 0; p < points; ++p, ++cursor) {
       positions(cursor, 0) = sl[p * 3 + 0];
       positions(cursor, 1) = sl[p * 3 + 1];
       positions(cursor, 2) = sl[p * 3 + 2];
     }
   }

   // Map cluster labels to TRX groups (one group per label).
   std::map<uint32_t, std::vector<uint32_t>> clusters;
   for (size_t i = 0; i < cluster_ids.size(); ++i) {
     clusters[cluster_ids[i]].push_back(static_cast<uint32_t>(i));
   }
   for (const auto &kv : clusters) {
     trx.add_group_from_indices("cluster_" + std::to_string(kv.first), kv.second);
   }

   trx.save("out.trx", ZIP_CM_STORE);
   trx.close();

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

   // nibrary uses Streamline = std::vector<Point3D> and Tractogram = std::vector<Streamline>.
   using NIBR::Streamline;
   using NIBR::Tractogram;

   Tractogram nibr_streamlines = /* nibrary tractogram data */;

   // Write nibrary streamlines to TRX.
   size_t total_vertices = 0;
   for (const auto &sl : nibr_streamlines) {
     total_vertices += sl.size();
   }

   trx::TrxFile<float> trx_out(static_cast<int>(total_vertices),
                               static_cast<int>(nibr_streamlines.size()));
   auto &positions = trx_out.streamlines->_data;
   auto &offsets = trx_out.streamlines->_offsets;
   auto &lengths = trx_out.streamlines->_lengths;

   size_t cursor = 0;
   offsets(0) = 0;
   for (size_t i = 0; i < nibr_streamlines.size(); ++i) {
     const auto &sl = nibr_streamlines[i];
     lengths(i) = static_cast<uint32_t>(sl.size());
     offsets(i + 1) = offsets(i) + sl.size();
     for (size_t p = 0; p < sl.size(); ++p, ++cursor) {
       positions(cursor, 0) = sl[p][0];
       positions(cursor, 1) = sl[p][1];
       positions(cursor, 2) = sl[p][2];
     }
   }

   trx_out.save("tracks.trx", ZIP_CM_STORE);
   trx_out.close();

   // Read TRX into nibrary-style streamlines.
   auto trx_in = trx::load_any("tracks.trx");
   const auto pos = trx_in.positions.as_matrix<float>();
   const auto offs = trx_in.offsets.as_matrix<uint64_t>();

   Tractogram out_streamlines;
   out_streamlines.reserve(trx_in.num_streamlines());
   for (size_t i = 0; i < trx_in.num_streamlines(); ++i) {
     const size_t start = static_cast<size_t>(offs(i, 0));
     const size_t end = static_cast<size_t>(offs(i + 1, 0));
     Streamline sl;
     sl.reserve(end - start);
     for (size_t j = start; j < end; ++j) {
       sl.push_back({pos(j, 0), pos(j, 1), pos(j, 2)});
     }
     out_streamlines.push_back(std::move(sl));
   }

   trx_in.close();

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

TRX in ITK-SNAP slice views
----------------------------

ITK-SNAP is a very nice image viewer that has not had the ability to visualize
streamlines. It is a very useful tool to check image alignment, especially if
you are working with ITK/ANTs, as it interprets image headers using ITK.

Streamlines could be added to ITK-SNAP slice views by adding a renderer delegate to
the slice rendering pipeline. DSI Studio and MRview both have the ability to plot
streamlines on slices, but neither use ITK to interpret nifti headers. Someday
when TRX is directly integrated into ITK, ITK-SNAP integration could be used
to check that ``antsApplyTransformsToTRX`` is working correctly.

Where to maybe integrate:

- ``GUI/Qt/Components/SliceViewPanel`` sets up the slice view and installs
  renderer delegates.
- ``GUI/Renderer/GenericSliceRenderer`` and ``SliceRendererDelegate`` define
  the overlay rendering API (lines, polylines, etc.).
- Existing overlays (e.g., ``CrosshairsRenderer`` and
  ``PolygonDrawingRenderer``) show how to draw line-based primitives.

Possible workflow:

1. Load a TRX file via GUI.
2. Create a new renderer delegate (e.g., ``StreamlineTrajectoryRenderer``) that:
   - Filters streamlines that intersect the current slice plane (optionally
     using cached AABBs for speed).
   - Projects 3D points into slice coordinates using
     ``GenericSliceModel::MapImageToSlice`` or
     ``GenericSliceModel::MapSliceToImagePhysical``.
   - Draws each trajectory with ``DrawPolyLine`` in the render context.
3. Register the delegate in ``SliceViewPanel`` so it renders above the image.

Coordinate systems:

- ITK-SNAP uses LPS+ physical coordinates by default.
- TRX stores positions in RAS+ world coordinates, so x/y should be negated when
  moving between TRX and ITK-SNAP physical space.

This design keeps the TRX integration localized to the slice overlay system and
does not require changes to core ITK-SNAP data structures.

TRX in SlicerDMRI
-----------------

SlicerDMRI represents tractography as ``vtkPolyData`` inside a
``vtkMRMLFiberBundleNode``. TRX support is implemented by converting TRX
structures to that polydata representation (and back on save).

High-level mapping:

- TRX ``positions`` + ``offsets`` map to polydata points and polyline cells.
  Each streamline becomes one line cell; point coordinates are stored in RAS+.
- TRX DPV (data-per-vertex) becomes ``PointData`` arrays on the polydata.
- TRX DPS (data-per-streamline) becomes ``CellData`` arrays on the polydata.
- TRX groups are represented as a single label array per streamline
  (``TRX_GroupId``), with a name table stored in ``FieldData`` as
  ``TRX_GroupNames``.

Round-trip metadata convention:

- DPV arrays are stored as ``TRX_DPV_<name>`` in ``PointData``.
- DPS arrays are stored as ``TRX_DPS_<name>`` in ``CellData``.
- The storage node only exports arrays with these prefixes back into TRX, so
  metadata remains recognizable and round-trippable.

How users can visualize and interact with TRX metadata in the Slicer GUI:

- **DPV**: can be used for per-point coloring (e.g., FA along the fiber) by
  selecting the corresponding ``TRX_DPV_*`` array as the scalar to display.
- **DPS**: can be used for per-streamline coloring or thresholding by selecting
  a ``TRX_DPS_*`` array in the fiber bundle display controls.
- **Groups**: color by ``TRX_GroupId`` to show group membership, and use
  thresholding or selection filters to isolate specific group ids. The group
  id-to-name mapping is stored in ``TRX_GroupNames`` for reference.

Users can add their own DPV/DPS arrays in Slicer (via Python, modules, or
filters). To ensure these arrays are written back into TRX, name them with the
``TRX_DPV_`` or ``TRX_DPS_`` prefixes and keep them single-component with the
correct tuple counts (points for DPV, streamlines for DPS).

TrxReader vs TrxFile
--------------------

``TrxFile<DT>`` is the core typed container. It owns the memory-mapped arrays,
exposes streamlines and metadata as Eigen matrices, and provides mutation and
save operations. The template parameter ``DT`` fixes the positions dtype
(``half``, ``float``, or ``double``), which allows zero-copy access and avoids
per-element conversions.

``TrxReader<DT>`` is a small RAII wrapper that loads a TRX file and manages the
backing resources. It ensures the temporary extraction directory (for zipped
TRX) is cleaned up when the reader goes out of scope, and provides safe access
to the underlying ``TrxFile``. This separation keeps ``TrxFile`` focused on the
data model, while ``TrxReader`` handles ownership and lifecycle concerns for
loaded files.

In practice, most downstream users do not need to instantiate ``TrxReader``
directly. The common entry points are convenience functions like
``trx::load_any`` or ``trx::with_trx_reader`` and higher-level wrappers that
return a ready-to-use ``TrxFile``. ``TrxReader`` remains available for advanced
use cases where explicit lifetime control of the backing resources is needed.