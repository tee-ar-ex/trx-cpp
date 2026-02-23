Integration Guide
=================

This page provides worked examples for integrating trx-cpp into common
tractography frameworks. Each example shows how to map the framework's
internal streamline representation to TRX and back.

All examples assume that coordinates are already in RAS+ world space
(millimeters). If your framework uses a different coordinate convention,
apply the appropriate affine transform before writing to TRX. A common case
is LPS+ (used by ITK-based tools such as MITK), where you negate the x and y
components to convert to RAS+.

MRtrix3
-------

MRtrix3 tracks are stored as ``GeneratedTrack`` objects
(``std::vector<Eigen::Vector3f>``) produced by the tracking engine.
Coordinates are in RAS+ world space and map directly to TRX ``positions``.

**Bulk conversion** — when all streamlines are available in memory:

.. code-block:: cpp

   #include <trx/trx.h>
   #include "dwi/tractography/tracking/generated_track.h"

   using MR::DWI::Tractography::Tracking::GeneratedTrack;

   void write_trx_from_mrtrix(const std::vector<GeneratedTrack>& tracks,
                               const std::string& out_path) {
     std::vector<const GeneratedTrack*> accepted;
     size_t total_vertices = 0;
     for (const auto& tck : tracks) {
       if (tck.get_status() != GeneratedTrack::status_t::ACCEPTED) continue;
       accepted.push_back(&tck);
       total_vertices += tck.size();
     }

     trx::TrxFile<float> trx(total_vertices, accepted.size());

     auto& positions = trx.streamlines->_data;
     auto& offsets   = trx.streamlines->_offsets;
     auto& lengths   = trx.streamlines->_lengths;

     size_t cursor = 0;
     offsets(0) = 0;
     for (size_t i = 0; i < accepted.size(); ++i) {
       const auto& tck = *accepted[i];
       lengths(i) = static_cast<uint32_t>(tck.size());
       offsets(i + 1) = offsets(i) + tck.size();
       for (const auto& pt : tck) {
         positions(cursor, 0) = pt.x();
         positions(cursor, 1) = pt.y();
         positions(cursor, 2) = pt.z();
         ++cursor;
       }
     }

     trx.save(out_path, ZIP_CM_STORE);
     trx.close();
   }

**Streaming** — appending as each streamline is accepted, without buffering:

.. code-block:: cpp

   #include <trx/trx.h>
   #include "dwi/tractography/tracking/generated_track.h"

   using MR::DWI::Tractography::Tracking::GeneratedTrack;

   trx::TrxStream trx_stream;

   void on_streamline(const GeneratedTrack& tck) {
     std::vector<float> xyz;
     xyz.reserve(tck.size() * 3);
     for (const auto& pt : tck) {
       xyz.push_back(pt[0]);
       xyz.push_back(pt[1]);
       xyz.push_back(pt[2]);
     }
     trx_stream.push_streamline(xyz);
   }

   // Call once after all streamlines are generated:
   trx_stream.finalize<float>("tracks.trx", ZIP_CM_STORE);

DSI Studio
----------

DSI Studio stores tractography in ``tract_model.cpp`` as
``std::vector<std::vector<float>>`` with interleaved XYZ values. Cluster
assignments, per-streamline scalars, and along-tract scalars map cleanly to
TRX groups, DPS, and DPV respectively.

.. code-block:: cpp

   std::vector<std::vector<float>> streamlines = /* DSI Studio tract_data */;
   std::vector<uint32_t> cluster_ids = /* one per streamline */;

   size_t total_vertices = 0;
   for (const auto& sl : streamlines) total_vertices += sl.size() / 3;

   trx::TrxFile<float> trx(total_vertices, streamlines.size());
   auto& positions = trx.streamlines->_data;
   auto& offsets   = trx.streamlines->_offsets;
   auto& lengths   = trx.streamlines->_lengths;

   size_t cursor = 0;
   offsets(0) = 0;
   for (size_t i = 0; i < streamlines.size(); ++i) {
     const auto& sl = streamlines[i];
     const size_t pts = sl.size() / 3;
     lengths(i) = static_cast<uint32_t>(pts);
     offsets(i + 1) = offsets(i) + pts;
     for (size_t p = 0; p < pts; ++p, ++cursor) {
       positions(cursor, 0) = sl[p * 3 + 0];
       positions(cursor, 1) = sl[p * 3 + 1];
       positions(cursor, 2) = sl[p * 3 + 2];
     }
   }

   std::map<uint32_t, std::vector<uint32_t>> clusters;
   for (size_t i = 0; i < cluster_ids.size(); ++i) {
     clusters[cluster_ids[i]].push_back(static_cast<uint32_t>(i));
   }
   for (const auto& [label, indices] : clusters) {
     trx.add_group_from_indices("cluster_" + std::to_string(label), indices);
   }

   trx.save("out.trx", ZIP_CM_STORE);
   trx.close();

nibrary / dmriTrekker
---------------------

nibrary uses ``Streamline = std::vector<Point3D>`` and
``Tractogram = std::vector<Streamline>``. Coordinates are in the same world
space as MRtrix3 ``.tck`` (RAS+) and map directly to TRX ``positions``.

**Write nibrary streamlines to TRX:**

.. code-block:: cpp

   using NIBR::Streamline;
   using NIBR::Tractogram;

   Tractogram nibr = /* nibrary tractogram */;

   size_t total_vertices = 0;
   for (const auto& sl : nibr) total_vertices += sl.size();

   trx::TrxFile<float> trx_out(total_vertices, nibr.size());
   auto& positions = trx_out.streamlines->_data;
   auto& offsets   = trx_out.streamlines->_offsets;
   auto& lengths   = trx_out.streamlines->_lengths;

   size_t cursor = 0;
   offsets(0) = 0;
   for (size_t i = 0; i < nibr.size(); ++i) {
     const auto& sl = nibr[i];
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

**Read TRX into nibrary-style streamlines:**

.. code-block:: cpp

   auto trx_in = trx::load_any("tracks.trx");
   const auto pos  = trx_in.positions.as_matrix<float>();
   const auto offs = trx_in.offsets.as_matrix<uint64_t>();

   Tractogram out;
   out.reserve(trx_in.num_streamlines());
   for (size_t i = 0; i < trx_in.num_streamlines(); ++i) {
     const size_t start = static_cast<size_t>(offs(i, 0));
     const size_t end   = static_cast<size_t>(offs(i + 1, 0));
     Streamline sl;
     sl.reserve(end - start);
     for (size_t j = start; j < end; ++j) {
       sl.push_back({pos(j, 0), pos(j, 1), pos(j, 2)});
     }
     out.push_back(std::move(sl));
   }

   trx_in.close();

MITK Diffusion
--------------

MITK Diffusion stores streamlines as ``BundleType``
(``std::vector<FiberType>``), where ``FiberType`` is
``std::deque<itk::Point<float>>``.

**Coordinate system note:** MITK/ITK uses LPS+ physical coordinates. TRX
expects RAS+. Negate the x and y components when writing to TRX, and negate
them again when reading back.

.. code-block:: cpp

   #include <trx/trx.h>
   #include <deque>
   #include <vector>
   #include <itkPoint.h>

   using FiberType  = std::deque<itk::Point<float>>;
   using BundleType = std::vector<FiberType>;

   void mitk_bundle_to_trx(const BundleType& bundle, const std::string& out_path) {
     size_t total_vertices = 0;
     for (const auto& fiber : bundle) total_vertices += fiber.size();

     trx::TrxFile<float> trx(total_vertices, bundle.size());
     auto& positions = trx.streamlines->_data;
     auto& offsets   = trx.streamlines->_offsets;
     auto& lengths   = trx.streamlines->_lengths;

     size_t cursor = 0;
     offsets(0) = 0;
     for (size_t i = 0; i < bundle.size(); ++i) {
       const auto& fiber = bundle[i];
       lengths(i) = static_cast<uint32_t>(fiber.size());
       offsets(i + 1) = offsets(i) + fiber.size();
       for (const auto& pt : fiber) {
         positions(cursor, 0) = -pt[0]; // LPS -> RAS: negate x
         positions(cursor, 1) = -pt[1]; // LPS -> RAS: negate y
         positions(cursor, 2) =  pt[2];
         ++cursor;
       }
     }

     trx.save(out_path, ZIP_CM_STORE);
     trx.close();
   }

   BundleType trx_to_mitk_bundle(const std::string& trx_path) {
     auto trx = trx::load_any(trx_path);
     const auto pos  = trx.positions.as_matrix<float>();
     const auto offs = trx.offsets.as_matrix<uint64_t>();

     BundleType bundle;
     bundle.reserve(trx.num_streamlines());
     for (size_t i = 0; i < trx.num_streamlines(); ++i) {
       const size_t start = static_cast<size_t>(offs(i, 0));
       const size_t end   = static_cast<size_t>(offs(i + 1, 0));
       FiberType fiber(end - start);
       for (size_t j = start; j < end; ++j) {
         fiber[j - start][0] = -pos(j, 0); // RAS -> LPS
         fiber[j - start][1] = -pos(j, 1);
         fiber[j - start][2] =  pos(j, 2);
       }
       bundle.push_back(std::move(fiber));
     }

     trx.close();
     return bundle;
   }

SlicerDMRI
----------

SlicerDMRI represents tractography as ``vtkPolyData`` inside a
``vtkMRMLFiberBundleNode``. TRX structures map to VTK data arrays as follows:

- TRX ``positions`` + ``offsets`` → polydata points and polyline cells.
  Each streamline becomes one line cell; point coordinates are in RAS+.
- TRX DPV → ``PointData`` arrays named ``TRX_DPV_<name>``.
- TRX DPS → ``CellData`` arrays named ``TRX_DPS_<name>``.
- TRX groups → a per-streamline ``TRX_GroupId`` label array in ``CellData``,
  with a ``TRX_GroupNames`` name table in ``FieldData``.

On save, the storage node exports only arrays with the ``TRX_DPV_`` or
``TRX_DPS_`` prefix back to TRX, ensuring clean round-trips without
extraneous fields.

**Visualization in the Slicer GUI:**

- DPV arrays appear in the fiber bundle display controls for per-point
  coloring (e.g., FA along the fiber).
- DPS arrays support per-streamline coloring or thresholding.
- Groups can be visualized by coloring on ``TRX_GroupId`` and using
  thresholding or selection filters to isolate specific group IDs.

ITK-SNAP
--------

ITK-SNAP uses LPS+ physical coordinates. TRX positions are in RAS+, so
negate the x and y components in both directions when converting.

Streamlines can be added to slice views by implementing a renderer delegate
in the slice rendering pipeline:

- ``GUI/Qt/Components/SliceViewPanel`` installs renderer delegates.
- ``GUI/Renderer/GenericSliceRenderer`` and ``SliceRendererDelegate`` define
  the overlay API (lines, polylines).
- ``CrosshairsRenderer`` and ``PolygonDrawingRenderer`` show how to draw
  line-based primitives.

A streamline renderer delegate would:

1. Filter streamlines intersecting the current slice plane (using cached
   AABBs from :func:`trx::TrxFile::build_streamline_aabbs` for speed).
2. Project 3D RAS+ points to slice coordinates via
   ``GenericSliceModel::MapImageToSlice`` (after negating x/y to convert to
   LPS+).
3. Draw each trajectory with ``DrawPolyLine`` in the render context.
