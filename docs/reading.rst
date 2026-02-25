Reading TRX Files
=================

This page covers the common patterns for loading and inspecting TRX data.
See :doc:`api_layers` for guidance on choosing between ``AnyTrxFile`` and
``TrxFile<DT>``.

Load and inspect
----------------

The simplest entry point is :func:`trx::load_any`, which detects the dtype
from the file and returns an :class:`trx::AnyTrxFile`:

.. code-block:: cpp

   #include <trx/trx.h>

   auto trx = trx::load_any("/path/to/tracks.trx");

   std::cout << "dtype       : " << trx.positions.dtype << "\n";
   std::cout << "streamlines : " << trx.num_streamlines() << "\n";
   std::cout << "vertices    : " << trx.num_vertices() << "\n";

   trx.close();

Access positions and offsets
-----------------------------

Positions and offsets are exposed as :class:`trx::TypedArray` objects. Call
``as_matrix<T>()`` to obtain an ``Eigen::Map`` view without copying data:

.. code-block:: cpp

   auto pos  = trx.positions.as_matrix<float>();     // (NB_VERTICES, 3)
   auto offs = trx.offsets.as_matrix<uint64_t>();    // (NB_STREAMLINES + 1, 1)

   for (size_t i = 0; i < trx.num_streamlines(); ++i) {
     const size_t start = static_cast<size_t>(offs(i, 0));
     const size_t end   = static_cast<size_t>(offs(i + 1, 0));
     // vertices for streamline i: pos.block(start, 0, end - start, 3)
   }

Access DPV and DPS
------------------

Per-vertex (DPV) and per-streamline (DPS) metadata are stored in
``std::map<std::string, TypedArray>`` containers:

.. code-block:: cpp

   // List all DPS fields
   for (const auto& [name, arr] : trx.data_per_streamline) {
     std::cout << "dps/" << name
               << " (" << arr.rows() << " x " << arr.cols() << ")\n";
   }

   // Access a specific DPV field
   auto fa = trx.data_per_vertex.at("fa").as_matrix<float>(); // (NB_VERTICES, 1)

Access groups
-------------

.. code-block:: cpp

   for (const auto& [name, indices] : trx.groups) {
     std::cout << "group " << name << ": "
               << indices.size() << " streamlines\n";
   }

Typed access via TrxFile<DT>
----------------------------

When the dtype is known ahead of time, use :func:`trx::load` for a typed
view. Positions and DPV arrays are exposed as ``Eigen::Matrix<DT, ...>``
directly, avoiding element-wise conversion:

.. code-block:: cpp

   auto reader = trx::load<float>("tracks.trx");
   auto& trx   = *reader;

   // trx.streamlines->_data    is Eigen::Matrix<float, Dynamic, 3>
   // trx.streamlines->_offsets is Eigen::Matrix<uint64_t, Dynamic, 1>

   reader->close();

Iterating streamlines without copying
--------------------------------------

Because TRX positions are memory-mapped, the full positions array is never
read into RAM at once — the OS pages in only the regions you touch. You do
not need a separate "streaming reader" to process a 10 M-streamline file
without exhausting memory.

To iterate over each streamline with zero per-streamline allocation, use
:func:`trx::TrxFile::for_each_streamline`. The callback receives the
streamline index, the start row in ``_data``, and the number of vertices:

.. code-block:: cpp

   auto reader = trx::load<float>("tracks.trx");
   auto& trx   = *reader;

   trx.for_each_streamline([&](size_t idx, uint64_t start, uint64_t length) {
     // Zero-copy block view of this streamline's vertices.
     auto pts = trx.streamlines->_data.block(
         static_cast<Eigen::Index>(start), 0,
         static_cast<Eigen::Index>(length), 3);

     // pts is an Eigen expression — no heap allocation.
     // Example: compute the centroid.
     Eigen::Vector3f centroid = pts.colwise().mean();
   });

   reader->close();

For random access to a single streamline, use
:func:`trx::TrxFile::get_streamline`. This copies the vertices into a
``std::vector`` and is convenient for one-off lookups, but avoid it in
tight loops over large tractograms:

.. code-block:: cpp

   auto pts = trx.get_streamline(42); // std::vector<std::array<float,3>>

Chunk-based iteration (AnyTrxFile)
------------------------------------

:class:`trx::AnyTrxFile` provides ``for_each_positions_chunk``, which
iterates the positions buffer in fixed-size byte chunks. This is useful
for transcoding or checksum passes that process all vertices but do not
need the per-streamline boundary structure:

.. code-block:: cpp

   auto trx = trx::load_any("tracks.trx");

   trx.for_each_positions_chunk(
       4 * 1024 * 1024, // 4 MB chunks
       [](trx::TrxScalarType dtype, const void* data,
          size_t point_offset, size_t point_count) {
         // data points to point_count * 3 values of the given dtype,
         // starting at global vertex index point_offset.
       });

   trx.close();;
