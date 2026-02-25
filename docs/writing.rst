Writing TRX Files
=================

This page covers creating TRX files from scratch and saving loaded files.
For append-only streaming writes when the total count is not known ahead of
time, see :doc:`streaming`.

Create and save a TRX file
--------------------------

Allocate a :class:`trx::TrxFile` with the desired number of vertices and
streamlines, fill the positions and offsets arrays, then call ``save``:

.. code-block:: cpp

   #include <trx/trx.h>

   const size_t nb_vertices    = 500;
   const size_t nb_streamlines = 10;

   trx::TrxFile<float> trx(nb_vertices, nb_streamlines);

   auto& positions = trx.streamlines->_data;    // (NB_VERTICES, 3)
   auto& offsets   = trx.streamlines->_offsets; // (NB_STREAMLINES + 1, 1)
   auto& lengths   = trx.streamlines->_lengths; // (NB_STREAMLINES, 1)

   size_t cursor = 0;
   offsets(0) = 0;
   for (size_t i = 0; i < nb_streamlines; ++i) {
     const size_t len = 50; // 50 vertices per streamline in this example
     lengths(i) = static_cast<uint32_t>(len);
     offsets(i + 1) = offsets(i) + len;
     for (size_t j = 0; j < len; ++j, ++cursor) {
       positions(cursor, 0) = /* x */;
       positions(cursor, 1) = /* y */;
       positions(cursor, 2) = /* z */;
     }
   }

   trx.save("tracks.trx", ZIP_CM_STORE);
   trx.close();

Pass ``ZIP_CM_DEFLATE`` instead of ``ZIP_CM_STORE`` to enable compression.
Compression reduces file size at the cost of slower read/write throughput;
``ZIP_CM_STORE`` is preferred for large files accessed over fast storage.

Modify and re-save a loaded file
---------------------------------

.. code-block:: cpp

   auto trx = trx::load_any("tracks.trx");

   // Add or update a header field.
   auto header_obj = trx.header.object_items();
   header_obj["COMMENT"] = "processed by my_tool";
   trx.header = json11::Json(header_obj);

   trx.save("tracks_annotated.trx", ZIP_CM_STORE);
   trx.close();

Save as a directory
-------------------

Pass a directory path (without a ``.trx`` extension) to write an unzipped
TRX directory instead of a zip archive. Directory output avoids ZIP overhead
and is faster for large files on spinning disks:

.. code-block:: cpp

   trx.save("/path/to/output_dir");
