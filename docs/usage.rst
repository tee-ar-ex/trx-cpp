Usage
=====

Read a TRX zip and inspect data
-------------------------------

.. code-block:: cpp

   #include <trx/trx.h>

   using namespace trxmmap;

   auto trx = load_from_zip<half>("tracks.trx");

   const auto num_vertices = trx->num_vertices();
   const auto num_streamlines = trx->num_streamlines();

   std::cout << "Vertices: " << num_vertices << "\n";
   std::cout << "Streamlines: " << num_streamlines << "\n";

   trx->close();

Convenience accessors
---------------------

`TrxFile` exposes helper methods so you do not need to access the underlying
Eigen buffers directly.

.. code-block:: cpp

   const auto num_vertices = trx->num_vertices();
   const auto num_streamlines = trx->num_streamlines();

Read from an on-disk TRX directory
----------------------------------

.. code-block:: cpp

   auto trx = load_from_directory<float>("/path/to/trx_dir");
   std::cout << "Header JSON:\n" << trx->header.dump() << "\n";
   trx->close();

Write a TRX file
----------------

.. code-block:: cpp

   auto trx = load_from_zip<half>("tracks.trx");
   auto header_obj = trx->header.object_items();
   header_obj["COMMENT"] = "saved by trx-cpp";
   trx->header = json(header_obj);

   save(*trx, "tracks_copy.trx", ZIP_CM_STORE);
   trx->close();
