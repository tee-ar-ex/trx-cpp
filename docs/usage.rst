Usage
=====

AnyTrxFile vs TrxFile
---------------------

`AnyTrxFile` is the runtime-typed API. It reads the dtype from the file and
exposes arrays as `TypedArray` with a `dtype` string. This is the simplest
entry point when you only have a TRX path.

`TrxFile<DT>` is the typed API. It is templated on the positions dtype
(`half`, `float`, or `double`) and maps data directly into Eigen matrices of
that type. It provides stronger compile-time guarantees but requires knowing
the dtype at compile time or doing manual dispatch. The recommended typed entry
point is :func:`trx::with_trx_reader`, which performs dtype detection and
dispatches to the matching `TrxReader<DT>`.

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

Convenience accessors
---------------------

`TrxFile` exposes helper methods so you do not need to access the underlying
Eigen buffers directly.

.. code-block:: cpp

   const auto num_vertices = trx.num_vertices();
   const auto num_streamlines = trx.num_streamlines();


Write a TRX file
----------------

.. code-block:: cpp

   auto trx = load_any("tracks.trx");
   auto header_obj = trx.header.object_items();
   header_obj["COMMENT"] = "saved by trx-cpp";
   trx.header = json(header_obj);

   trx.save("tracks_copy.trx", ZIP_CM_STORE);
   trx.close();
