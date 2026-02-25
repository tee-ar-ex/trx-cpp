API Layers
==========

trx-cpp provides three complementary interfaces. Understanding when to use
each one avoids boilerplate and makes code easier to reason about.

AnyTrxFile — runtime-typed
--------------------------

:class:`trx::AnyTrxFile` is the simplest entry point. It reads the positions
dtype directly from the file and exposes all arrays as :class:`trx::TypedArray`
objects with a ``dtype`` string field. Use this when you only have a file path
and do not need to perform arithmetic on positions at the C++ type level.

.. code-block:: cpp

   auto trx = trx::load_any("tracks.trx");

   std::cout << trx.positions.dtype << "\n";     // e.g. "float16"
   std::cout << trx.num_streamlines() << "\n";

   // Access positions as any floating-point type you choose.
   auto pos = trx.positions.as_matrix<float>();  // (NB_VERTICES, 3)

   trx.close();

TrxFile<DT> — compile-time typed
---------------------------------

:class:`trx::TrxFile` is templated on the positions dtype (``Eigen::half``,
``float``, or ``double``). Positions and DPV arrays are exposed as
``Eigen::Matrix<DT, ...>`` directly — no element-wise conversion. Use this
when the dtype is known, or when you are performing per-vertex arithmetic and
want the compiler to enforce type consistency.

.. code-block:: cpp

   auto reader = trx::load<float>("tracks.trx");
   auto& trx   = *reader;

   // trx.streamlines->_data is Eigen::Matrix<float, Dynamic, 3>
   std::cout << trx.streamlines->_data.rows() << " vertices\n";

   reader->close();

The recommended typed entry point is :func:`trx::with_trx_reader`, which
detects the dtype at runtime and dispatches to the correct instantiation:

.. code-block:: cpp

   trx::with_trx_reader("tracks.trx", [](auto& trx) {
     // trx is TrxFile<DT> for the detected dtype
     std::cout << trx.num_vertices() << "\n";
   });

TrxReader<DT> — RAII lifetime management
----------------------------------------

:class:`trx::TrxReader` is a thin RAII wrapper around :class:`trx::TrxFile`.
When a TRX zip is loaded, trx-cpp extracts it to a temporary directory.
``TrxReader`` owns that directory and removes it when it goes out of scope,
ensuring no temporary files are leaked.

In most cases you do not need to instantiate ``TrxReader`` directly. The
convenience functions ``trx::load_any`` and ``trx::with_trx_reader`` handle
the lifetime automatically. ``TrxReader`` is available for advanced use cases
where explicit control over the backing resource lifetime is required — for
example, when passing a ``TrxFile`` across a function boundary and needing the
temporary directory to outlive the calling scope.

Summary
-------

+--------------------+----------------------------------+-----------------------------+
| Class              | Dtype                            | Best for                    |
+====================+==================================+=============================+
| ``AnyTrxFile``     | Runtime (read from file)         | Inspection, generic tools   |
+--------------------+----------------------------------+-----------------------------+
| ``TrxFile<DT>``    | Compile-time                     | Per-vertex computation      |
+--------------------+----------------------------------+-----------------------------+
| ``TrxReader<DT>``  | Compile-time + RAII cleanup      | Explicit lifetime control   |
+--------------------+----------------------------------+-----------------------------+
