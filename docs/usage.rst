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

Thread-safe streaming pattern
-----------------------------

``TrxStream`` is **not** thread-safe for concurrent writes. A common pattern for
multi-core streamline generation is to use worker threads for generation and a
single writer thread (or the main thread) to append to ``TrxStream``.

.. code-block:: cpp

   #include <trx/trx.h>
   #include <condition_variable>
   #include <mutex>
   #include <queue>
   #include <thread>

   struct Batch {
     std::vector<std::vector<std::array<float, 3>>> streamlines;
   };

   std::mutex mutex;
   std::condition_variable cv;
   std::queue<Batch> queue;
   bool done = false;

   // Worker threads: generate streamlines and push batches.
   auto producer = [&]() {
     Batch batch;
     batch.streamlines.reserve(1000);
     for (int i = 0; i < 1000; ++i) {
       std::vector<std::array<float, 3>> points = {/* ... generate ... */};
       batch.streamlines.push_back(std::move(points));
     }
     {
       std::lock_guard<std::mutex> lock(mutex);
       queue.push(std::move(batch));
     }
     cv.notify_one();
   };

   // Writer thread (single): pop batches and push into TrxStream.
   trx::TrxStream stream("float16");
   auto consumer = [&]() {
     for (;;) {
       std::unique_lock<std::mutex> lock(mutex);
       cv.wait(lock, [&]() { return done || !queue.empty(); });
       if (queue.empty() && done) {
         return;
       }
       Batch batch = std::move(queue.front());
       queue.pop();
       lock.unlock();

       for (const auto &points : batch.streamlines) {
         stream.push_streamline(points);
       }
     }
   };

   std::thread writer(consumer);
   std::thread t1(producer);
   std::thread t2(producer);
   t1.join();
   t2.join();
   {
     std::lock_guard<std::mutex> lock(mutex);
     done = true;
   }
   cv.notify_all();
   writer.join();

   stream.finalize<Eigen::half>("tracks.trx", ZIP_CM_STORE);

Process-based sharding and merge
--------------------------------

For large tractograms it is common to generate streamlines in separate
processes, write shard outputs, and merge them later. ``TrxStream`` provides two
finalization methods for directory output:

- ``finalize_directory()`` — Single-process variant that removes any existing
  directory before writing. Use when you control the entire lifecycle.
  
- ``finalize_directory_persistent()`` — Multiprocess-safe variant that does NOT
  remove existing directories. Use when coordinating parallel writes where a
  parent process may pre-create output directories.

Recommended multiprocess pattern:

1. **Parent** pre-creates shard directories to validate filesystem writability.
2. Each **child process** writes a directory shard using
   ``finalize_directory_persistent()``.
3. After finalization completes, child writes a sentinel file (e.g., ``SHARD_OK``)
   to signal completion.
4. **Parent** waits for all ``SHARD_OK`` markers before merging shards.

This pattern avoids race conditions where the parent checks for directory
existence while children are still writing.

.. code-block:: cpp

   // Parent process: pre-create shard directories
   for (size_t i = 0; i < num_shards; ++i) {
     const std::string shard_path = "shards/shard_" + std::to_string(i);
     std::filesystem::create_directories(shard_path);
   }
   
   // Fork child processes...

.. code-block:: cpp

   // Child process: write to pre-created directory
   trx::TrxStream stream("float16");
   // ... push streamlines, dpv, dps, groups ...
   stream.finalize_directory_persistent("/path/to/shards/shard_0");
   
   // Signal completion to parent
   std::ofstream ok("/path/to/shards/shard_0/SHARD_OK");
   ok << "ok\n";
   ok.close();

.. code-block:: cpp

   // Parent process (after waiting for all SHARD_OK markers)
   // Merge by concatenating positions/DPV/DPS, adjusting offsets/groups.
   // See bench/bench_trx_stream.cpp for a reference merge implementation.

.. note::
   Use ``finalize_directory()`` for single-process writes where you want to
   ensure a clean output state. Use ``finalize_directory_persistent()`` for
   multiprocess workflows to avoid removing directories that may be checked
   for existence by other processes.

MRtrix-style write kernel (single-writer)
-----------------------------------------

MRtrix uses a multi-threaded producer stage and a single-writer kernel to
serialize streamlines to disk. The same pattern works for TRX by letting the
writer own the ``TrxStream`` and accepting batches from the thread queue.

.. code-block:: cpp

   #include <trx/trx.h>
   #include <vector>
   #include <array>

   struct TrxWriteKernel {
     explicit TrxWriteKernel(const std::string &path)
         : stream("float16"), out_path(path) {}

     void operator()(const std::vector<std::vector<std::array<float, 3>>> &batch) {
       for (const auto &points : batch) {
         stream.push_streamline(points);
       }
     }

     void finalize() {
       stream.finalize<Eigen::half>(out_path, ZIP_CM_STORE);
     }

   private:
     trx::TrxStream stream;
     std::string out_path;
   };

This kernel can be used as the final stage of a producer pipeline. The key rule
is: **only the writer thread touches ``TrxStream``**, while worker threads only
generate streamlines.

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
