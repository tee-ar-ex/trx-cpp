Streaming Writes
================

:class:`trx::TrxStream` is an append-only writer for cases where the total
streamline count is not known ahead of time. It writes to temporary files and
finalizes to a standard TRX archive or directory when complete.

.. note::

   ``TrxStream`` is **not** thread-safe for concurrent writes. Use a single
   writer thread (or the main thread) to append to the stream, while other
   threads generate streamlines and deliver them via a queue.

Single-threaded streaming
--------------------------

.. code-block:: cpp

   #include <trx/trx.h>

   trx::TrxStream stream("float16");

   for (/* each generated streamline */) {
     std::vector<std::array<float, 3>> points = /* ... */;
     stream.push_streamline(points);
   }

   stream.finalize<Eigen::half>("tracks.trx", ZIP_CM_STORE);

Multi-threaded producer / single writer
----------------------------------------

Worker threads generate streamlines and push batches into a queue. A
dedicated writer thread owns the ``TrxStream`` and consumes from the queue.

.. code-block:: cpp

   #include <trx/trx.h>
   #include <condition_variable>
   #include <mutex>
   #include <queue>
   #include <thread>

   struct Batch {
     std::vector<std::vector<std::array<float, 3>>> streamlines;
   };

   std::mutex mtx;
   std::condition_variable cv;
   std::queue<Batch> q;
   bool done = false;

   // Producer: generates streamlines, pushes batches into the queue.
   auto producer = [&]() {
     Batch batch;
     batch.streamlines.reserve(1000);
     for (int i = 0; i < 1000; ++i) {
       batch.streamlines.push_back(/* generate points */);
     }
     {
       std::lock_guard<std::mutex> lock(mtx);
       q.push(std::move(batch));
     }
     cv.notify_one();
   };

   // Writer: owns TrxStream, appends batches from the queue.
   trx::TrxStream stream("float16");
   auto writer = [&]() {
     for (;;) {
       std::unique_lock<std::mutex> lock(mtx);
       cv.wait(lock, [&] { return done || !q.empty(); });
       if (q.empty() && done) return;
       Batch batch = std::move(q.front());
       q.pop();
       lock.unlock();
       for (const auto& pts : batch.streamlines) {
         stream.push_streamline(pts);
       }
     }
   };

   std::thread writer_thread(writer);
   std::thread t1(producer), t2(producer);
   t1.join(); t2.join();
   { std::lock_guard<std::mutex> lock(mtx); done = true; }
   cv.notify_all();
   writer_thread.join();

   stream.finalize<Eigen::half>("tracks.trx", ZIP_CM_STORE);

MRtrix-style write kernel
--------------------------

MRtrix3 uses a multi-threaded producer stage and a single-writer kernel to
serialize output. The same pattern works with TRX by encapsulating the
``TrxStream`` inside a kernel object:

.. code-block:: cpp

   struct TrxWriteKernel {
     explicit TrxWriteKernel(const std::string& path)
         : stream("float16"), out_path(path) {}

     void operator()(const std::vector<std::vector<std::array<float, 3>>>& batch) {
       for (const auto& pts : batch) {
         stream.push_streamline(pts);
       }
     }

     void finalize() {
       stream.finalize<Eigen::half>(out_path, ZIP_CM_STORE);
     }

   private:
     trx::TrxStream stream;
     std::string out_path;
   };

The key rule is: **only the writer thread touches ``TrxStream``**, while
worker threads only generate streamlines.

Process-based sharding
-----------------------

For very large tractograms generated in parallel processes, each process can
write to a shard directory and a parent process merges the shards afterward.

``TrxStream`` provides two finalization methods for directory output:

- ``finalize_directory()`` — removes any existing directory before writing.
  Safe for single-process workflows where you control the full lifecycle.
- ``finalize_directory_persistent()`` — does **not** remove existing
  directories. Required when a parent process pre-creates the output
  directory.

Recommended multiprocess pattern:

1. **Parent** pre-creates shard directories.
2. Each **child** calls ``finalize_directory_persistent()`` after appending
   all streamlines.
3. Child writes a sentinel file (e.g., ``SHARD_OK``) to signal completion.
4. **Parent** waits for all sentinels, then merges shards.

.. code-block:: cpp

   // Parent: pre-create shard directories
   for (size_t i = 0; i < num_shards; ++i) {
     std::filesystem::create_directories("shards/shard_" + std::to_string(i));
   }

   // Child: write shard and signal completion
   trx::TrxStream stream("float16");
   // ... push_streamline calls ...
   stream.finalize_directory_persistent("/path/to/shards/shard_0");
   std::ofstream ok("/path/to/shards/shard_0/SHARD_OK");
   ok << "ok\n";

   // Parent: merge shards after all SHARD_OK files are present.
   // See bench/bench_trx_stream.cpp for a reference merge implementation.

.. note::
   Use ``finalize_directory()`` for single-process writes where you want a
   clean output state. Use ``finalize_directory_persistent()`` for
   multiprocess workflows to avoid removing directories that may be checked
   for existence by other processes.
