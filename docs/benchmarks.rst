Benchmarks
==========

This page documents the benchmarking suite and how to interpret the results.
The benchmarks are designed for realistic tractography workloads (HPC scale),
not for CI. They focus on file size, throughput, and interactive spatial queries.

Data model
----------

All benchmarks synthesize smooth, slightly curved streamlines in a realistic
field of view:

- **Lengths:** random between 20 and 500 mm (profiles skew short/medium/long)
- **Field of view:** x = [-70, 70], y = [-108, 79], z = [-60, 75] (mm, RAS+)
- **Streamline counts:** 100k, 500k, 1M, 5M, 10M
- **Groups:** none, 80 bundle groups, or 4950 connectome groups (100 regions)
- **DPV/DPS:** either present (1 value) or absent

Positions are stored as float16 to highlight storage efficiency.

TRX size vs streamline count
----------------------------

This benchmark writes TRX files with float16 positions and measures the final
on-disk size for different streamline counts. It compares short/medium/long
length profiles, DPV/DPS presence, and zip compression (store vs deflate).

.. figure:: _static/benchmarks/trx_size_vs_streamlines.png
   :alt: TRX file size vs streamlines
   :align: center

   File size (MB) as a function of streamline count.

Translate + stream write throughput
-----------------------------------

This benchmark loads a TRX file, iterates through every streamline, translates
each point by +1 mm in x/y/z, and streams the result into a new TRX file. It
reports total wall time and max RSS so researchers can understand throughput
and memory pressure on both clusters and laptops.

.. figure:: _static/benchmarks/trx_translate_write_time.png
   :alt: Translate + stream write time
   :align: center

   End-to-end time for translating and rewriting streamlines.

.. figure:: _static/benchmarks/trx_translate_write_rss.png
   :alt: Translate + stream write RSS
   :align: center

   Max RSS during translate + stream write.

Spatial slab query latency
--------------------------

This benchmark precomputes per-streamline AABBs and then issues 100 spatial
queries using 5 mm slabs that sweep through the tractogram volume. Each slab
query mimics a GUI slice update and records its timing so distributions can be
visualized.

.. figure:: _static/benchmarks/trx_query_slab_timings.png
   :alt: Slab query timings
   :align: center

   Distribution of per-slab query latency.

Performance characteristics
---------------------------

Benchmark results vary significantly based on storage performance:

**SSD (solid-state drives):**
- **CPU-bound**: Disk writes complete faster than streamline generation
- High CPU utilization (~100%)
- Results reflect pure computational throughput

**HDD (spinning disks):**
- **I/O-bound**: Disk writes are the bottleneck
- Low CPU utilization (~5-10%)
- Results reflect realistic workstation performance with storage latency

Both scenarios are valuable. SSD results show the library's maximum throughput,
while HDD results show real-world performance on cost-effective storage. On
Linux, monitor I/O wait time with ``iostat -x 1`` to identify the bottleneck.

For spinning disks or network filesystems, you may want to increase buffer sizes
to amortize I/O latency. Set ``TRX_BENCH_BUFFER_MULTIPLIER`` to use larger
buffers (e.g., ``TRX_BENCH_BUFFER_MULTIPLIER=4`` uses 4× the default buffer
sizes).

Running the benchmarks
----------------------

Build and run the benchmarks, then plot results with matplotlib:

.. code-block:: bash

   cmake -S . -B build  -DTRX_BUILD_BENCHMARKS=ON
   cmake --build build --target bench_trx_stream

   # Run benchmarks (this can be long for large datasets).
   ./build/bench/bench_trx_stream \
     --benchmark_out=bench/results.json \
     --benchmark_out_format=json
   
   # For slower storage (HDD, NFS), use larger buffers:
   TRX_BENCH_BUFFER_MULTIPLIER=4 \
     ./build/bench/bench_trx_stream \
     --benchmark_out=bench/results_hdd.json \
     --benchmark_out_format=json

   # Capture per-slab timings for query distributions.
   TRX_QUERY_TIMINGS_PATH=bench/query_timings.jsonl \
     ./build/bench/bench_trx_stream \
     --benchmark_filter=BM_TrxQueryAabb_Slabs \
     --benchmark_out=bench/results.json \
     --benchmark_out_format=json

   # Optional: record RSS samples for file-size runs.
   TRX_RSS_SAMPLES_PATH=bench/rss_samples.jsonl \
     TRX_RSS_SAMPLE_EVERY=50000 \
     TRX_RSS_SAMPLE_MS=500 \
     ./build/bench/bench_trx_stream \
     --benchmark_filter=BM_TrxFileSize_Float16 \
     --benchmark_out=bench/results.json \
     --benchmark_out_format=json

   # Generate plots into docs/_static/benchmarks.
   python bench/plot_bench.py bench/results.json \
     --query-json bench/query_timings.jsonl \
     --out-dir docs/_static/benchmarks

The query plot defaults to the "no groups, no DPV/DPS" case. Use
``--group-case``, ``--dpv``, and ``--dps`` in ``plot_bench.py`` to select other
scenarios.

Environment variables
---------------------

The benchmark suite supports several environment variables for customization:

**Multiprocessing:**

- ``TRX_BENCH_PROCESSES`` (default: 1): Number of processes for parallel shard
  generation. Recommended: number of physical cores.
- ``TRX_BENCH_MP_MIN_STREAMLINES`` (default: 1000000): Minimum streamline count
  to enable multiprocessing. Below this threshold, single-process mode is used.
- ``TRX_BENCH_KEEP_SHARDS`` (default: 0): Set to 1 to preserve shard directories
  after merging for debugging.
- ``TRX_BENCH_SHARD_WAIT_MS`` (default: 10000): Timeout in milliseconds for
  waiting for shard completion markers.

**Buffering (for slow storage):**

- ``TRX_BENCH_BUFFER_MULTIPLIER`` (default: 1): Scales position and metadata
  buffer sizes. Use larger values (2-8) for spinning disks or network
  filesystems to reduce I/O latency. Example: multiplier=4 uses 64 MB → 256 MB
  for small datasets, 256 MB → 1 GB for 1M streamlines, 2 GB → 8 GB for 5M+
  streamlines.

**Performance tuning:**

- ``TRX_BENCH_THREADS`` (default: hardware_concurrency): Worker threads for
  streamline generation within each process.
- ``TRX_BENCH_BATCH`` (default: 1000): Streamlines per batch in the producer-
  consumer queue.
- ``TRX_BENCH_QUEUE_MAX`` (default: 8): Maximum batches in flight between
  producers and consumer.

**Dataset control:**

- ``TRX_BENCH_ONLY_STREAMLINES`` (default: 0): If nonzero, benchmark only this
  streamline count instead of the full range.
- ``TRX_BENCH_MAX_STREAMLINES`` (default: 10000000): Maximum streamline count
  to benchmark. Use smaller values for faster iteration.
- ``TRX_BENCH_SKIP_ZIP_AT`` (default: 5000000): Skip zip compression for
  streamline counts at or above this threshold.

**Logging and diagnostics:**

- ``TRX_BENCH_LOG`` (default: 0): Enable benchmark progress logging to stderr.
- ``TRX_BENCH_CHILD_LOG`` (default: 0): Enable logging from child processes in
  multiprocess mode.
- ``TRX_BENCH_LOG_PROGRESS_EVERY`` (default: 0): Log progress every N
  streamlines.

When running with multiprocessing, the benchmark uses
``finalize_directory_persistent()`` to write shard outputs without removing
pre-created directories, avoiding race conditions in the parallel workflow.
