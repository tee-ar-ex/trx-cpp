Spatial Queries
===============

trx-cpp can build per-streamline axis-aligned bounding boxes (AABBs) and
use them to extract spatial subsets efficiently. This is useful for
interactive slice-view updates or region-of-interest filtering.

Query by bounding box
---------------------

Pass minimum and maximum corners in RAS+ world coordinates (mm):

.. code-block:: cpp

   #include <trx/trx.h>

   auto trx = trx::load<float>("/path/to/tracks.trx");

   std::array<float, 3> min_corner{-10.0f, -10.0f, -10.0f};
   std::array<float, 3> max_corner{ 10.0f,  10.0f,  10.0f};

   auto subset = trx->query_aabb(min_corner, max_corner);
   subset->save("subset.trx", ZIP_CM_STORE);
   subset->close();

Precompute the AABB cache
--------------------------

When issuing multiple spatial queries on the same file — for example, as a
user scrubs through slices in a viewer — precompute the AABB cache once and
pass it to each query:

.. code-block:: cpp

   auto aabbs = trx->build_streamline_aabbs();

   // Query 1
   auto s1 = trx->query_aabb(min1, max1, &aabbs);

   // Query 2 — reuses the same cached bounding boxes
   auto s2 = trx->query_aabb(min2, max2, &aabbs);

   // Optionally build the AABB cache for the result as well
   auto s3 = trx->query_aabb(min3, max3, &aabbs, /*build_aabbs_for_result=*/true);

AABBs are stored in ``float16`` for memory efficiency. Comparisons are
performed in ``float32`` to avoid precision issues at the boundary.

Subset by streamline IDs
-------------------------

If you have a list of streamline indices from a prior step (clustering,
spatial query, manual selection), create a subset directly:

.. code-block:: cpp

   std::vector<uint32_t> ids{0, 4, 42, 99};
   auto subset = trx->subset_streamlines(ids);

   subset->save("subset_by_id.trx", ZIP_CM_STORE);
   subset->close();
