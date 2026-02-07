Overview
========

TRX-cpp provides:

- Read and write TRX archives and directories
- Memory-mapped access for large datasets
- Simple access to streamlines and metadata

The API is header-focused under ``include/trx``, with implementation in ``src/``.

TRX in C++
==========

TRX file format overview
------------------------

TRX is a tractography container for streamlines and auxiliary data. The core
geometry lives in two arrays: a single ``positions`` array containing all
points and an ``offsets`` array that delineates each streamline. Coordinates
are stored in **RAS+ world space (millimeters)**. This matches the coordinate
convention used by MRtrix ``.tck`` and many other tractography tools.
It also avoids the pitfalls of image-based coordinate systems.

TRX can be stored either as:

- A **directory** containing ``header.json`` and data files (positions,
  offsets, dpv/dps/groups/dpg).
- A **zip archive** (``.trx``) with the same internal structure.

Auxiliary data stored alongside streamlines:

- **DPV** (data per vertex): values aligned with each point.
- **DPS** (data per streamline): values aligned with each streamline.
- **Groups**: named sets of streamline indices for labeling or clustering.
- **DPG** (data per group): values aligned with groups (one set per group).

The ``header.json`` includes spatial metadata such as ``VOXEL_TO_RASMM`` and
``DIMENSIONS`` to preserve interpretation of coordinates. See the
`TRX specification <https://github.com/tee-ar-ex/trx-spec>`_ for details.


Positions array
--------------

The ``positions`` array is a single contiguous matrix of shape
``(NB_VERTICES, 3)``. Storing all vertices in one array is cache-friendly and
enables efficient memory mapping. In trx-cpp, ``positions`` is backed by a
``mio::shared_mmap_sink`` and exposed as an ``Eigen::Matrix`` for zero-copy access
when possible.

Offsets and the sentinel value
------------------------------

The ``offsets`` array is a prefix-sum index into ``positions``. Its length is
``NB_STREAMLINES + 1``. The final element is a **sentinel** that equals
``NB_VERTICES`` and makes length computation trivial:

``length_i = offsets[i + 1] - offsets[i]``.

This design avoids per-streamline allocations and supports fast slicing of the
global ``positions`` array. In trx-cpp, offsets are stored as ``uint64`` and
mapped directly into Eigen.

Data per vertex (DPV)
---------------------

DPV stores a value for each vertex in ``positions``. Examples include FA values
along the tract, per-point colors, or confidence measures. DPV arrays have
shape ``(NB_VERTICES, 1)`` or ``(NB_VERTICES, N)`` for multi-component values.

In trx-cpp, DPV fields are stored under ``dpv/`` and are memory-mapped similarly
to ``positions``. This keeps per-point metadata aligned and contiguous, which
is important for large tractograms.

Data per streamline (DPS)
-------------------------

DPS stores a value per streamline. Examples include streamline length, average
FA, or per-tract weights. DPS arrays have shape ``(NB_STREAMLINES, 1)`` or
``(NB_STREAMLINES, N)``.

In trx-cpp, DPS fields live under ``dps/`` and are mapped into ``MMappedMatrix``
objects, enabling efficient access without loading entire arrays into RAM.

Groups and data per group (DPG)
-------------------------------

Groups provide a sparse, overlapping labeling of streamlines. Each group is a
named list of streamline indices, and a streamline can belong to multiple
groups. Examples:

- **Bundle labels** (e.g., ``CST_L``, ``CST_R``, ``CC``)
- **Clusters** from quickbundles or similar algorithms
- **Connectivity subsets** (e.g., streamlines that connect two ROIs)

DPG (data per group) attaches metadata to each group. Examples:

- Mean FA for each bundle
- A per-group color or display weight
- Scalar summaries computed over the group

In the TRX on-disk layout, groups are stored under ``groups/`` as index arrays,
and DPG data is stored under ``dpg/<group_name>/`` as one or more arrays.

In trx-cpp, groups are represented as ``MMappedMatrix<uint32_t>`` objects and
DPG fields are stored as ``MMappedMatrix<DT>`` entries. This keeps group data
as memory-mapped arrays so large group sets can be accessed without copying.
