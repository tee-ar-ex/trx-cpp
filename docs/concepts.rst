Core Concepts
=============

This page explains how TRX-cpp represents tractography data on disk and in
memory.

The TRX format
--------------

A TRX file is a ZIP archive (or on-disk directory) whose layout directly
encodes its data model:

- ``header.json`` — spatial metadata
- ``positions.<dtype>`` — all streamline vertices in a single flat array
- ``offsets.<dtype>`` — prefix-sum index from streamlines into positions
- ``dpv/<name>.<dtype>`` — per-vertex metadata arrays
- ``dps/<name>.<dtype>`` — per-streamline metadata arrays
- ``groups/<name>.uint32`` — named index sets of streamlines
- ``dpg/<group>/<name>.<dtype>`` — per-group metadata arrays

Coordinates are stored in **RAS+ world space** (millimeters), matching the
convention used by MRtrix3 ``.tck`` and NIfTI qform outputs.

Positions array
---------------

All streamline vertices are stored in a single flat matrix of shape
``(NB_VERTICES, 3)``. Keeping all vertices contiguous enables efficient
memory mapping and avoids per-streamline allocations.

In trx-cpp, ``positions`` is backed by a ``mio::shared_mmap_sink`` and
exposed as an ``Eigen::Matrix`` view, giving zero-copy read access after
the file is mapped.

Offsets and the sentinel
------------------------

``offsets`` is a prefix-sum index of length ``NB_STREAMLINES + 1``. Element
*i* is the offset in ``positions`` of the first vertex of streamline *i*.
The final element is a **sentinel** equal to ``NB_VERTICES``, which makes
length computation trivial without special-casing the last streamline:

.. code-block:: cpp

   size_t length_i = offsets[i + 1] - offsets[i];

This design avoids per-streamline allocations and makes slicing the global
positions array fast and uniform.

Data per vertex (DPV)
---------------------

A DPV array stores one value per vertex in ``positions``. It has shape
``(NB_VERTICES, 1)`` for scalar fields or ``(NB_VERTICES, N)`` for
vector-valued fields. Typical uses:

- FA values along the tract
- Per-point RGB colors
- Confidence or weight measures per vertex

DPV arrays live under ``dpv/`` and are memory-mapped in the same way as
``positions``.

Data per streamline (DPS)
-------------------------

A DPS array stores one value per streamline. It has shape
``(NB_STREAMLINES, 1)`` or ``(NB_STREAMLINES, N)``. Typical uses:

- Mean FA or average curvature per tract
- Per-streamline cluster labels
- Tractography algorithm weights

DPS arrays live under ``dps/`` and are mapped into ``MMappedMatrix`` objects.

Groups
------

A group is a named list of streamline indices stored as a ``uint32`` array
under ``groups/``. Groups enable sparse, overlapping labeling: a streamline
can belong to multiple groups, and groups can have different sizes. Typical
uses:

- Bundle labels (``CST_L``, ``CC``, ``SLF_R``, ...)
- Cluster assignments from QuickBundles or similar algorithms
- Connectivity subsets (streamlines connecting two ROIs)

Data per group (DPG)
--------------------

DPG attaches metadata to a group. Each group folder ``dpg/<name>/`` can
contain any number of scalar or vector arrays. Typical uses:

- Mean FA across the bundle
- Per-bundle display color
- Volume or surface-area estimates

In trx-cpp, groups are ``MMappedMatrix<uint32_t>`` objects and DPG fields are
``MMappedMatrix<DT>`` entries, both memory-mapped.

Header
------

``header.json`` stores:

- ``VOXEL_TO_RASMM`` — 4×4 affine mapping voxel indices to RAS+ world
  coordinates (mm)
- ``DIMENSIONS`` — reference image grid dimensions as three ``uint16`` values
- ``NB_STREAMLINES`` — number of streamlines (``uint32``)
- ``NB_VERTICES`` — total number of vertices across all streamlines (``uint64``)

The header is primarily for human readability and downstream compatibility.
The authoritative sizes come from the array dimensions themselves.
