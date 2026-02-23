TRX Format Specification
========================

This page documents the on-disk layout and data model of the TRX
tractography format. TRX-cpp is an implementation of this specification;
the authoritative specification repository is at
https://github.com/tee-ar-ex/trx-spec.

General layout
--------------

A TRX file is either an **uncompressed or compressed ZIP archive**, or a
**plain directory**. In both cases the internal structure is identical: the
file hierarchy describes the data, and filename components encode metadata.

- Each file's **basename** is the name of the metadata field.
- Each file's **extension** is the dtype (e.g., ``float16``, ``uint32``).
- Multi-component arrays encode the number of components between the basename
  and the extension (e.g., ``positions.3.float16`` has 3 components per row).
  Single-component arrays may omit this field for readability.

All arrays use **C-style (row-major) memory layout** and **little-endian
byte order**.

Compression is optional. Use ``ZIP_STORE`` for uncompressed storage; use
``ZIP_DEFLATE`` if compression is desired. Compressed files must be
decompressed before memory-mapping.

Header
------

``header.json`` is a JSON dictionary with the following fields:

+------------------+----------------------------+---------------------------+
| Field            | Type                       | Description               |
+==================+============================+===========================+
| VOXEL_TO_RASMM   | 4×4 array of float         | Affine from voxel to RAS+ |
+------------------+----------------------------+---------------------------+
| DIMENSIONS       | list of 3 uint16           | Reference image grid size |
+------------------+----------------------------+---------------------------+
| NB_STREAMLINES   | uint32                     | Number of streamlines     |
+------------------+----------------------------+---------------------------+
| NB_VERTICES      | uint64                     | Total number of vertices  |
+------------------+----------------------------+---------------------------+

The header is primarily for human readability and downstream compatibility
checks. The authoritative array sizes come from the data arrays themselves.

Arrays
------

**positions** (``positions.float16`` / ``positions.float32`` / ``positions.float64``)
  All streamline vertices as a contiguous C array of shape ``(NB_VERTICES, 3)``.
  Stored in **RAS+ world space (millimeters)**, matching the MRtrix3 ``.tck``
  convention.

**offsets** (``offsets.uint32`` / ``offsets.uint64``)
  Prefix-sum index of length ``NB_STREAMLINES + 1``. Element *i* is the index
  in ``positions`` of the first vertex of streamline *i*. The final element
  is a sentinel equal to ``NB_VERTICES``.

  Streamline length: ``length_i = offsets[i+1] - offsets[i]``.

**dpv — data per vertex** (``dpv/<name>.<dtype>``)
  Shape ``(NB_VERTICES, 1)`` or ``(NB_VERTICES, N)``. Values are aligned
  with ``positions`` row-by-row.

**dps — data per streamline** (``dps/<name>.<dtype>``)
  Shape ``(NB_STREAMLINES, 1)`` or ``(NB_STREAMLINES, N)``. Values are
  aligned with streamlines.

**groups** (``groups/<name>.uint32``)
  Variable-length index arrays. Each file lists the 0-based indices of
  streamlines belonging to the named group. All indices must satisfy
  ``0 ≤ id < NB_STREAMLINES``. Groups are non-exclusive: a streamline may
  appear in multiple groups.

**dpg — data per group** (``dpg/<group>/<name>.<dtype>``)
  Shape ``(1,)`` or ``(N,)``. Each subdirectory corresponds to one group.
  Not all metadata fields need to be present in every group.

Accepted dtypes
---------------

+----------+----------+----------+
| Signed   | Unsigned | Float    |
+==========+==========+==========+
| int8     | uint8    | float16  |
+----------+----------+----------+
| int16    | uint16   | float32  |
+----------+----------+----------+
| int32    | uint32   | float64  |
+----------+----------+----------+
| int64    | uint64   |          |
+----------+----------+----------+

Example structure
-----------------

.. code-block:: text

   OHBM_demo.trx
   ├── dpg/
   │   ├── AF_L/
   │   │   ├── mean_fa.float16
   │   │   ├── shuffle_colors.3.uint8
   │   │   └── volume.uint32
   │   ├── AF_R/  CC/  CST_L/  CST_R/  SLF_L/  SLF_R/
   ├── dpv/
   │   ├── color_x.uint8
   │   ├── color_y.uint8
   │   ├── color_z.uint8
   │   └── fa.float16
   ├── dps/
   │   ├── algo.uint8
   │   ├── algo.json
   │   ├── clusters_QB.uint16
   │   ├── commit_colors.3.uint8
   │   └── commit_weights.float32
   ├── groups/
   │   ├── AF_L.uint32
   │   ├── AF_R.uint32
   │   ├── CC.uint32
   │   ├── CST_L.uint32
   │   ├── CST_R.uint32
   │   ├── SLF_L.uint32
   │   └── SLF_R.uint32
   ├── header.json
   ├── offsets.uint64
   └── positions.3.float16
