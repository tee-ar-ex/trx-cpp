Introduction
============

TRX-cpp is a C++17 library for reading, writing, and memory-mapping the
`TRX tractography format <https://github.com/tee-ar-ex/trx-spec>`_. TRX is
a ZIP-based container for fiber tract geometry and associated metadata,
designed for large-scale diffusion MRI tractography.

Features
--------

**Zero-copy memory mapping**
  Streamline positions, per-vertex data (DPV), and per-streamline data (DPS)
  are exposed as ``Eigen::Map`` views directly over memory-mapped files.
  Accessing a 10 M-streamline dataset does not require loading the full array
  into RAM.

**Streaming writes**
  :class:`trx::TrxStream` appends streamlines incrementally and finalizes to
  a TRX archive or directory once tracking is complete. Suitable for
  tractography pipelines where the total count is unknown at the start.

**Spatial queries**
  Build per-streamline axis-aligned bounding boxes (AABBs) and extract spatial
  subsets in sub-millisecond time per query. Designed for interactive
  slice-view workflows that need to filter streamlines as the user moves
  through a volume.

**Typed and type-erased APIs**
  :class:`trx::TrxFile` is templated on the positions dtype for compile-time
  type safety. :class:`trx::AnyTrxFile` reads the dtype from disk and
  dispatches at runtime — useful when the file format is not known in advance.

**ZIP and directory storage**
  Read and write ``.trx`` zip archives and plain on-disk directories with the
  same API. Directory storage is convenient for in-place access; zip storage is
  convenient for distribution and transfer.

**Optional NIfTI support**
  Read qform/sform affines from ``.nii``, ``.hdr``, or ``.nii.gz`` and embed
  them in the TRX header for consistent coordinate interpretation downstream.

Where to go next
----------------

- :doc:`quick_start` — install the library and write a first program
- :doc:`building` — full dependency and build options reference
- :doc:`concepts` — how TRX-cpp represents streamlines and metadata internally
- :doc:`api_layers` — choosing between ``AnyTrxFile``, ``TrxFile<DT>``, and
  ``TrxReader<DT>``
