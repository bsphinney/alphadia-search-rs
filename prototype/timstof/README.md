# timsTOF / dia-PASEF prototype integration (feature/timstof-im-axis)

Python-side glue that feeds a Bruker `.d` (dia-PASEF) into the IM-aware Rust
backend (`DIAData.from_arrays_im`). The Rust changes (IM scan axis through
storage/builder/extraction/selection/scoring) are in `src/`; this directory
holds the Python plumbing used to drive a real AlphaDIA search through them.

## Files
- `build_converter.py` — `convert(dd, frames_per_cycle)`: reads a non-transposed
  `alphadia.raw_data.bruker.TimsTOFBase` (clean AlphaTims push layout, materialized
  `push_indptr` of length n_frames*n_scans+1) and emits per-MS2-event arrays
  (mz, intensity, scan/IM index, cycle_idx, delta_scan_idx per distinct DIA
  isolation window). NO summing over mobility.
- `tims_feeder.py` — `build_dia_data_ng_im(dd, fpc, DIAData)`: groups events into
  spectra (one per (delta_scan_idx, cycle_idx)) with contiguous peak ranges and
  calls `DIAData.from_arrays_im(...)`. Passes the REAL `dd.cycle` 4D isolation-window
  array (alphadia's `init_spectral_library` reads its m/z values).
- `config_rust_tims.yaml` — single-file rust-backend search config.
- `alphadia_base_py.orig` — pristine copy of the patched `alphadia/workflow/base.py`.

## base.py patch (applied to the installed alphadia for the prototype run)
In `Workflow.load(dia_data_path, ...)`, the `extraction_backend == "rust"` block:
- removed the `NOT_SUPPORTED_BY_NG` raise for `TimsTOFTranspose`;
- for timsTOF, loads the `.d` via `TimsTOFBase(dia_data_path)` (NOT
  `self._dia_data.directory`, which points at a transposed `.hdf` cache) and builds
  the NG object with `build_dia_data_ng_im`;
- non-timsTOF path unchanged (`dia_data_to_ng` + asserts).

This is PROTOTYPE plumbing. For upstream it would move into
`alphadia/workflow/peptidecentric/ng/ng_mapper.py` as a `timstof_to_ng()` and the
gate removal would be gated on a passing concordance test (see WRITEUP.md).
