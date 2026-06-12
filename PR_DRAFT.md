# Draft PR for MannLabs/alphadia-search-rs — timsTOF / dia-PASEF ion-mobility support

**Branch:** `feature/timstof-im-axis`  ·  **Base:** `main` @ `eea3f01` (v1.1.2-dev0)
**Status:** Rust search core complete + unit-tested + validated on real dia-PASEF; AlphaDIA-side NG framework integration is prototype (see "Remaining").

## Motivation
The Rust ("NG") backend currently raises `NOT_SUPPORTED_BY_NG` for timsTOF (`alphadia/workflow/base.py`), forcing dia-PASEF users onto the Python extraction backend, which is the >1 h/file bottleneck (on our dog dia-PASEF test file it did not finish one file in **6 h**). The blocker is architectural: the NG data model had **no ion-mobility (scan) axis** — extraction was 2D `[fragment, cycle]`, `has_mobility()` was hard-coded `false`, and the NG mapper hard-coded `scan_start=0, scan_stop=1`. This PR adds the IM axis end-to-end through the Rust search core.

## What this PR does (Rust core — done, tested)
1. **IM data model** (`QuadrupoleObservation`, `AlphaRawView`, `DIAData`): adds a per-event `scan_indices` axis (empty ⇒ mobility-agnostic, unchanged behavior). New `DIAData::from_arrays_im(...)` taking `peak_scan_idx` + `num_scans`; real `has_mobility()`/`num_scans`. Mobility-windowed 1D filler and a 3D `[scan, cycle]` filler.
2. **Builder**: threads the scan index into observations; for IM data exposes the per-cycle `rt_index` as `rt_values` (AlphaDIA `_norm_to_rt` uses `rt_values[0]/[-1]` as RT bounds).
3. **Selection**: computes a per-candidate **ion-mobility apex window** via 3D extraction and stores `scan_start/center/stop` on the `Candidate` (instead of the previous hard-coded flat IM).
4. **Scoring + quantification**: extraction is restricted to the candidate's mobility band (real dia-PASEF selectivity — *not* a mobility-summed shortcut). `DenseXICObservation`/`DenseXICMZObservation` take `scan_start/stop`; trait gains mobility-windowed + 3D methods with mobility-agnostic-preserving defaults.

All additive: the Thermo/Orbitrap path is byte-for-byte unchanged when `scan_indices` is empty.

## Tests
- **227 Rust unit tests pass** (was 223 + 4 new IM tests), incl. a builder→mobility-windowed-extraction integration test and a 3D-sum ↔ 1D-sum invariant.
- Toolchain note for offline CI: `rust-toolchain.toml` pins 1.88.0; we built with `RUSTUP_TOOLCHAIN=stable` (1.94.1) after a `cargo fetch` — the crate compiles clean against 1.94.1 with the committed `Cargo.lock` (no `time`/`zerocopy` bump needed).

## Validation on real dia-PASEF (dog, `…21552.d`, 352,755,250 MS2 events, 36 DIA windows, 812 IM scans)
- Selection finds candidates with **real mobility windows** (e.g. `scan_start/stop=[412,445]`) and positive scores; scoring yields **43 non-NaN features** (`mean_correlation≈0.74` on true hits); quantification yields 15 k precursor rows.
- Against a **proper wrong-window null** (same fragments, precursor m/z shifted to a different quad window): targets rank systematically above the null on every feature (top-500 by an LDA over the 43 features is **78 % targets** vs 50 % chance).

## Benchmark
Rust extraction (select 15.6 M lib precursors + score 7.4 M candidates) ≈ **10 s**; Python backend on the same file **timed out at 6 h**. ≳ ~2000× on extraction (Python never completed, so a precise equal-output ratio is pending the items below).

## Remaining for a production-complete timsTOF path (follow-on PRs)
1. **`timstof_to_ng()` ingestion in `ng_mapper.py`** replacing the prototype Python feeder; the production reader should be a **native chunked-parallel `timsrust` reader** (Sage already uses timsrust) — RT-windowed cycle chunks with overlap + core-ownership, no 52 GB blow-up. (The prototype Python feeder's `convert()` is ~10 min / 52 GB and is the only slow part; the Rust search itself is seconds.)
2. **NG per-batch FDR handoff for timsTOF**: in the full pipeline the Rust core scores millions of balanced target/decoy candidates (verified: `parse_candidates` 4.05 M rows, `decoy_nan=0`, balanced), but the per-batch `validate="one_to_one"` quant↔feature merge + NN-FDR accumulation in `optimization_handler` yields `Extracted 0` — an AlphaDIA-framework integration detail for the NG/timsTOF path that still needs work (no Rust-core defect).
3. **Mobility-error scoring feature**: thread `mobility_library` into `SpecLibFlat` and add a mobility-error feature — the strongest dia-PASEF discriminator, needed for production FDR depth.
4. **Remove the `NOT_SUPPORTED_BY_NG` gate** in `base.py`, gated on a passing concordance + entrapment-FDR test.

## Commits
`391ee89` IM foundation · `3c6cfd3` PR-2 IM axis through storage/builder/extraction · `5b169b4` PR-3 IM-aware selection/scoring/quant · `1b4b826` prototype feeder · `c3668f6` RT-axis fix · validation + WRITEUP in `e166826`/`fe7a1e7`/`4e6086b`.
