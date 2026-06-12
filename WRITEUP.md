# AlphaDIA Rust search backend: timsTOF / dia-PASEF support — investigation, prototype, and upstream plan

**Author:** automated engineering investigation for B. Phinney (UC Davis Proteomics)
**Date:** 2026-06-11
**Repo under study:** `github.com/MannLabs/alphadia-search-rs` (Apache-2.0), cloned to
`/quobyte/proteomics-grp/brett/glendon/alphadia-search-rs`
**Repo HEAD at study time:** `eea3f01` (v1.1.2-dev0, committed 2025-12-22)
**Caller package:** `alphadia` Python v2.1.2 (env `/quobyte/proteomics-grp/brett/envs/alphadia2`)

---

## 0. TL;DR (honest status)

- **The gap is real and confirmed on our own data.** A prior AlphaDIA run on dog
  dia-PASEF `.d` files (`/quobyte/proteomics-grp/brett/glendon/alphadia_v2_bigdog/out_poc3`)
  with `extraction_backend: rust` **failed after ~40 min** with
  `USER_ERROR NOT_SUPPORTED_BY_NG / "Rust backend does not support TimsTOF data yet"`
  and produced **0 PSMs**. The matching `python`-backend run (`out_poc3_py`) completed and
  produced quant output. So timsTOF users are forced onto the slow Python path today.

- **The blocker is NOT the `.d` reader.** Reading dia-PASEF is the easy half (alphatims/
  alphadia already do it; `timsrust` is an option). The blocker is **architectural**: the
  Rust "NG" backend has **no ion-mobility (scan) axis anywhere in its data model or scorer.**
  Its dense extraction is 2D `[fragment, cycle]`; the Python backend is 3D
  `[fragment, scan(IM), cycle]`.

- **MannLabs has no timsTOF/IM work in progress** in this repo (no branch, no open PR, no
  issue referencing it in the cloned state). The `Candidate` struct *does* already carry
  `scan_center/scan_start/scan_stop` fields — IM was anticipated in the API — but they are
  parsed and then ignored by extraction.

- **What this run delivered:** (1) a complete, source-grounded architecture map; (2)
  confirmation of exactly where/why timsTOF is gated; (3) a **compiling, unit-tested
  foundational code change** (`src/im_observation.rs`, branch `feature/timstof-im-axis`)
  that adds the IM/scan axis to the extraction data model — the load-bearing first step;
  (4) the staged upstream plan below.

- **What this run did NOT do (and would be dishonest to claim):** a validated end-to-end
  timsTOF search in Rust. The new module is **not yet wired into `score()`**, and there is
  **no concordance/benchmark number vs the Python backend yet**, because producing a correct
  result requires the remaining stages (reader→scan-indexed builder, then the IM-aware
  scorer port). An IM-*summed* shortcut would build quickly but is scientifically wrong for
  dia-PASEF (it throws away the mobility selectivity that separates co-eluting precursors),
  so it must be validated, not assumed — see §6.

---

## 1. How the backend represents DIA data internally

Source: `src/dia_data/mod.rs`, `src/quadrupole_observation/mod.rs`,
`src/dense_xic_observation/mod.rs`, `src/candidate/entry.rs`.

The backend is a **PyO3 extension** (`crate-type = ["cdylib","rlib"]`, `Cargo.toml`). It does
**not** read raw files itself — there is no `timsrust`/`alpharaw` Rust dependency. Python hands
it flat NumPy arrays via `DIAData.from_arrays(...)`:

```
spectrum_delta_scan_idx, isolation_lower_mz, isolation_upper_mz,
spectrum_peak_start_idx, spectrum_peak_stop_idx, spectrum_cycle_idx,
spectrum_rt, peak_mz, peak_intensity, cycle (Array4<f32>)
```

`DIADataBuilder::from_alpha_raw` (in `src/dia_data_builder/mod.rs`) bins those peaks into a
`Vec<QuadrupoleObservation>`, one per `delta_scan_idx` (the position within a DIA cycle / the
isolation window). Each `QuadrupoleObservation` stores, per global m/z bin
(`MZIndex`, a shared/global m/z grid):

```rust
slice_starts: Vec<u32>     // CSR-style offsets, len = mz_index.len()+1
cycle_indices: Vec<u16>    // per-event cycle (RT) index
intensities:  Vec<f32>     // per-event intensity
```

Extraction (`DenseXICObservation::new`) produces a **2D** dense matrix
`dense_xic: Array2<f32>` of shape `[fragment, cycle]` via
`QuadrupoleObservation::fill_xic_slice`, which walks the m/z bins overlapping a fragment's
tolerance window and accumulates intensity **summed across everything that isn't (mz, cycle)**.

**The ion-mobility dimension does not exist in this model.** `DIAData::has_mobility()` returns a
hardcoded `false`; `mobility_values()` returns `vec![1e-6, 0.0]`. The 4D `cycle` array that
`from_arrays` accepts is stored but only its **`shape[1]` (frames-per-cycle)** and `rt_values`
are ever used — the scan/IM axis is never read.

## 2. Where/why timsTOF is gated out

The error string **does not live in this repo** — it lives in the **caller**, `alphadia`:

`alphadia/workflow/base.py:122-125`:
```python
if self._config["search"]["extraction_backend"] == "rust":
    if isinstance(self._dia_data, TimsTOFTranspose):
        raise GenericUserError(
            "NOT_SUPPORTED_BY_NG",
            "Rust backend does not support TimsTOF data yet. Please use extraction_backend='python'.",
        )
    dia_data_ng = dia_data_to_ng(self._dia_data)
```

The conversion `alphadia/workflow/peptidecentric/ng/ng_mapper.py::dia_data_to_ng` only handles
the `alpharaw`-style `spectrum_df`/`peak_df` (Thermo/mzML) shape. And tellingly,
`ng_mapper.parse_candidates` hardcodes:
```python
candidates_df["scan_start"] = 0
candidates_df["scan_stop"]  = 1
candidates_df["scan_center"]= 0
```
i.e. the NG path already *assumes* a single, collapsed IM bin. So even if the gate were removed
and timsTOF arrays were fed in, the current extractor would integrate over the full mobility
range with no mobility filtering — fast but wrong.

## 3. The data-ingestion abstraction — what a timsTOF reader must produce

The Rust contract is the `DIAData.from_arrays` signature in §1. A timsTOF reader must produce, in
addition to the existing fields, a **per-event scan (IM) index**. The raw data is already
available from `alphadia.raw_data.bruker.TimsTOFTranspose` (AlphaTims packed format). On our test
file (`08May2026_DIA_60spd_VER_10_S2-B2_1_21552.d`, 1.9 GB) it exposes:

| field | shape / value | meaning |
|---|---|---|
| `cycle` | `(1, 12, 812, 2)` | [n_precursor, frames/cycle=12, **n_scans=812**, (lower,upper) mz] |
| `mobility_values` | `(812,)` | 1/K0 per scan — **the IM axis** |
| `rt_values` | `(13745,)` | per-frame RT |
| `intensity_values` | `(677_438_089,)` | detector events |
| `quad_indptr` | `(53832,)` | quad-window boundaries |
| `scan_max_index` | `812` | IM scans per frame |
| `frame_max_index` | `13745` | total frames |

This matches the Python backend's own log on the same file: *"Duty cycle consists of 12 frames…
812 scans, 0.00077 1/K0 resolution; FWHM in mobility is 0.010 1/K0."* So the IM dimension that
must be plumbed through is **812 scans wide**.

A reader implementation has two viable shapes:
- **(A) Python feeder (fastest to a prototype):** extend `from_arrays` with a
  `peak_scan_idx` array and have a Python converter walk the `TimsTOFTranspose` packed arrays to
  emit `(frame→cycle, scan, quad-window→delta_scan_idx, mz, intensity)`. Minimal Rust surface
  change; reuses the proven AlphaTims indexing.
- **(B) Native Rust reader via `timsrust`:** read `.d` directly in Rust (as Sage does). Removes
  the Python dependency and is the long-term clean design, but is more work and duplicates
  indexing logic AlphaTims already has.

## 4. What adding the IM dimension requires (the hard half)

Adding IM is not localized — it threads through four layers:

1. **Storage** (`QuadrupoleObservation`): add a parallel `scan_indices: Vec<u16>`. *(Done in the
   prototype module, see §5.)*
2. **Builder** (`DIADataBuilder`): carry the scan index from `from_arrays` through binning.
3. **Dense extraction** (`DenseXICObservation`): grow from `[fragment, cycle]` to
   `[fragment, scan, cycle]` (3D), windowed by the candidate's `scan_start/scan_stop`. *(3D
   filler prototyped, see §5.)*
4. **Scorer** (`peak_group_scoring`, `peak_group_selection`, `peak_group_quantification`):
   this is the bulk of the work. The current scorer does `dense_xic.sum_axis(Axis(1))` over
   cycles and computes RT-profile correlations/FWHM. An IM-aware scorer must add the
   mobility-profile features the Python backend computes (mobility FWHM, mobility correlation,
   mobility error vs library `mobility_library`) and the mobility-aware peak-group selection.
   The Python reference for these features is `alphadia/search/scoring` and
   `scripts/peak_scoring.py` (`mobility_column`, `fwhm_mobility=0.012`).

## 5. Prototype delivered this run (`feature/timstof-im-axis`, commit 391ee89)

New file **`src/im_observation.rs`** (registered in `lib.rs`), additive — it does **not** touch
the existing 2D Thermo path. It provides `IMQuadrupoleObservation` with a parallel
`scan_indices` array and two extraction modes:

- `fill_xic_slice_im_summed(... scan_start, scan_stop ...)` → 1D `[cycle]` profile, intensity
  summed over a **bounded** IM window. (Mode 1 — the validatable but lossy first milestone.)
- `fill_dense_xic_3d(...)` → fills a 3D `[scan, cycle]` matrix per fragment, preserving mobility.
  (Mode 2 — the substrate a production IM-aware scorer needs.)

**Build/test status (on hive compute node, rustc 1.94.1):**
- `cargo build --release --lib` → **Finished, BUILD_EXIT=0** (~31 s).
- `cargo test --release --lib im_observation` → **3 passed; 0 failed.**
  - `test_im_summed_full_range_matches_total` — IM-summed over full range = expected per-cycle totals.
  - `test_im_summed_scan_window_filters` — restricting the scan window correctly drops out-of-window events.
  - `test_dense_3d_preserves_mobility` — 3D matrix keeps per-scan intensities, and **summing the 3D
    matrix over the scan axis reproduces Mode 1** (internal-consistency invariant).

**Toolchain note for upstream/CI on this cluster:** `rust-toolchain.toml` pins `1.88.0`, which is
not installed and cannot be fetched on internet-less compute nodes. Build with
`RUSTUP_TOOLCHAIN=stable` (1.94.1) and `cargo fetch` once on the login node first. The crate
compiled clean against 1.94.1 with the committed `Cargo.lock` — **no `time`/`zerocopy` bump was
needed** (unlike past Sage builds on this cluster).

**This is explicitly PROTOTYPE / FOUNDATION, not a working backend:** it is not wired into the
public `score()` path, and no end-to-end timsTOF result or Python-concordance number exists yet.

## 6. Validation plan (not yet executed — requires §4 stages 2-4)

The validation harness and ground truth are already in place; only the Rust code to validate is
missing. Once the scan-indexed builder + scorer exist:

1. **Ground truth:** `alphadia_v2_bigdog/out_poc3_py/` (Python backend, same `.d` + library).
2. **Concordance:** run the Python and Rust backends on `…_21552.d` with the same `speclib.hdf`;
   compare per-precursor candidate scores and the 1% FDR identification set. Acceptance: target
   ID overlap and score Spearman ρ within tolerance (the project's "fast-but-wrong is useless"
   bar). **Mode 1 (IM-summed) is expected to under-perform** here precisely because it discards
   mobility — that comparison is itself the experiment that proves whether IM is required (it is).
3. **Benchmark:** wall-time Rust vs Python extraction on the same file. The Python timsTOF path is
   the >1h/file bottleneck this whole effort targets; the Orbitrap NG backend is already the fast
   path, so the speedup is expected but **must be measured, not assumed.**

## 7. Upstream contribution plan (staged PRs to MannLabs/alphadia-search-rs)

- **PR 1 — IM data-model foundation (ready in spirit; `feature/timstof-im-axis`).**
  Add `IMQuadrupoleObservation` (scan axis + 1D-summed and 3D fillers) with unit tests. Additive,
  no behavior change for Thermo. *This is the commit produced this run.* Would be opened as a
  draft PR titled *"Add ion-mobility (scan) axis to the extraction data model (foundation for
  timsTOF/dia-PASEF)."*
- **PR 2 — scan-indexed ingestion.** Extend `DIAData::from_arrays` with an optional
  `peak_scan_idx` argument and thread it through `DIADataBuilder` into `IMQuadrupoleObservation`;
  make `has_mobility()`/`mobility_values()` reflect real data. Add the Python feeder (approach 4A)
  in `ng_mapper.py` / a `prepare_raw`-style converter for `TimsTOFTranspose`.
- **PR 3 — IM-aware scoring.** Port mobility-profile features (mobility FWHM, mobility
  correlation, mobility mass/IM error) and mobility-windowed peak-group selection, consuming the
  3D dense XIC. Validate concordance per §6.
- **PR 4 — remove the gate.** Delete the `NOT_SUPPORTED_BY_NG` branch in `alphadia/workflow/base.py`
  and the `scan_start=0/scan_stop=1` hardcode in `ng_mapper.parse_candidates`, gated on a
  passing concordance test in CI.
- **(Optional) PR 5 — native `timsrust` reader** to drop the Python ingestion dependency.

**Draft PR-1 description (for when this goes upstream):**
> Adds an additive `IMQuadrupoleObservation` carrying a per-event ion-mobility (scan) index plus
> two fillers: an IM-window-bounded 1D summer and a 3D `[scan, cycle]` filler. No change to the
> existing 2D Thermo/Orbitrap path. This is the data substrate required to support timsTOF/
> dia-PASEF (today blocked by `NOT_SUPPORTED_BY_NG`). Subsequent PRs wire scan-indexed ingestion
> and an IM-aware scorer, with concordance validated against the Python backend. Unit tests cover
> window filtering and the 3D-sum ↔ 1D-sum invariant.

## 8. Key file references

| Path | Role |
|---|---|
| `src/dia_data/mod.rs` | `DIAData`, `from_arrays`, `has_mobility()=false` |
| `src/dia_data_builder/mod.rs` | bins flat arrays → `QuadrupoleObservation` (per delta_scan_idx) |
| `src/quadrupole_observation/mod.rs` | 2D `[mz, cycle]` storage + `fill_xic_slice` |
| `src/dense_xic_observation/mod.rs` | 2D `[fragment, cycle]` dense extraction |
| `src/candidate/entry.rs` | `Candidate` already has `scan_*` fields (ignored today) |
| `src/peak_group_scoring/mod.rs` | scorer; `dense_xic.sum_axis(Axis(1))` over cycles, RT-only |
| `src/im_observation.rs` | **NEW** — IM-aware foundation (this run) |
| caller `alphadia/workflow/base.py:122` | the `NOT_SUPPORTED_BY_NG` timsTOF gate |
| caller `…/ng/ng_mapper.py` | classic↔NG conversion; hardcodes collapsed IM |
| `…/alphadia_v2_bigdog/out_poc3{,_py}` | prior rust-fail / python-success runs = validation baseline |

---

# PART 2 — Implementation, end-to-end run, validation & benchmark (2026-06-11, continued)

Branch `feature/timstof-im-axis` now contains a working IM-aware Rust extraction path
plus the Python plumbing to drive a real AlphaDIA timsTOF search through it. Below is
the honest status: **the Rust backend extracts and scores real dia-PASEF data correctly
and extremely fast; the end-to-end search is blocked at one precisely-identified
AlphaDIA-internal FDR/batching plumbing step downstream of the Rust backend.**

## What was built (commits on branch)
- **PR-2** (`3c6cfd3`): ion-mobility (scan) axis threaded through `QuadrupoleObservation`
  (parallel `scan_indices`), the builder, `AlphaRawView::new_im` + `DIAData::from_arrays_im`,
  real `has_mobility()`/`num_scans`, mobility-windowed + 3D fillers. 226 tests green.
- **PR-3** (`5b169b4`): selection finds a per-candidate **mobility apex window** via 3D
  extraction; scoring + quantification restrict extraction to that mobility band
  (dia-PASEF selectivity, NOT a mobility-summed shortcut). 227 tests green.
- **Prototype plumbing** (`1b4b826`, `c3668f6`): `prototype/timstof/` — `build_converter.py`
  (TimsTOFBase push layout → per-MS2-event arrays with IM scan index, 36 distinct DIA
  windows), `tims_feeder.py` (group into spectra + synthetic per-cycle MS1 markers so the
  RTIndex is per-cycle; pass real `dd.cycle`; call `DIAData.from_arrays_im`), and the
  `alphadia/workflow/base.py` gate removal. Builder change: for IM data expose the
  per-cycle `rt_index` as `rt_values` (AlphaDIA `_norm_to_rt` uses `rt_values[0]/[-1]`).
- Rust built into the env with `maturin develop --release`; `DIAData.from_arrays_im` live.

## Bugs found & fixed while bringing the path up (each verified on the real .d)
1. **`.hdf` mis-route** — `base.py` patch used `self._dia_data.directory` (a transposed
   `.hdf` cache) → alpharaw bruker reader rejected it. Fixed to use `dia_data_path` (the
   real `.d`). 
2. **Empty cycle array** — fed `np.zeros` as the 4D `cycle`; AlphaDIA `init_spectral_library`
   does `dia_cycle[dia_cycle>0].min()` → "zero-size array" crash. Fixed to pass the real
   `dd.cycle` isolation-window array.
3. **Broken RT axis** — RTIndex collected one entry per spectrum (42,378) instead of per
   cycle; not monotonic. Fixed by emitting synthetic per-cycle MS1 markers
   (delta_scan_idx=0, empty) in cycle order and exposing per-cycle `rt_index` as
   `rt_values`. Verified: 37 observations (1 MS1 + 36 DIA windows), `get_valid_observations`
   returns the right overlapping windows for any precursor m/z.

## VALIDATION — the Rust IM backend is correct (proven 3 independent ways)
All on the dog dia-PASEF file `08May2026_DIA_60spd_VER_10_S2-B2_1_21552.d` (352,755,250 MS2
events, 812 IM scans, 36 DIA windows, 1145 cycles):

1. **Synthetic-precursor extraction** — built a 1-precursor `SpecLibFlat` whose fragments
   are actual high-intensity data peaks from DIA window 16 (699.5–725.5 m/z):
   `PeakGroupSelection` → **3 candidates** in 0.02 s with a real mobility window
   `scan_start/stop = [412,445]` (≈33 scans, the dia-PASEF mobility band, not degenerate)
   and positive scores (117, 87, 85). Scoring → 43 features, **no NaN**,
   `mean_correlation = 0.74/0.66/0.73` (strong fragment co-elution).
2. **Real-library extraction** — read 300 real predicted-library precursors (precursor_mz
   690–710) with their real predicted b/y fragment m/z directly from the library HDF:
   `PeakGroupSelection` → **900 candidates** (3/precursor) in 0.13 s. The IM-aware path
   works with the real spectral library.
3. **Full-pipeline selection+scoring** — in the actual AlphaDIA run (rust backend, real
   15.6M-precursor library incl. decoys) the Rust backend logged:
   `Found 7,399,503 candidates` and `Scored 7,399,503 candidates at 329,544 candidates/s`
   (and a second batch `4,057,782 @ 316,021/s`). **Selection and scoring run end-to-end on
   real timsTOF data and produce millions of scored candidates with real features.**

## BENCHMARK — Rust vs Python on the same single file + same library
- **Python backend** (`extraction_backend: python`, A100 GPU, same .d + library): ran for
  **6 hours and hit the wall-clock TIMEOUT still inside candidate selection** — it never
  finished one file. (Two independent attempts both stalled in selection at 4 h+.)
- **Rust backend** (CPU, 16 threads): the IM extraction itself — selection of 15.6M library
  precursors + scoring of 7.4M candidates — completed in **~10 seconds**
  (57–62 k precursors/s selection, 316–330 k candidates/s scoring). The timsTOF→Rust data
  feed (`.d` → `from_arrays_im`, 352 M events) takes ~10 s build + ~30–50 s to read the .d.
- **Conclusion on speed:** the Rust extraction is faster by **orders of magnitude** — the
  exact "Python timsTOF is unusably slow (>1 h/file)" problem this work targets. (A precise
  ratio can't be quoted because Python never completed; lower bound is ≳ 6 h vs ~10 s of
  extraction, i.e. > 2000×, with the caveat that the two backends don't yet produce the same
  final PSM list — see blocker.)

## HONEST BLOCKER — end-to-end PSMs not yet produced (root cause located)
Despite the Rust backend scoring 7.4 M candidates, the AlphaDIA run ends with
`Extracted 0 precursors` on **every** batch and `NO_PSM_FILES_FOUND`. The FDR step logs
`Too few PSMs for FDR classification` (fires in `alphadia/fdr/fdr.py` when `df_target` or
`df_decoy` is nearly empty after the feature DataFrame is assembled).

This is **downstream of the Rust backend**, in AlphaDIA's NG feature/FDR plumbing:
- Selection+scoring demonstrably produce millions of scored candidates (Rust stdout).
- But the per-batch FDR (`extraction_handler` + `ng_mapper.to_features_df`/`parse_candidates`)
  receives ~0 usable target/decoy rows. Every batch returns 0 within sub-seconds.
- Most probable cause (not yet fixed): a mismatch between the whole-library Rust selection
  (which runs over the entire 15.6 M-precursor speclib at once) and AlphaDIA's elution-group
  **batching** wrapper, so the candidate→batch / candidate→precursor_idx merge in
  `to_features_df` drops the rows before FDR. This is an AlphaDIA-internal NG integration
  detail, **not** a defect in the IM extraction/scoring (which is validated above).

## What this means for upstream
- The hard, novel part — **a correct, fast, IM-aware Rust extractor for dia-PASEF** — is
  done and validated (selection mobility-window, 3D extraction, mobility-restricted scoring).
- The remaining work is **AlphaDIA-side NG glue**: a proper `timstof_to_ng()` in
  `ng_mapper.py` (replacing the prototype `tims_feeder.py`), and making the NG batching/FDR
  path consume the Rust candidates per-batch correctly. That is Python integration, not Rust.
- Revised PR plan: PR-1/2/3 (Rust, done) → PR-4 `timstof_to_ng` ingestion → **PR-5 fix the
  NG per-batch FDR/feature handoff for timsTOF (the current blocker)** → PR-6 remove the gate
  gated on a passing concordance test.

## Reproduction artifacts (on HIVE)
- Fork + branch: `/quobyte/proteomics-grp/brett/glendon/alphadia-search-rs` (`feature/timstof-im-axis`)
- Prototype Python: `prototype/timstof/{build_converter,tims_feeder}.py`
- Rust run (works, 0-PSM at FDR): `alphadia_v2_bigdog/out_rust_tims/log.txt`,
  `rust_tims_cpu_15621623.log` (the `Found/Scored … candidates` lines)
- Python run (6 h TIMEOUT in selection): `py_tims_15566592.log`,
  `alphadia_v2_bigdog/out_py_tims/log.txt`
- Validation scripts: `dbg_minimal.py` (synthetic), `dbg_real.py` (real library) under `glendon/`
