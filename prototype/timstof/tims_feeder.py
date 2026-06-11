"""Feed a timsTOF .d into the IM-aware Rust DIAData via from_arrays_im.

Groups MS2 events into spectra (one per (delta_scan_idx, cycle_idx)) with
contiguous peak ranges, preserving the per-peak ion-mobility scan index.
"""
import numpy as np, time, sys
sys.path.insert(0, "/quobyte/proteomics-grp/brett/glendon")
from build_converter import convert


def build_dia_data_ng_im(dd, frames_per_cycle, DIAData):
    out = convert(dd, frames_per_cycle)
    n_scans = out["n_scans"]
    dsi = out["delta_scan_idx"]
    cyc = out["cycle_idx"]
    # spectrum key = dsi * (cycle_max+1) + cyc ; sort events by it
    cyc_span = int(cyc.max()) + 1
    key = dsi.astype(np.int64) * cyc_span + cyc.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    peak_mz = out["ev_mz"][order]
    peak_int = out["ev_int"][order]
    peak_scan = out["ev_scan"][order].astype(np.int64)
    iso_lo = out["iso_lo"][order]
    iso_hi = out["iso_hi"][order]

    # spectrum boundaries = where key changes
    change = np.empty(len(key_s), dtype=bool)
    change[0] = True
    change[1:] = key_s[1:] != key_s[:-1]
    starts = np.flatnonzero(change)
    stops = np.empty_like(starts)
    stops[:-1] = starts[1:]
    stops[-1] = len(key_s)

    spec_key = key_s[starts]
    spec_dsi = (spec_key // cyc_span).astype(np.int64)
    spec_cyc = (spec_key % cyc_span).astype(np.int64)
    spec_iso_lo = iso_lo[starts].astype(np.float32)
    spec_iso_hi = iso_hi[starts].astype(np.float32)
    spec_rt = spec_cyc.astype(np.float32)  # placeholder rt per cycle (monotonic)

    # real RT per cycle: use frame rt. rt_values length n_frames; cycle rt = rt of MS1 frame of that cycle
    rt_values = np.asarray(dd.rt_values)
    # cycle c spans frames [c*fpc, (c+1)*fpc); take first frame rt
    cyc_rt = rt_values[(spec_cyc * frames_per_cycle).clip(0, len(rt_values)-1)].astype(np.float32)

    n_spec = len(starts)
    spectrum_peak_start = starts.astype(np.int64)
    spectrum_peak_stop = stops.astype(np.int64)

    # Use the REAL timsTOF DIA cycle array (shape [1, frames_per_cycle, n_scans, 2]
    # holding the per-(frame,scan) isolation [lower_mz, upper_mz]). The Rust
    # extraction only reads shape[1], but alphadia's init_spectral_library reads
    # the actual m/z values (dia_cycle[dia_cycle>0].min/max) to set library mz
    # limits, so a zeros array breaks it. This matches what the python backend
    # passes (dia_data.cycle).
    cycle_arr = np.asarray(dd.cycle).astype(np.float32)

    t = time.time()
    ng = DIAData.from_arrays_im(
        spec_dsi,                         # spectrum_delta_scan_idx
        spec_iso_lo,                      # isolation_lower_mz
        spec_iso_hi,                      # isolation_upper_mz
        spectrum_peak_start,              # spectrum_peak_start_idx
        spectrum_peak_stop,               # spectrum_peak_stop_idx
        spec_cyc,                         # spectrum_cycle_idx
        cyc_rt,                           # spectrum_rt (seconds)
        peak_mz.astype(np.float32),       # peak_mz
        peak_int.astype(np.float32),      # peak_intensity
        cycle_arr,                        # cycle (4D)
        peak_scan,                        # peak_scan_idx
        int(n_scans),                     # num_scans
    )
    build_sec = time.time() - t
    return ng, dict(n_spectra=n_spec, n_peaks=len(peak_mz), build_sec=build_sec,
                    n_scans=n_scans, frames_per_cycle=frames_per_cycle)
