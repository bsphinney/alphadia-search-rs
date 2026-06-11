"""Convert TimsTOFBase (non-transposed alphatims layout) to flat arrays for
DIAData.from_arrays_im, preserving the IM/scan axis.

delta_scan_idx = unique id per DISTINCT (iso_lo, iso_hi) DIA isolation window.
"""
import numpy as np

def convert(dd, frames_per_cycle):
    n_scans = dd.scan_max_index
    push_indptr = dd.push_indptr
    tof_indices = dd.tof_indices
    intensity_values = dd.intensity_values
    mz_values = dd.mz_values
    quad_indptr = dd.quad_indptr
    quad_mz = np.asarray(dd.quad_mz_values)

    n_pushes = len(push_indptr) - 1
    n_frames = n_pushes // n_scans
    counts = np.diff(push_indptr)
    push_ids = np.arange(n_pushes, dtype=np.int64)
    frame_of_push = push_ids // n_scans
    scan_of_push = push_ids % n_scans

    ev_frame = np.repeat(frame_of_push, counts)
    ev_scan = np.repeat(scan_of_push, counts).astype(np.int64)
    ev_mz = mz_values[tof_indices].astype(np.float32)
    ev_int = intensity_values.astype(np.float32)

    n_events = len(ev_mz)
    ev_bin = np.searchsorted(quad_indptr, np.arange(n_events, dtype=np.int64), side='right') - 1
    ev_iso_lo = quad_mz[ev_bin, 0].astype(np.float32)
    ev_iso_hi = quad_mz[ev_bin, 1].astype(np.float32)

    # MS2 events only: quad mz > 0
    ms2 = ev_iso_lo > 0
    ev_frame = ev_frame[ms2]; ev_scan = ev_scan[ms2]; ev_mz = ev_mz[ms2]
    ev_int = ev_int[ms2]; ev_iso_lo = ev_iso_lo[ms2]; ev_iso_hi = ev_iso_hi[ms2]

    cycle_idx = (ev_frame // frames_per_cycle).astype(np.int64)

    # delta_scan_idx = unique id per distinct (iso_lo, iso_hi) window
    win = np.stack([ev_iso_lo, ev_iso_hi], axis=1)
    uniq, inv = np.unique(win, axis=0, return_inverse=True)
    delta_scan_idx = inv.astype(np.int64)

    return dict(
        n_events=len(ev_mz), n_frames=n_frames, n_scans=n_scans,
        ev_mz=ev_mz, ev_int=ev_int, ev_scan=ev_scan,
        cycle_idx=cycle_idx, delta_scan_idx=delta_scan_idx,
        iso_lo=ev_iso_lo, iso_hi=ev_iso_hi,
        n_windows=len(uniq), windows=uniq,
    )
