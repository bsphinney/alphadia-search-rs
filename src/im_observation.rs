//! IM-aware (ion-mobility) extraction primitives for timsTOF / dia-PASEF.
//!
//! STATUS: PROTOTYPE / FOUNDATIONAL. This module adds the ion-mobility (scan)
//! axis that the original `QuadrupoleObservation` / `DenseXICObservation` data
//! model collapses. It is **additive**: it does not change the existing 2D
//! `[fragment, cycle]` extraction path used by Thermo/Orbitrap data. It provides
//! the storage + extraction needed to grow toward a full 3D
//! `[fragment, scan(IM), cycle]` extractor matching AlphaDIA's Python backend.
//!
//! Two modes are provided:
//!   1. `IMQuadrupoleObservation::fill_xic_slice_im_summed` — sums intensity over
//!      the IM/scan window [scan_start, scan_stop). This reproduces what the
//!      AlphaDIA NG mapper *pretends* to do today (scan_start=0, scan_stop=1) but
//!      does it correctly over the real scan range. It is LOSSY w.r.t. mobility
//!      selectivity and is intended only as a validatable first milestone.
//!   2. `IMQuadrupoleObservation::fill_dense_xic_3d` — fills a true 3D
//!      `[scan, cycle]` slice per fragment. This is the data substrate a
//!      production IM-aware scorer needs; the scorer itself is NOT yet ported.
//!
//! NONE of this is wired into the public `score()` path yet — see WRITEUP.md for
//! the staged plan.

use crate::mz_index::MZIndex;
use numpy::ndarray::{ArrayViewMut1, ArrayViewMut2};

/// An IM-aware quadrupole observation. Mirrors `QuadrupoleObservation` but stores
/// a parallel `scan_indices` array so that the ion-mobility (scan) coordinate of
/// each detector event is preserved instead of being summed away at build time.
#[derive(Debug, Clone, Default)]
pub struct IMQuadrupoleObservation {
    pub isolation_window: [f32; 2],
    pub num_cycles: usize,
    pub num_scans: usize,

    /// Start indices for each mz_index slice. Length = mz_index.len() + 1
    pub slice_starts: Vec<u32>,

    /// Per-event cycle index (concatenated over all mz slices)
    pub cycle_indices: Vec<u16>,

    /// Per-event scan (ion-mobility) index, parallel to `cycle_indices`
    pub scan_indices: Vec<u16>,

    /// Per-event intensity, parallel to `cycle_indices`
    pub intensities: Vec<f32>,
}

impl IMQuadrupoleObservation {
    pub fn new_with_capacity(
        isolation_window: [f32; 2],
        num_cycles: usize,
        num_scans: usize,
        mz_index_len: usize,
        total_peaks: usize,
    ) -> Self {
        let mut slice_starts = Vec::with_capacity(mz_index_len + 1);
        slice_starts.push(0);
        Self {
            isolation_window,
            num_cycles,
            num_scans,
            slice_starts,
            cycle_indices: Vec::with_capacity(total_peaks),
            scan_indices: Vec::with_capacity(total_peaks),
            intensities: Vec::with_capacity(total_peaks),
        }
    }

    pub fn add_peak_data(&mut self, cycle_idx: u16, scan_idx: u16, intensity: f32) {
        self.cycle_indices.push(cycle_idx);
        self.scan_indices.push(scan_idx);
        self.intensities.push(intensity);
    }

    pub fn finalize_slice(&mut self) {
        self.slice_starts.push(self.cycle_indices.len() as u32);
    }

    /// MODE 1 (lossy, IM-summed): accumulate intensity over the IM window
    /// `[scan_start, scan_stop)` into a 1D `[cycle]` profile. Equivalent to the
    /// existing 2D path when `scan_start=0, scan_stop=num_scans`.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_xic_slice_im_summed(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        scan_start: usize,
        scan_stop: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        let delta_mz = mz * mass_tolerance * 1e-6;
        let lower_mz = mz - delta_mz;
        let upper_mz = mz + delta_mz;

        for mz_idx in mz_index.mz_range_indices(lower_mz, upper_mz) {
            let start = self.slice_starts[mz_idx] as usize;
            let stop = self.slice_starts[mz_idx + 1] as usize;
            for i in start..stop {
                let cycle_idx = self.cycle_indices[i] as usize;
                if cycle_idx < cycle_start_idx || cycle_idx >= cycle_stop_idx {
                    continue;
                }
                let scan_idx = self.scan_indices[i] as usize;
                if scan_idx < scan_start || scan_idx >= scan_stop {
                    continue;
                }
                dense_xic[cycle_idx - cycle_start_idx] += self.intensities[i];
            }
        }
    }

    /// MODE 2 (full 3D substrate): fill a `[scan, cycle]` dense matrix for one
    /// fragment m/z. This preserves the ion-mobility dimension and is the data
    /// a production IM-aware scorer (mobility correlation, mobility FWHM, etc.)
    /// would consume. Rows outside `[scan_start, scan_stop)` are left untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_dense_xic_3d(
        &self,
        mz_index: &MZIndex,
        dense_xic_2d: &mut ArrayViewMut2<f32>, // shape [scan, cycle]
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        scan_start: usize,
        scan_stop: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        let delta_mz = mz * mass_tolerance * 1e-6;
        let lower_mz = mz - delta_mz;
        let upper_mz = mz + delta_mz;

        for mz_idx in mz_index.mz_range_indices(lower_mz, upper_mz) {
            let start = self.slice_starts[mz_idx] as usize;
            let stop = self.slice_starts[mz_idx + 1] as usize;
            for i in start..stop {
                let cycle_idx = self.cycle_indices[i] as usize;
                if cycle_idx < cycle_start_idx || cycle_idx >= cycle_stop_idx {
                    continue;
                }
                let scan_idx = self.scan_indices[i] as usize;
                if scan_idx < scan_start || scan_idx >= scan_stop {
                    continue;
                }
                dense_xic_2d[[scan_idx - scan_start, cycle_idx - cycle_start_idx]] +=
                    self.intensities[i];
            }
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        std::mem::size_of::<[f32; 2]>()
            + 2 * std::mem::size_of::<usize>()
            + self.slice_starts.len() * std::mem::size_of::<u32>()
            + self.cycle_indices.len() * std::mem::size_of::<u16>()
            + self.scan_indices.len() * std::mem::size_of::<u16>()
            + self.intensities.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mz_index::MZIndex;
    use numpy::ndarray::{Array1, Array2};

    fn build_two_scan_obs() -> IMQuadrupoleObservation {
        // One fragment m/z that lands in a single mz_index bin, present in two
        // IM scans (3 and 7) across two cycles (0 and 1).
        let mz_index = MZIndex::global();
        let mz = 500.0_f32;
        let mz_idx = mz_index.find_closest_index(mz);
        let mut obs =
            IMQuadrupoleObservation::new_with_capacity([400.0, 600.0], 2, 8, mz_index.len(), 4);
        // events: (cycle, scan, intensity) — must be appended into the matching mz slice
        let events = [(0u16, 3u16, 10.0f32), (1, 3, 20.0), (0, 7, 5.0), (1, 7, 7.0)];
        let mut current = 0usize;
        while current < mz_index.len() {
            if current == mz_idx {
                for (c, s, inten) in events.iter() {
                    obs.add_peak_data(*c, *s, *inten);
                }
            }
            obs.finalize_slice();
            current += 1;
        }
        obs
    }

    #[test]
    fn test_im_summed_full_range_matches_total() {
        let obs = build_two_scan_obs();
        let mz_index = MZIndex::global();
        let mut xic = Array1::<f32>::zeros(2);
        let mut view = xic.view_mut();
        obs.fill_xic_slice_im_summed(mz_index, &mut view, 0, 2, 0, 8, 20.0, 500.0);
        // cycle 0: 10 + 5 = 15 ; cycle 1: 20 + 7 = 27
        assert!((xic[0] - 15.0).abs() < 1e-4, "cycle0 {}", xic[0]);
        assert!((xic[1] - 27.0).abs() < 1e-4, "cycle1 {}", xic[1]);
    }

    #[test]
    fn test_im_summed_scan_window_filters() {
        let obs = build_two_scan_obs();
        let mz_index = MZIndex::global();
        let mut xic = Array1::<f32>::zeros(2);
        let mut view = xic.view_mut();
        // restrict to scan window [3,4): only scan 3 events survive
        obs.fill_xic_slice_im_summed(mz_index, &mut view, 0, 2, 3, 4, 20.0, 500.0);
        assert!((xic[0] - 10.0).abs() < 1e-4, "cycle0 {}", xic[0]);
        assert!((xic[1] - 20.0).abs() < 1e-4, "cycle1 {}", xic[1]);
    }

    #[test]
    fn test_dense_3d_preserves_mobility() {
        let obs = build_two_scan_obs();
        let mz_index = MZIndex::global();
        // scan window [0,8) -> 8 rows, 2 cycles
        let mut m = Array2::<f32>::zeros((8, 2));
        let mut view = m.view_mut();
        obs.fill_dense_xic_3d(mz_index, &mut view, 0, 2, 0, 8, 20.0, 500.0);
        assert!((m[[3, 0]] - 10.0).abs() < 1e-4);
        assert!((m[[3, 1]] - 20.0).abs() < 1e-4);
        assert!((m[[7, 0]] - 5.0).abs() < 1e-4);
        assert!((m[[7, 1]] - 7.0).abs() < 1e-4);
        // summing over the scan axis must equal the IM-summed mode
        let col0: f32 = (0..8).map(|s| m[[s, 0]]).sum();
        assert!((col0 - 15.0).abs() < 1e-4);
    }
}
