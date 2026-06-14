//! Ms1Observation: survey-frame (MS1) peak store for precursor / isotopologue XICs.
//!
//! Parallel to `QuadrupoleObservation` but for the MS1 survey frames the engine
//! previously discarded (see A1: `build_converter.convert()` used to drop every
//! event with `iso_lo <= 0`). The cache now persists `ms1_mz/ms1_int/ms1_scan/
//! ms1_cycle_idx`; this struct indexes those flat arrays by m/z so a target m/z
//! (precursor or +1/+2/+3 C13 isotope) can be extracted as a per-cycle
//! chromatogram over the SAME RT(cycle) x IM(scan) window as the fragment XICs.
//!
//! Additive + non-breaking: built only when the Python feeder calls
//! `DIAData.set_ms1_arrays`. When absent, `DIAData::has_ms1()` is false and the
//! scoring path computes the MS1/isotope features as 0.0 (no-op for the NN).

use numpy::ndarray::Array1;

/// MS1 survey peaks sorted by m/z, with parallel (cycle, scan, intensity) arrays.
///
/// Extraction does a binary search over the sorted m/z to find the `[mz-tol, mz+tol]`
/// band, then linearly accumulates intensity per cycle within the candidate's
/// RT(cycle) and IM(scan) window. This mirrors the mass-tolerance + IM-window
/// logic in `QuadrupoleObservation::fill_xic_slice_mobility_windowed`, but over
/// a single global MS1 peak list (there is no quadrupole isolation for MS1).
pub struct Ms1Observation {
    /// MS1 peak m/z values, ascending-sorted (the sort key).
    pub mz: Vec<f32>,
    /// MS1 peak intensities, in the same (m/z-sorted) order.
    pub intensity: Vec<f32>,
    /// MS1 peak RT cycle index, same order.
    pub cycle: Vec<u32>,
    /// MS1 peak IM scan index, same order.
    pub scan: Vec<u32>,
    /// Number of ion-mobility scans (for window clamping; informational).
    pub num_scans: usize,
}

impl Ms1Observation {
    /// Build from the cached flat MS1 arrays. Sorts all peaks by m/z so the
    /// per-target band can be found by binary search.
    pub fn from_arrays(mz: Vec<f32>, intensity: Vec<f32>, cycle: Vec<u32>, scan: Vec<u32>, num_scans: usize) -> Self {
        let n = mz.len();
        // sort by m/z ascending
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| mz[a].partial_cmp(&mz[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mz_s: Vec<f32> = order.iter().map(|&i| mz[i]).collect();
        let int_s: Vec<f32> = order.iter().map(|&i| intensity[i]).collect();
        let cyc_s: Vec<u32> = order.iter().map(|&i| cycle[i]).collect();
        let scan_s: Vec<u32> = order.iter().map(|&i| scan[i]).collect();
        Self {
            mz: mz_s,
            intensity: int_s,
            cycle: cyc_s,
            scan: scan_s,
            num_scans,
        }
    }

    /// Number of MS1 peaks.
    #[inline]
    pub fn len(&self) -> usize {
        self.mz.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mz.is_empty()
    }

    /// Extract a per-cycle MS1 chromatogram for `target_mz` over the candidate's
    /// RT(cycle) window `[cycle_start, cycle_stop)` and IM(scan) window
    /// `[scan_start, scan_stop)`, using a symmetric ppm `mass_tolerance`.
    ///
    /// Returns a `Vec<f32>` of length `cycle_stop - cycle_start` (cycle-relative),
    /// summing the intensity of every MS1 peak whose m/z falls in the tolerance
    /// band and whose (cycle, scan) lies inside the window. All-zero if no MS1
    /// signal (e.g. empty store or target out of range) — the caller's correlation
    /// helper treats an all-zero profile as 0.0 correlation.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_xic(
        &self,
        target_mz: f32,
        cycle_start: usize,
        cycle_stop: usize,
        scan_start: usize,
        scan_stop: usize,
        mass_tolerance: f32,
    ) -> Array1<f32> {
        let n_cyc = cycle_stop.saturating_sub(cycle_start);
        let mut xic = Array1::<f32>::zeros(n_cyc);
        if self.mz.is_empty() || n_cyc == 0 || target_mz <= 0.0 {
            return xic;
        }
        let delta = target_mz * mass_tolerance * 1e-6;
        let lo = target_mz - delta;
        let hi = target_mz + delta;
        // binary search for the lower bound of the m/z band
        let start = self.mz.partition_point(|&m| m < lo);
        let mut i = start;
        while i < self.mz.len() && self.mz[i] <= hi {
            let c = self.cycle[i] as usize;
            if c >= cycle_start && c < cycle_stop {
                let s = self.scan[i] as usize;
                if s >= scan_start && s < scan_stop {
                    xic[c - cycle_start] += self.intensity[i];
                }
            }
            i += 1;
        }
        xic
    }

    /// Memory footprint in bytes (for the DIAData accounting).
    pub fn memory_footprint_bytes(&self) -> usize {
        self.mz.len() * std::mem::size_of::<f32>()
            + self.intensity.len() * std::mem::size_of::<f32>()
            + self.cycle.len() * std::mem::size_of::<u32>()
            + self.scan.len() * std::mem::size_of::<u32>()
    }
}

#[cfg(test)]
mod tests;
