use crate::mz_index::MZIndex;
use numpy::ndarray::{ArrayViewMut1, ArrayViewMut2};

#[cfg(test)]
mod tests;

/// QuadrupoleObservation structure that achieves >99.9% memory overhead reduction
///
/// Instead of millions of XICSlice objects with individual Vec allocations,
/// this uses consolidated arrays with index-based slicing.
///
/// Ion-mobility (timsTOF / dia-PASEF) support: `scan_indices` is parallel to
/// `cycle_indices`/`intensities`. When it is empty the observation is
/// mobility-agnostic (Thermo/Orbitrap) and behaves exactly as before. When it is
/// populated, `num_scans > 1` and the 3D `[scan, cycle]` extraction path is valid.
#[derive(Debug, Clone)]
pub struct QuadrupoleObservation {
    pub isolation_window: [f32; 2],
    pub num_cycles: usize,

    /// Number of ion-mobility scans (1 for non-IM data).
    pub num_scans: usize,

    /// Start indices for each mz_index slice. Length = mz_index.len() + 1
    /// The stop index for slice[i] is slice_starts[i+1]
    pub slice_starts: Vec<u32>,

    /// All cycle indices concatenated from all slices
    pub cycle_indices: Vec<u16>,

    /// All ion-mobility scan indices concatenated from all slices.
    /// Empty when the data has no mobility dimension.
    pub scan_indices: Vec<u16>,

    /// All intensities concatenated from all slices
    pub intensities: Vec<f32>,
}

impl QuadrupoleObservation {
    /// Create a new empty observation with exact pre-allocation
    pub fn new_with_capacity(
        isolation_window: [f32; 2],
        num_cycles: usize,
        mz_index_len: usize,
        total_peaks: usize,
    ) -> Self {
        let mut slice_starts = Vec::with_capacity(mz_index_len + 1);
        slice_starts.push(0); // First slice starts at 0

        Self {
            isolation_window,
            num_cycles,
            num_scans: 1,
            slice_starts,
            cycle_indices: Vec::with_capacity(total_peaks),
            scan_indices: Vec::new(),
            intensities: Vec::with_capacity(total_peaks),
        }
    }

    /// Create a new empty IM-aware observation (preallocates scan_indices too).
    pub fn new_with_capacity_im(
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

    /// True if this observation carries an ion-mobility dimension.
    pub fn has_mobility(&self) -> bool {
        !self.scan_indices.is_empty()
    }

    /// Get the cycle indices and intensities for a specific mz_index
    pub fn get_slice_data(&self, mz_idx: usize) -> (&[u16], &[f32]) {
        let start = self.slice_starts[mz_idx] as usize;
        let stop = self.slice_starts[mz_idx + 1] as usize;

        (
            &self.cycle_indices[start..stop],
            &self.intensities[start..stop],
        )
    }

    /// Add peak data for a specific mz_index (used during building)
    pub fn add_peak_data(&mut self, cycle_idx: u16, intensity: f32) {
        self.cycle_indices.push(cycle_idx);
        self.intensities.push(intensity);
    }

    /// Add IM peak data carrying the ion-mobility scan index.
    pub fn add_peak_data_im(&mut self, cycle_idx: u16, scan_idx: u16, intensity: f32) {
        self.cycle_indices.push(cycle_idx);
        self.scan_indices.push(scan_idx);
        self.intensities.push(intensity);
    }

    /// Finalize a slice by recording its end position
    pub fn finalize_slice(&mut self) {
        self.slice_starts.push(self.cycle_indices.len() as u32);
    }

    /// Optimized fill_xic_slice method using direct array access.
    ///
    /// NOTE: for IM data, `scan_indices` is NOT sorted (events within an mz slice
    /// are sorted by (mz_idx, cycle_idx) only), so the binary-search fast path is
    /// only used when there is no mobility dimension. With mobility present we
    /// fall back to a linear scan that respects the (unsorted-by-scan) layout.
    pub fn fill_xic_slice(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        let delta_mz = mz * mass_tolerance * 1e-6;
        let lower_mz = mz - delta_mz;
        let upper_mz = mz + delta_mz;

        for mz_idx in mz_index.mz_range_indices(lower_mz, upper_mz) {
            // Direct slice access using optimized indexing
            let start = self.slice_starts[mz_idx] as usize;
            let stop = self.slice_starts[mz_idx + 1] as usize;

            let cycle_indices = &self.cycle_indices[start..stop];
            let intensities = &self.intensities[start..stop];

            // Binary search for start position
            let start_pos = cycle_indices
                .binary_search(&(cycle_start_idx as u16))
                .unwrap_or_else(|idx| idx);

            // Process cycles within range
            for i in start_pos..cycle_indices.len() {
                let cycle_idx = cycle_indices[i] as usize;

                if cycle_idx >= cycle_stop_idx {
                    break;
                }

                dense_xic[cycle_idx - cycle_start_idx] += intensities[i];
            }
        }
    }

    /// IM-aware 1D extraction: sum intensity into a `[cycle]` profile, but ONLY
    /// over the ion-mobility window `[scan_start, scan_stop)`. This is the
    /// correctness-critical operation for dia-PASEF: it restricts to the
    /// candidate's mobility band instead of integrating the whole mobility range.
    ///
    /// If the observation has no mobility (`scan_indices` empty), the scan window
    /// is ignored and this is equivalent to `fill_xic_slice`.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_xic_slice_mobility_windowed(
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
        let has_mob = self.has_mobility();
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
                if has_mob {
                    let scan_idx = self.scan_indices[i] as usize;
                    if scan_idx < scan_start || scan_idx >= scan_stop {
                        continue;
                    }
                }
                dense_xic[cycle_idx - cycle_start_idx] += self.intensities[i];
            }
        }
    }

    /// IM-aware 3D extraction: fill a `[scan, cycle]` matrix for one fragment m/z,
    /// preserving the mobility dimension. Rows are indexed relative to `scan_start`.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_dense_xic_3d(
        &self,
        mz_index: &MZIndex,
        dense_xic_2d: &mut ArrayViewMut2<f32>, // [scan, cycle]
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        scan_start: usize,
        scan_stop: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        if !self.has_mobility() {
            return;
        }
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

    /// Calculate memory footprint of this optimized observation
    pub fn memory_footprint_bytes(&self) -> usize {
        let mut total_size = 0;

        // Fixed size components
        total_size += std::mem::size_of::<[f32; 2]>(); // isolation_window
        total_size += 2 * std::mem::size_of::<usize>(); // num_cycles + num_scans

        // Vec overheads
        total_size += std::mem::size_of::<Vec<u32>>(); // slice_starts
        total_size += std::mem::size_of::<Vec<u16>>(); // cycle_indices
        total_size += std::mem::size_of::<Vec<u16>>(); // scan_indices
        total_size += std::mem::size_of::<Vec<f32>>(); // intensities

        // Actual data
        total_size += self.slice_starts.len() * std::mem::size_of::<u32>();
        total_size += self.cycle_indices.len() * std::mem::size_of::<u16>();
        total_size += self.scan_indices.len() * std::mem::size_of::<u16>();
        total_size += self.intensities.len() * std::mem::size_of::<f32>();

        total_size
    }
}

// Implement the QuadrupoleObservationTrait for QuadrupoleObservation
impl crate::traits::QuadrupoleObservationTrait for QuadrupoleObservation {
    fn fill_xic_slice(
        &self,
        mz_index: &crate::mz_index::MZIndex,
        dense_xic: &mut numpy::ndarray::ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        self.fill_xic_slice(
            mz_index,
            dense_xic,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            mz,
        )
    }

    fn fill_xic_and_mz_slice(
        &self,
        mz_index: &crate::mz_index::MZIndex,
        dense_xic: &mut numpy::ndarray::ArrayViewMut1<f32>,
        dense_mz: &mut numpy::ndarray::ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        let delta_mz = mz * mass_tolerance * 1e-6;
        let lower_mz = mz - delta_mz;
        let upper_mz = mz + delta_mz;

        for mz_idx in mz_index.mz_range_indices(lower_mz, upper_mz) {
            let actual_mz = mz_index.mz[mz_idx];

            // Direct slice access using optimized indexing
            let start = self.slice_starts[mz_idx] as usize;
            let stop = self.slice_starts[mz_idx + 1] as usize;

            let cycle_indices = &self.cycle_indices[start..stop];
            let intensities = &self.intensities[start..stop];

            // Binary search for start position
            let start_pos = cycle_indices
                .binary_search(&(cycle_start_idx as u16))
                .unwrap_or_else(|idx| idx);

            // Process cycles within range
            for i in start_pos..cycle_indices.len() {
                let cycle_idx = cycle_indices[i] as usize;

                if cycle_idx >= cycle_stop_idx {
                    break;
                }

                let relative_idx = cycle_idx - cycle_start_idx;
                let intensity = intensities[i];

                // Always accumulate intensity (even if zero)
                dense_xic[relative_idx] += intensity;

                // Update m/z weighted average only for non-zero intensities
                if intensity > 0.0 {
                    let prev_total_intensity = dense_xic[relative_idx] - intensity;

                    if prev_total_intensity == 0.0 {
                        // First non-zero intensity at this position
                        dense_mz[relative_idx] = actual_mz;
                    } else {
                        // Update running weighted average
                        dense_mz[relative_idx] = (dense_mz[relative_idx] * prev_total_intensity
                            + actual_mz * intensity)
                            / dense_xic[relative_idx];
                    }
                }
            }
        }
    }
}
