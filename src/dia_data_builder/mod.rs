use crate::dia_data::AlphaRawView;
use crate::dia_data::DIAData;
use crate::mz_index::MZIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::rt_index::RTIndex;

/// DIAData builder with single-pass binning and full parallelization
pub struct DIADataBuilder;

impl DIADataBuilder {
    pub fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> DIAData {
        let rt_index = RTIndex::from_alpha_raw(alpha_raw_view);

        // Single-phase parallel observation building
        let quadrupole_observations = Self::build_observations_rayon_parallel(alpha_raw_view);

        // For mobility data the per-spectrum rt array is not a clean per-cycle RT
        // axis (MS1 markers + many MS2 windows), so expose the per-cycle rt_index
        // as `rt_values`. alphadia's _norm_to_rt uses rt_values[0]/[-1] as the RT
        // bounds, which must be the sorted first/last cycle RT. For mobility-
        // agnostic data we keep the historical per-spectrum array.
        let rt_values = if alpha_raw_view.has_mobility() {
            rt_index.rt.to_owned()
        } else {
            alpha_raw_view.spectrum_rt.to_owned()
        };

        DIAData {
            rt_index,
            quadrupole_observations,
            rt_values,
            cycle: alpha_raw_view.cycle.to_owned(),
            num_scans: alpha_raw_view.num_scans,
            has_mobility: alpha_raw_view.has_mobility(),
            mobility_per_scan: Vec::new(),
            ms1: None,
        }
    }

    /// Single-phase parallel observation building using rayon
    /// Iterates over all delta_scan_idx values and builds each observation in parallel
    fn build_observations_rayon_parallel(
        alpha_raw_view: &AlphaRawView,
    ) -> Vec<QuadrupoleObservation> {
        use rayon::prelude::*;

        // Find the maximum delta_scan_idx (around 300)
        let max_delta_scan_idx = alpha_raw_view
            .spectrum_delta_scan_idx
            .iter()
            .max()
            .copied()
            .unwrap_or(0);

        // Build observations in parallel for each delta_scan_idx
        let observations: Vec<QuadrupoleObservation> = (0..=max_delta_scan_idx)
            .into_par_iter()
            .map(|delta_scan_idx| {
                Self::build_single_observation_rayon(alpha_raw_view, delta_scan_idx)
            })
            .collect();

        observations
    }

    /// Build a single observation for a given delta_scan_idx using rayon for internal parallelization
    fn build_single_observation_rayon(
        alpha_raw_view: &AlphaRawView,
        target_delta_scan_idx: i64,
    ) -> QuadrupoleObservation {
        let mz_index = MZIndex::global();
        let has_mobility = alpha_raw_view.has_mobility();
        // 2.1: Get all spectra with this delta_scan_idx and build list with spectra_idx
        let matching_spectra: Vec<usize> = alpha_raw_view
            .spectrum_delta_scan_idx
            .iter()
            .enumerate()
            .filter_map(|(spectrum_idx, &delta_scan_idx)| {
                if delta_scan_idx == target_delta_scan_idx {
                    Some(spectrum_idx)
                } else {
                    None
                }
            })
            .collect();

        if matching_spectra.is_empty() {
            // Return empty observation for missing delta_scan_idx
            let mut empty_obs = if has_mobility {
                QuadrupoleObservation::new_with_capacity_im(
                    [0.0, 0.0],
                    0,
                    alpha_raw_view.num_scans,
                    mz_index.len(),
                    0,
                )
            } else {
                QuadrupoleObservation::new_with_capacity([0.0, 0.0], 0, mz_index.len(), 0)
            };
            // Finalize all slices to ensure slice_starts has correct length
            for _ in 0..mz_index.len() {
                empty_obs.finalize_slice();
            }
            return empty_obs;
        }

        // Extract isolation window and count unique cycles
        let first_spectrum_idx = matching_spectra[0];
        let isolation_lower = alpha_raw_view.isolation_lower_mz[first_spectrum_idx];
        let isolation_upper = alpha_raw_view.isolation_upper_mz[first_spectrum_idx];

        let unique_cycles: std::collections::HashSet<_> = matching_spectra
            .iter()
            .map(|&spectrum_idx| alpha_raw_view.spectrum_cycle_idx[spectrum_idx])
            .collect();
        let num_cycles = unique_cycles.len();

        // 2.2: Sort peaks and accumulate. For IM data we carry a scan index too.
        // tuple = (mz_idx, cycle_idx, scan_idx, intensity)
        let mut all_peaks: Vec<(usize, u16, u16, f32)> = Vec::new();

        // Collect all peaks from matching spectra
        for &spectrum_idx in &matching_spectra {
            let cycle_idx = alpha_raw_view.spectrum_cycle_idx[spectrum_idx] as u16;
            let peak_start = alpha_raw_view.spectrum_peak_start_idx[spectrum_idx] as usize;
            let peak_stop = alpha_raw_view.spectrum_peak_stop_idx[spectrum_idx] as usize;

            for peak_idx in peak_start..peak_stop {
                let mz = alpha_raw_view.peak_mz[peak_idx];
                let intensity = alpha_raw_view.peak_intensity[peak_idx];
                let mz_idx = mz_index.find_closest_index(mz);
                let scan_idx = match &alpha_raw_view.peak_scan_idx {
                    Some(scan_arr) => scan_arr[peak_idx] as u16,
                    None => 0u16,
                };
                all_peaks.push((mz_idx, cycle_idx, scan_idx, intensity));
            }
        }

        // Sort peaks by (mz_idx, cycle_idx) to preserve temporal order within each mz_idx
        // This is critical for binary search correctness in fill_xic_slice.
        // (scan_idx is a tie-breaker and does not affect the cycle-sorted invariant.)
        all_peaks.sort_by_key(|(mz_idx, cycle_idx, scan_idx, _)| (*mz_idx, *cycle_idx, *scan_idx));

        // Build observation with sorted peaks
        let mut obs = if has_mobility {
            QuadrupoleObservation::new_with_capacity_im(
                [isolation_lower, isolation_upper],
                num_cycles,
                alpha_raw_view.num_scans,
                mz_index.len(),
                all_peaks.len(),
            )
        } else {
            QuadrupoleObservation::new_with_capacity(
                [isolation_lower, isolation_upper],
                num_cycles,
                mz_index.len(),
                all_peaks.len(),
            )
        };

        let mut current_mz_idx = 0;
        let mut peak_idx = 0;

        while current_mz_idx < mz_index.len() {
            // Add all peaks for current mz_idx
            while peak_idx < all_peaks.len() && all_peaks[peak_idx].0 == current_mz_idx {
                let (_, cycle_idx, scan_idx, intensity) = all_peaks[peak_idx];
                if has_mobility {
                    obs.add_peak_data_im(cycle_idx, scan_idx, intensity);
                } else {
                    obs.add_peak_data(cycle_idx, intensity);
                }
                peak_idx += 1;
            }

            obs.finalize_slice();
            current_mz_idx += 1;
        }

        obs
    }
}

#[cfg(test)]
mod tests;
