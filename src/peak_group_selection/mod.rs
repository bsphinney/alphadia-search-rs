use numpy::ndarray::Array1;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::min;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::convolution::convolution;
use crate::dense_xic_observation::DenseXICObservation;
use crate::dia_data::DIAData;
use crate::kernel::GaussianKernel;
use crate::precursor::Precursor;
use crate::score::axis_log_dot_product;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use crate::SpecLibFlat;

pub mod parameters;
pub use parameters::SelectionParameters;

/// Finds local maxima in a 1D array.
/// A local maximum is defined as a point that is higher than the 2 points to its left and right. NOT COMPLETELY
/// Returns a tuple of two vectors: (indices, values) sorted by value in descending order.
fn find_local_maxima(array: &Array1<f32>, offset: usize) -> (Vec<usize>, Vec<f32>) {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let len = array.len();

    // Need at least 5 points to find a local maximum with 2 points on each side
    if len < 5 {
        return (indices, values);
    }

    // Check each point (except the first and last 2) for local maxima
    for i in 2..len - 2 {
        if array[i - 2] < array[i - 1]
            && array[i - 1] < array[i]
            && array[i] > array[i + 1]
            && array[i + 1] > array[i + 2]
        {
            indices.push(i + offset);
            values.push(array[i]);
        }
    }

    // Sort by value in descending order
    if !values.is_empty() {
        // Create index mapping for sorting
        let mut idx_map: Vec<usize> = (0..values.len()).collect();
        idx_map.sort_by(|&a, &b| {
            values[b]
                .partial_cmp(&values[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Reorder both vectors using the sorted mapping
        let sorted_indices: Vec<usize> = idx_map.iter().map(|&i| indices[i]).collect();
        let sorted_values: Vec<f32> = idx_map.iter().map(|&i| values[i]).collect();

        indices = sorted_indices;
        values = sorted_values;
    }

    (indices, values)
}

#[pyclass]
pub struct PeakGroupSelection {
    kernel: GaussianKernel,
    params: SelectionParameters,
}

#[pymethods]
impl PeakGroupSelection {
    #[new]
    pub fn new(params: SelectionParameters) -> Self {
        Self {
            kernel: GaussianKernel::new(
                params.fwhm_rt,
                1.0, // sigma_scale_rt
                params.kernel_size,
                1.0, // rt_resolution
            ),
            params,
        }
    }

    pub fn search(&self, dia_data: &DIAData, lib: &SpecLibFlat) -> CandidateCollection {
        if self.params.fragment_index_min_cofrag > 0 {
            let t0 = Instant::now();
            let fi = crate::fragment_index::FragmentIndex::build(lib);
            let keep = fi.shortlist(
                dia_data,
                self.params.rt_tolerance,
                self.params.im_tolerance,
                self.params.kernel_size,
                self.params.fragment_index_min_cofrag,
            );
            let nkeep = keep.iter().filter(|&&b| b).count();
            println!(
                "FragmentIndex shortlist: {}/{} precursors kept ({:.1}%) in {:.1}s",
                nkeep, keep.len(), 100.0 * nkeep as f32 / keep.len().max(1) as f32, t0.elapsed().as_secs_f32()
            );
            if let Ok(path) = std::env::var("FRAG_INDEX_DUMP") {
                let bytes: Vec<u8> = keep.iter().map(|&b| b as u8).collect();
                let _ = std::fs::write(&path, &bytes);
                println!("FragmentIndex: dumped keep mask ({} bytes) to {}", bytes.len(), path);
            }
            if std::env::var("FRAG_INDEX_SHORTLIST_ONLY").is_ok() {
                println!("FRAG_INDEX_SHORTLIST_ONLY set -> returning empty candidates (shortlist-only test)");
                return CandidateCollection::default();
            }
            self.search_generic(dia_data, lib, Some(&keep))
        } else {
            self.search_generic(dia_data, lib, None)
        }
    }
}

impl PeakGroupSelection {
    /// Generic search function that works with any type implementing DIADataTrait
    fn search_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        keep: Option<&[bool]>,
    ) -> CandidateCollection {
        let start_time = Instant::now();
        // Count precursors that survive selection (produced >=1 candidate) vs those
        // pruned before extraction (RT window empty, or — with the IM gate on —
        // predicted 1/K0 outside every scan). This is the candidate-reduction number
        // that pruning is meant to cut; reported below alongside throughput.
        let pruned = std::sync::atomic::AtomicUsize::new(0);
        // Parallel iteration over precursor indices with filter_map to collect candidates
        let candidates: Vec<Candidate> = (0..lib.num_precursors())
            .into_par_iter()
            .filter_map(|i| {
                if let Some(k) = keep {
                    if !k[i] {
                        pruned.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return None;
                    }
                }
                let precursor = lib.get_precursor_filtered(
                    i,
                    true, // Always filter non-zero intensities for scoring
                    true, // Filter Y1 ions by default
                    self.params.top_k_fragments,
                );
                let out = self.search_precursor_generic(
                    dia_data,
                    &precursor,
                    self.params.mass_tolerance,
                    self.params.rt_tolerance,
                    self.params.candidate_count,
                );
                if out.is_none() {
                    pruned.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                out
            })
            .flatten()
            .collect();
        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let n_pre = lib.num_precursors();
        let n_pruned = pruned.load(std::sync::atomic::Ordering::Relaxed);
        let precursors_per_second = n_pre as f32 / duration.as_secs_f32();
        println!("Precursors per second: {precursors_per_second:?}");
        println!(
            "Pruned {}/{} precursors before extraction ({:.1}%) [im_tol={}]",
            n_pruned,
            n_pre,
            100.0 * n_pruned as f32 / n_pre.max(1) as f32,
            self.params.im_tolerance
        );
        println!("Found {} candidates", candidates.len());

        CandidateCollection::from_vec(candidates)
    }

    /// Generic precursor search function that works with any type implementing DIADataTrait
    fn search_precursor_generic<T: DIADataTrait>(
        &self,
        dia_data: &T,
        precursor: &Precursor,
        mass_tolerance: f32,
        rt_tolerance: f32,
        candidate_count: usize,
    ) -> Option<Vec<Candidate>> {
        let (cycle_start_idx, cycle_stop_idx) = dia_data.rt_index().get_cycle_idx_limits(
            precursor.rt,
            rt_tolerance,
            self.params.kernel_size,
        );

        // Validate cycle range
        if cycle_stop_idx <= cycle_start_idx {
            return None;
        }

        // Selection scan (ion-mobility) window. Historically selection integrated
        // over the FULL mobility range (0..num_scans) so no precursor is missed.
        // With `im_tolerance > 0` (dia-PASEF), we PRUNE selection to only the scans
        // whose 1/K0 is within `precursor.mobility ± im_tolerance` — this is a proper
        // ion-mobility candidate gate (the audit noted delta_mobility was only a
        // scoring feature, never a gate). It both cuts extraction work and removes
        // co-eluting-but-wrong-mobility decoy competition. A precursor whose predicted
        // 1/K0 has no matching scan is pruned entirely (returns None below).
        let (sel_scan_start, sel_scan_stop) = if dia_data.has_mobility() {
            if self.params.im_tolerance > 0.0 && precursor.mobility > 0.0 {
                match Self::mobility_scan_band(dia_data, precursor.mobility, self.params.im_tolerance)
                {
                    Some(band) => band,
                    None => return None, // predicted 1/K0 outside every scan's mobility -> prune
                }
            } else {
                (0, dia_data.num_scans())
            }
        } else {
            (0, 1)
        };
        // Fragment-presence pre-filter (fragment-index screen): require at least
        // `min_matched_fragments` of the precursor's library fragments to have ANY
        // observed signal in this RT/IM window BEFORE paying for dense-XIC extraction +
        // convolution + scoring. Prunes signal-free (noise/decoy/foreign) precursors.
        // 0 => OFF (byte-identical to legacy).
        if self.params.min_matched_fragments > 0 {
            let valid_obs = dia_data.get_valid_observations(precursor.mz);
            let mut matched = 0usize;
            for &f_mz in precursor.fragment_mz.iter() {
                let mut present = false;
                for &obs_idx in &valid_obs {
                    if dia_data.quadrupole_observations()[obs_idx].fragment_has_signal_windowed(
                        dia_data.mz_index(),
                        cycle_start_idx,
                        cycle_stop_idx,
                        sel_scan_start,
                        sel_scan_stop,
                        mass_tolerance,
                        f_mz,
                    ) {
                        present = true;
                        break;
                    }
                }
                if present {
                    matched += 1;
                    if matched >= self.params.min_matched_fragments {
                        break;
                    }
                }
            }
            if matched < self.params.min_matched_fragments {
                return None;
            }
        }
        let dense_xic_obs = DenseXICObservation::new(
            dia_data,
            precursor.mz,
            cycle_start_idx,
            cycle_stop_idx,
            sel_scan_start,
            sel_scan_stop,
            mass_tolerance,
            &precursor.fragment_mz,
        );

        let convolved_xic = convolution(&self.kernel, &dense_xic_obs.dense_xic);

        let score = axis_log_dot_product(&convolved_xic, &precursor.fragment_intensity);

        let (local_maxima_indices, local_maxima_values) =
            find_local_maxima(&score, cycle_start_idx);

        // Take top maxima (they're already sorted by value in descending order)
        let max_count = std::cmp::min(candidate_count, local_maxima_indices.len());

        let mut candidates = Vec::new();

        for i in 0..max_count {
            let cycle_center_idx = local_maxima_indices[i];
            let score = local_maxima_values[i];

            // C1 fix: cycle_center_idx and peak_length are usize; `max(0, a - b)` underflows
            // (wraps to a huge usize) on early-eluting precursors before max(0,..) can clamp.
            // saturating_sub clamps the subtraction at 0 correctly.
            let cycle_start_idx = cycle_center_idx.saturating_sub(self.params.peak_length);
            let cycle_stop_idx = min(
                cycle_center_idx + self.params.peak_length + 1,
                dia_data.rt_index().len(),
            );

            let mut candidate = Candidate::new(
                precursor.precursor_idx,
                i,
                score,
                cycle_start_idx,
                cycle_center_idx,
                cycle_stop_idx,
            );

            // dia-PASEF: determine the candidate's ion-mobility window by finding
            // the mobility apex over this RT window and keeping the scans whose
            // summed fragment intensity is above a fraction of the apex.
            if dia_data.has_mobility() {
                let (scan_start, scan_stop) = Self::mobility_window(
                    dia_data,
                    precursor,
                    cycle_start_idx,
                    cycle_stop_idx,
                    mass_tolerance,
                );
                candidate.scan_start = scan_start;
                candidate.scan_center = (scan_start + scan_stop) / 2;
                candidate.scan_stop = scan_stop;
            }

            candidates.push(candidate);
        }

        Some(candidates)
    }

    /// Ion-mobility candidate gate: return the contiguous scan window
    /// `[scan_lo, scan_hi)` whose per-scan 1/K0 lies within
    /// `predicted_mobility ± im_tolerance`, or `None` if NO scan qualifies (the
    /// predicted mobility is outside the instrument's mobility range for this run,
    /// so the precursor is pruned before any extraction). Scans of a timsTOF frame
    /// are monotonic in 1/K0, but we do not rely on that: we take the min/max
    /// qualifying scan index so the returned window is a superset of all in-band
    /// scans regardless of ordering direction.
    fn mobility_scan_band<T: DIADataTrait>(
        dia_data: &T,
        predicted_mobility: f32,
        im_tolerance: f32,
    ) -> Option<(usize, usize)> {
        let num_scans = dia_data.num_scans();
        if num_scans <= 1 {
            return Some((0, num_scans.max(1)));
        }
        let lo_k0 = predicted_mobility - im_tolerance;
        let hi_k0 = predicted_mobility + im_tolerance;
        let mut lo: Option<usize> = None;
        let mut hi: usize = 0;
        for s in 0..num_scans {
            let k0 = dia_data.mobility_of_scan(s);
            // mobility_of_scan returns 0.0 when a per-scan 1/K0 is unavailable; a
            // 0.0 sentinel is not a real mobility, so skip it (never gate on it).
            if k0 <= 0.0 {
                continue;
            }
            if k0 >= lo_k0 && k0 <= hi_k0 {
                if lo.is_none() {
                    lo = Some(s);
                }
                hi = s;
            }
        }
        lo.map(|l| (l, hi + 1))
    }

    /// Determine an ion-mobility (scan) window for a candidate by summing the
    /// fragment XIC into a per-scan profile over the candidate's RT window and
    /// keeping the contiguous band around the apex above `MOBILITY_REL_CUTOFF`.
    fn mobility_window<T: DIADataTrait>(
        dia_data: &T,
        precursor: &Precursor,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
    ) -> (usize, usize) {
        use numpy::ndarray::Array2;
        const MOBILITY_REL_CUTOFF: f32 = 0.10;
        let num_scans = dia_data.num_scans();
        let n_cycles = cycle_stop_idx - cycle_start_idx;
        if num_scans <= 1 || n_cycles == 0 {
            return (0, num_scans.max(1));
        }

        // Accumulate a per-scan intensity profile summed over fragments and cycles.
        let mut scan_profile = vec![0.0f32; num_scans];
        let valid_obs = dia_data.get_valid_observations(precursor.mz);
        let mut dense = Array2::<f32>::zeros((num_scans, n_cycles));
        for &obs_idx in &valid_obs {
            let obs = &dia_data.quadrupole_observations()[obs_idx];
            for &f_mz in precursor.fragment_mz.iter() {
                // fill_dense_xic_3d is additive; reuse one buffer per fragment.
                dense.fill(0.0);
                obs.fill_dense_xic_3d_trait(
                    dia_data.mz_index(),
                    &mut dense.view_mut(),
                    cycle_start_idx,
                    cycle_stop_idx,
                    0,
                    num_scans,
                    mass_tolerance,
                    f_mz,
                );
                for s in 0..num_scans {
                    let mut row_sum = 0.0f32;
                    for c in 0..n_cycles {
                        row_sum += dense[[s, c]];
                    }
                    scan_profile[s] += row_sum;
                }
            }
        }

        // Find apex scan
        let mut apex = 0usize;
        let mut apex_val = 0.0f32;
        for (s, &v) in scan_profile.iter().enumerate() {
            if v > apex_val {
                apex_val = v;
                apex = s;
            }
        }
        if apex_val <= 0.0 {
            // no signal: fall back to full range so we don't drop the candidate
            return (0, num_scans);
        }
        let cutoff = apex_val * MOBILITY_REL_CUTOFF;
        // expand left/right while above cutoff
        let mut lo = apex;
        while lo > 0 && scan_profile[lo - 1] >= cutoff {
            lo -= 1;
        }
        let mut hi = apex;
        while hi + 1 < num_scans && scan_profile[hi + 1] >= cutoff {
            hi += 1;
        }
        (lo, hi + 1)
    }
}

#[cfg(test)]
mod tests;
