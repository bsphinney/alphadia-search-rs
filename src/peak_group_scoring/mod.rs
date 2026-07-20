use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{
    Candidate, CandidateCollection, CandidateFeature, CandidateFeatureCollection,
};
use crate::constants::{FragmentType, C13_C12};
use crate::dense_xic_observation::{DenseXICMZObservation, DenseXICObservation};
use crate::dia_data::DIAData;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, calculate_dot_product, calculate_fwhm_rt, calculate_hyperscore,
    calculate_matched_frag_fraction, calculate_spectral_angle, calculate_spectral_entropy_similarity,
    calculate_weighted_spectral_entropy_similarity,
    calculate_hyperscore_inverse_mass_error, calculate_longest_ion_series, correlation_axis_0,
    filter_non_zero, intensity_ion_series, iso_pattern_metrics, median_axis_0, normalize_profiles,
    theoretical_isotope_envelope, Ms1IsotopeFeatures,
};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;
use crate::utils::{
    calculate_fragment_mz_and_errors, calculate_median, calculate_std,
    calculate_weighted_mean_absolute_error, calculate_weighted_mean_signed_error,
    count_values_above, create_ranked_mask,
};
use crate::SpecLibFlat;
use numpy::ndarray::Axis;
use std::sync::atomic::{AtomicU64, Ordering};

// Global counters for the fragment-interference co-elution filter (fix #1).
// Reset at the start of each score() call; reported at the end. Relaxed is fine
// (monotone increment, printed after the rayon join = happens-before).
static FRAG_MATCHED: AtomicU64 = AtomicU64::new(0);
static FRAG_EXCLUDED: AtomicU64 = AtomicU64::new(0);

pub mod parameters;
pub mod tests;
pub mod utils;
pub use parameters::ScoringParameters;

#[pyclass]
#[allow(dead_code)]
pub struct PeakGroupScoring {
    params: ScoringParameters,
}

#[pymethods]
impl PeakGroupScoring {
    #[new]
    pub fn new(params: ScoringParameters) -> Self {
        Self { params }
    }

    pub fn score(
        &self,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> CandidateFeatureCollection {
        self.score_generic(dia_data, lib, candidates)
    }
}

impl PeakGroupScoring {
    /// Generic scoring function that works with any type implementing DIADataTrait
    fn score_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> CandidateFeatureCollection {
        let start_time = Instant::now();

        // Reset the fragment-interference counters for this pass.
        FRAG_MATCHED.store(0, Ordering::Relaxed);
        FRAG_EXCLUDED.store(0, Ordering::Relaxed);

        // Read the interference-cleaning env config ONCE per pass (not per candidate —
        // there are tens of millions of candidates, and env::var is a syscall+alloc).
        let coelut_thresh: f32 = std::env::var("COELUT_THRESH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1.0);
        let ms1_clean: bool = std::env::var("MS1_CLEAN").ok().as_deref() == Some("1");
        let ms1_clean_pow: f32 = std::env::var("MS1_CLEAN_POW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        // Parallel iteration over candidates to score each one
        let scored_candidates: Vec<CandidateFeature> = candidates
            .par_iter()
            .filter_map(|candidate| {
                // Find precursor by idx (not array position)
                match lib.get_precursor_by_idx_filtered(
                    candidate.precursor_idx,
                    true, // Always filter non-zero intensities for scoring
                    true, // Filter Y1 ions by default
                    self.params.top_k_fragments,
                ) {
                    Some(precursor) => self.score_candidate_generic(
                        dia_data,
                        lib,
                        &precursor,
                        candidate,
                        coelut_thresh,
                        ms1_clean,
                        ms1_clean_pow,
                    ),
                    None => {
                        eprintln!(
                            "Warning: Candidate precursor_idx {} not found in library. Skipping.",
                            candidate.precursor_idx
                        );
                        None
                    }
                }
            })
            .collect();

        // Create collection from Vec
        let feature_collection = CandidateFeatureCollection::from_vec(scored_candidates);

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);

        let candidates_per_second = candidates.len() as f32 / duration.as_secs_f32();
        println!(
            "Scored {} candidates at {:.2} candidates/second",
            candidates.len(),
            candidates_per_second
        );

        // Fragment-interference filter report (fix #1). % of matched product ions
        // excluded for non-co-elution vs the peptide's consensus elution profile.
        let fm = FRAG_MATCHED.load(Ordering::Relaxed);
        let fe = FRAG_EXCLUDED.load(Ordering::Relaxed);
        if fe > 0 {
            println!(
                "FRAG_INTERF: excluded {}/{} matched fragments ({:.1}%) below co-elution threshold",
                fe,
                fm,
                100.0 * fe as f32 / fm.max(1) as f32
            );
        }

        feature_collection
    }

    /// Generic candidate scoring function that works with any type implementing DIADataTrait
    fn score_candidate_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        precursor: &Precursor,
        candidate: &Candidate,
        coelut_thresh: f32,
        ms1_clean: bool,
        ms1_clean_pow: f32,
    ) -> Option<CandidateFeature> {
        // Scoring implementation for individual candidate will be added here
        // For now, return the original score

        let cycle_start_idx = candidate.cycle_start;
        let cycle_stop_idx = candidate.cycle_stop;
        let mass_tolerance = self.params.mass_tolerance;

        // Ion-mobility window for dia-PASEF: restrict extraction to the candidate's
        // mobility band. For mobility-agnostic data (or when the candidate carries
        // no usable scan window) we extract the full scan range, which the
        // mobility-windowed fillers treat as a no-op.
        let (scan_start, scan_stop) = if dia_data.has_mobility() {
            if candidate.scan_stop > candidate.scan_start {
                (candidate.scan_start, candidate.scan_stop)
            } else {
                // No usable per-candidate window: fall back to the full mobility range.
                (0usize, dia_data.num_scans())
            }
        } else {
            (0usize, 1usize)
        };

        // Ion-mobility features (dia-PASEF): observed 1/K0 at the candidate's
        // mobility apex scan, and the delta vs the library-predicted 1/K0.
        // 0.0 when the data or library lacks mobility (then these are no-ops for FDR).
        let mobility_observed = if dia_data.has_mobility() {
            dia_data.mobility_of_scan(candidate.scan_center)
        } else {
            0.0
        };
        let delta_mobility = if dia_data.has_mobility() && precursor.mobility > 0.0 {
            mobility_observed - precursor.mobility
        } else {
            0.0
        };

        // Create dense XIC and m/z observation using the filtered precursor fragments
        let dense_xic_mz_obs = DenseXICMZObservation::new(
            dia_data,
            precursor.mz,
            cycle_start_idx,
            cycle_stop_idx,
            scan_start,
            scan_stop,
            mass_tolerance,
            &precursor.fragment_mz,
        );

        // Normalize the profiles before calculating median
        let normalized_xic = normalize_profiles(&dense_xic_mz_obs.dense_xic, 1);

        // Filter to only non-zero profiles for median calculation
        let filtered_xic = filter_non_zero(&normalized_xic);

        let median_profile = median_axis_0(&normalized_xic);
        let median_profile_filtered = median_axis_0(&filtered_xic);

        let num_profiles = normalized_xic.shape()[0];
        let num_profiles_filtered = filtered_xic.shape()[0];

        // Calculate sum of median profile
        let median_profile_sum = median_profile.iter().sum::<f32>();
        let median_profile_sum_filtered = median_profile_filtered.iter().sum::<f32>();

        let fwhm_rt = calculate_fwhm_rt(
            &median_profile_filtered,
            cycle_start_idx,
            &dia_data.rt_index().rt,
        );

        // Calculate correlations of each profile with the median profile
        let correlations: Vec<f32> = correlation_axis_0(&median_profile_filtered, &normalized_xic);

        let observation_intensities = dense_xic_mz_obs.dense_xic.sum_axis(Axis(1));

        let intensity_correlations = calculate_correlation_safe(
            observation_intensities.as_slice().unwrap(),
            &precursor.fragment_intensity,
        );

        // all of this part is highly experimental and needs to be refined

        // Calculate feature values (using score as proxy for now)
        let mean_correlation = if !correlations.is_empty() {
            correlations.iter().sum::<f32>() / correlations.len() as f32
        } else {
            0.0
        };

        let median_correlation = calculate_median(&correlations);

        let correlation_std = calculate_std(&correlations);

        let num_over_95 = count_values_above(&correlations, 0.95, None);
        let num_over_90 = count_values_above(&correlations, 0.90, None);
        let num_over_80 = count_values_above(&correlations, 0.80, None);
        let num_over_50 = count_values_above(&correlations, 0.50, None);
        let num_over_0 = count_values_above(&correlations, 0.0, None);

        // Calculate ranked features using masks based on library intensities
        // Create masks selecting specific rank ranges
        let mask_0_5 = create_ranked_mask(&precursor.fragment_intensity, 0, 6); // ranks 0-5 (top 6)
        let mask_6_11 = create_ranked_mask(&precursor.fragment_intensity, 6, 12); // ranks 6-11 (next 6)
        let mask_12_17 = create_ranked_mask(&precursor.fragment_intensity, 12, 18); // ranks 12-17 (next 6)
        let mask_18_23 = create_ranked_mask(&precursor.fragment_intensity, 18, 24); // ranks 18-23 (next 6)

        // Calculate num_over_0 for each rank range
        let num_over_0_rank_0_5 = count_values_above(&correlations, 0.0, Some(&mask_0_5));
        let num_over_0_rank_6_11 = count_values_above(&correlations, 0.0, Some(&mask_6_11));
        let num_over_0_rank_12_17 = count_values_above(&correlations, 0.0, Some(&mask_12_17));
        let num_over_0_rank_18_23 = count_values_above(&correlations, 0.0, Some(&mask_18_23));

        // Calculate num_over_50 for each rank range
        let num_over_50_rank_0_5 = count_values_above(&correlations, 0.50, Some(&mask_0_5));
        let num_over_50_rank_6_11 = count_values_above(&correlations, 0.50, Some(&mask_6_11));
        let num_over_50_rank_12_17 = count_values_above(&correlations, 0.50, Some(&mask_12_17));
        let num_over_50_rank_18_23 = count_values_above(&correlations, 0.50, Some(&mask_18_23));

        let intensity_correlation = intensity_correlations;
        let num_fragments = precursor.fragment_mz.len();
        let num_scans = cycle_stop_idx - cycle_start_idx;

        // ---- FRAGMENT INTERFERENCE REMOVAL (fix #1): per-fragment co-elution filter ----
        // `correlations[i]` (computed above) = corr(consensus elution profile
        // `median_profile_filtered`, fragment_i XIC). Product ions whose XIC does not
        // co-elute with the consensus are interference-contaminated (a different peptide's
        // ion falling in the same wide DIA window). Zero their observed intensity BEFORE the
        // quality features (hyperscore, spectral angle/entropy, matched-fraction, ion series,
        // quant) so the scorer sees only co-eluting signal — the Spectronaut recipe (~41% of
        // fragments excluded). coelut_thresh (from score_generic) < 0 (default) = OFF.
        let obs_raw = observation_intensities.as_slice().unwrap();
        let mut obs_clean: Vec<f32> = obs_raw.to_vec();
        if coelut_thresh >= 0.0 {
            let mut n_match = 0u64;
            let mut n_excl = 0u64;
            for i in 0..obs_clean.len() {
                if obs_clean[i] > 0.0 {
                    n_match += 1;
                    let c = if i < correlations.len() {
                        correlations[i]
                    } else {
                        1.0
                    };
                    if c < coelut_thresh {
                        obs_clean[i] = 0.0;
                        n_excl += 1;
                    }
                }
            }
            FRAG_MATCHED.fetch_add(n_match, Ordering::Relaxed);
            FRAG_EXCLUDED.fetch_add(n_excl, Ordering::Relaxed);
        }
        let matched_mask_intensity: Vec<bool> =
            obs_clean.iter().map(|&x| x > 0.0).collect();
        let observation_intensities_slice = obs_clean.as_slice();

        let hyperscore_intensity_observation = calculate_hyperscore(
            &precursor.fragment_type,
            observation_intensities_slice,
            &matched_mask_intensity,
        );

        let hyperscore_intensity_library = calculate_hyperscore(
            &precursor.fragment_type,
            &precursor.fragment_intensity,
            &matched_mask_intensity,
        );

        // Calculate longest continuous ion series
        let (longest_b_series, longest_y_series) = calculate_longest_ion_series(
            &precursor.fragment_type,
            &precursor.fragment_number,
            &matched_mask_intensity,
        );

        // Calculate fragment m/z and mass errors
        let (_fragment_mz_observed, fragment_mass_errors) = calculate_fragment_mz_and_errors(
            &dense_xic_mz_obs.dense_mz,
            &dense_xic_mz_obs.dense_xic,
            &precursor.fragment_mz,
        );

        // Calculate weighted mean absolute mass error using library intensities
        let weighted_mass_error = calculate_weighted_mean_absolute_error(
            &fragment_mass_errors,
            &precursor.fragment_intensity,
        );

        // C6 fix: also compute the SIGNED weighted mean mass error (no .abs()) for the
        // L2 recalibration polynomial fit in extract_feats.py. The absolute version above
        // is kept unchanged because the classifier still uses it.
        let weighted_mass_error_signed = calculate_weighted_mean_signed_error(
            &fragment_mass_errors,
            &precursor.fragment_intensity,
        );

        // Calculate hyperscore with inverse mass error weighting
        // Use observed intensities (sum across cycles) and exclude zero intensity fragments
        let hyperscore_inverse_mass_error = calculate_hyperscore_inverse_mass_error(
            &precursor.fragment_type,
            observation_intensities_slice,
            &matched_mask_intensity,
            &fragment_mass_errors,
        );

        // Calculate retention time features
        let rt_observed = dia_data.rt_index().rt[candidate.cycle_center];
        let delta_rt = rt_observed - precursor.rt;

        // Calculate intensity scores for b and y series
        let intensity_b_raw = intensity_ion_series(
            &precursor.fragment_type,
            observation_intensities_slice,
            &matched_mask_intensity,
            FragmentType::B,
        );

        let intensity_y_raw = intensity_ion_series(
            &precursor.fragment_type,
            observation_intensities_slice,
            &matched_mask_intensity,
            FragmentType::Y,
        );

        // Apply log10 transformation (add epsilon to avoid log(0))
        const EPSILON: f32 = 1e-8;
        let log10_b_ion_intensity = (intensity_b_raw + EPSILON).log10();
        let log10_y_ion_intensity = (intensity_y_raw + EPSILON).log10();

        // Calculate IDF values for this precursor's fragments
        let idf_values = lib.idf.get_idf(&precursor.fragment_mz_library);

        // Calculate IDF-based scores
        let idf_hyperscore = calculate_hyperscore(
            &precursor.fragment_type,
            &idf_values,
            &matched_mask_intensity,
        );

        let idf_xic_dot_product = calculate_dot_product(&idf_values, &correlations);
        let idf_intensity_dot_product =
            calculate_dot_product(&idf_values, observation_intensities_slice);

        let log_idf_intensity_dot_product = (idf_intensity_dot_product + EPSILON).log10();

        // Create mask for top 6 IDF values
        let mask_top6_idf = create_ranked_mask(&idf_values, 0, 6); // ranks 0-5 (top 6 by IDF)

        // Calculate IDF-based correlation features
        let num_over_0_top6_idf = count_values_above(&correlations, 0.0, Some(&mask_top6_idf));
        let num_over_50_top6_idf = count_values_above(&correlations, 0.50, Some(&mask_top6_idf));

        // ---- MS2 similarity panel (predicted-vs-observed fragment intensities) ----
        // obs = summed observed XIC intensity per library fragment; lib = library
        // predicted intensity. Index-aligned; obs==0 => unmatched fragment.
        let obs_ints = observation_intensities_slice;
        let lib_ints: &[f32] = &precursor.fragment_intensity;
        let spectral_entropy_similarity =
            calculate_spectral_entropy_similarity(obs_ints, lib_ints);
        let weighted_spectral_entropy_similarity =
            calculate_weighted_spectral_entropy_similarity(obs_ints, lib_ints);
        let spectral_angle = calculate_spectral_angle(obs_ints, lib_ints);
        let matched_frag_fraction = calculate_matched_frag_fraction(obs_ints, lib_ints);

        // ---- MS1 + isotopologue co-elution panel (DIA-NN 2020 Suppl Note 1) ----
        // The "best"/reference smoothed elution profile is `median_profile_filtered`
        // (the same smoothed reference the existing MS2 co-elution features use —
        // one definition, per the engine's existing convention). All 14 features
        // default to 0.0; they stay 0.0 when MS1 is absent (old caches) — a no-op
        // for the NN. The MS1 store indexes survey peaks by m/z; we extract the
        // precursor + isotope chromatograms over the SAME RT(cycle) x IM(scan)
        // window as the fragments, so the correlations are directly comparable.
        let mut ms1iso = Ms1IsotopeFeatures::default();
        if let Some(ms1) = dia_data.ms1() {
            let ref_profile = median_profile_filtered.as_slice();
            let base_tol = self.params.mass_tolerance.max(30.0f32);  // MS1 needs wider tol than MS2 fragments (miscalibration)
            let p_mz = precursor.mz;

            // (1) MS1 precursor co-elution at base / 0.45*base / 0.2*base mass acc.
            let extract_ms1 = |tol: f32, mz: f32| -> Vec<f32> {
                ms1.extract_xic(
                    mz,
                    cycle_start_idx,
                    cycle_stop_idx,
                    scan_start,
                    scan_stop,
                    tol,
                )
                .to_vec()
            };
            // APEX-CENTERED MS1 co-elution (#45 fix): correlate ref vs MS1 only over a narrow
            // window (+-APEX_HALF cycles) around the ref profile's apex, where the true co-elution
            // lives. The full candidate window drags in off-peak noise/interference that collapses
            // the correlation (validated offline: full-window median r ~0.16 -> apex-centered ~0.60
            // on underscored dogs; our MS1 XIC matches DIA-NN's at r~0.64, so the signal is real).
            const APEX_HALF: usize = 8;
            let n_ref = ref_profile.len();
            let apex = ref_profile
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let alo = apex.saturating_sub(APEX_HALF);
            let ahi = (apex + APEX_HALF + 1).min(n_ref);
            let corr_apex = |xic: &[f32]| -> f32 {
                if ahi > alo && xic.len() == n_ref {
                    calculate_correlation_safe(&ref_profile[alo..ahi], &xic[alo..ahi])
                } else {
                    calculate_correlation_safe(ref_profile, xic)
                }
            };
            let xic_base = extract_ms1(base_tol, p_mz);
            ms1iso.ms1_coelution_base = corr_apex(&xic_base);
            ms1iso.ms1_coelution_045 = corr_apex(&extract_ms1(base_tol * 0.45, p_mz));
            ms1iso.ms1_coelution_02 = corr_apex(&extract_ms1(base_tol * 0.2, p_mz));

            // (2) Isotopologue MS1 co-elution: precursor + 1/2/3 C13. Needs charge
            // for the m/z spacing C13_C12 / z. Skip (leave 0.0) if charge unknown.
            if precursor.charge > 0 {
                let z = precursor.charge as f32;
                let step = C13_C12 / z;
                ms1iso.iso_coelution_c13_1 = corr_apex(&extract_ms1(base_tol, p_mz + step));
                ms1iso.iso_coelution_c13_2 = corr_apex(&extract_ms1(base_tol, p_mz + 2.0 * step));
                ms1iso.iso_coelution_c13_3 = corr_apex(&extract_ms1(base_tol, p_mz + 3.0 * step));
            }

            // (3) Fragment-level isotopologue features. For each fragment, the +1
            // and -1 C13 shifts are C13_C12 / fragment_charge (library fragments
            // are z=1 here; use the per-fragment charge, default 1). Extract the
            // shifted-fragment MS2 XICs with the same IM-windowed machinery used
            // for the matched fragments, normalize identically, correlate with the
            // reference. Top-6 by library intensity, ordered by descending lib int.
            let n_frag = precursor.fragment_mz.len();
            if n_frag > 0 {
                // build +shift and -shift fragment m/z arrays
                let mut fmz_plus = Vec::with_capacity(n_frag);
                let mut fmz_minus = Vec::with_capacity(n_frag);
                for i in 0..n_frag {
                    let fz = if i < precursor.fragment_charge.len() && precursor.fragment_charge[i] > 0
                    {
                        precursor.fragment_charge[i] as f32
                    } else {
                        1.0
                    };
                    let fstep = C13_C12 / fz;
                    fmz_plus.push(precursor.fragment_mz[i] + fstep);
                    fmz_minus.push(precursor.fragment_mz[i] - fstep);
                }

                let corr_shifted = |fmz_shifted: &[f32]| -> Vec<f32> {
                    let obs = DenseXICObservation::new(
                        dia_data,
                        p_mz,
                        cycle_start_idx,
                        cycle_stop_idx,
                        scan_start,
                        scan_stop,
                        base_tol,
                        fmz_shifted,
                    );
                    let norm = normalize_profiles(&obs.dense_xic, 1);
                    correlation_axis_0(ref_profile, &norm)
                };
                let corr_plus = corr_shifted(&fmz_plus);
                let corr_minus = corr_shifted(&fmz_minus);

                // rank fragments by library intensity (descending); take top 6.
                let mut order: Vec<usize> = (0..n_frag).collect();
                order.sort_by(|&a, &b| {
                    precursor.fragment_intensity[b]
                        .partial_cmp(&precursor.fragment_intensity[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let top6: Vec<usize> = order.into_iter().take(6).collect();

                // +1 C13 fragment-isotopologue sum over top 6
                ms1iso.iso_frag_plus_c13_sum =
                    top6.iter().map(|&i| corr_plus[i]).sum();

                // anti-features: per-fragment -(C13) corr for top 6 (+ their sum)
                let mut anti_sum = 0.0;
                for (k, &i) in top6.iter().enumerate() {
                    if k < 6 {
                        ms1iso.iso_frag_minus_c13[k] = corr_minus[i];
                    }
                    anti_sum += corr_minus[i];
                }
                ms1iso.iso_frag_minus_c13_sum = anti_sum;
            }

            // (4) Isotope-envelope-RATIO panel (A2; AlphaDIA isotope_intensity_correlation).
            // OBSERVED envelope obs[0..3]: sum each per-isotope MS1 XIC over the SAME
            // RT(cycle) x IM(scan) window (the IM-window extraction is our timsTOF edge)
            // at base mass tol. obs[0] is the precursor (monoisotopic) channel; obs[k]
            // is p_mz + k*C13_C12/charge. Zero-fill a channel with no signal (its XIC
            // sum is just 0.0). THEORETICAL envelope theo[0..3] from the averagine model
            // (Senko 1995; see theoretical_isotope_envelope). Then 3 ratio/shape metrics.
            if precursor.charge > 0 {
                let z = precursor.charge as f32;
                let step = C13_C12 / z;
                let mut obs = [0.0f32; 4];
                for (k, obs_k) in obs.iter_mut().enumerate() {
                    let xic = extract_ms1(base_tol, p_mz + (k as f32) * step);
                    *obs_k = xic.iter().sum();
                }
                let theo = theoretical_isotope_envelope(precursor.naa, precursor.charge, p_mz);
                let (corr, sa, residual) = iso_pattern_metrics(&theo, &obs);
                ms1iso.iso_pattern_corr = corr;
                ms1iso.iso_pattern_sa = sa;
                ms1iso.iso_m1_over_m_residual = residual;
            }

            // (5) MS1 signal-MAGNITUDE panel (#51). Absolute abundance terms the
            // linear rollup lacked: integrated area + apex of the monoisotope XIC,
            // and total signal over the M/M+1/M+2 envelope. xic_base is the base-tol
            // monoisotope precursor XIC extracted above (L376).
            ms1iso.ms1_area = xic_base.iter().sum();
            ms1iso.ms1_apex = xic_base.iter().cloned().fold(0.0f32, f32::max);
            ms1iso.ms1_total = if precursor.charge > 0 {
                let z = precursor.charge as f32;
                let step = C13_C12 / z;
                (0..3)
                    .map(|k| extract_ms1(base_tol, p_mz + (k as f32) * step).iter().sum::<f32>())
                    .sum()
            } else {
                ms1iso.ms1_area
            };

            // ---- MS1 INTERFERENCE CLEANING (fix #2) ----
            // ms1_area is the engine's DOMINANT scoring feature, but the wide dia-PASEF
            // isolation window lets other co-eluting precursors contaminate the MS1 signal
            // (Spectronaut flags MS1 interference on ~70% of precursors). A contaminated
            // MS1 XIC (a) does NOT co-elute with this peptide's fragment consensus and/or
            // (b) does NOT match the theoretical isotope envelope. Downweight the MS1
            // magnitude features by that quality so chimeric signal can't inflate a wrong
            // candidate's score; a clean precursor (co-elution~1, iso-match~1) is unchanged.
            // Env MS1_CLEAN=1 enables; unset = raw (byte-identical). MS1_CLEAN_POW (default 1)
            // controls gate sharpness. The gate is driven by ms1_coelution_base — the
            // apex-centered correlation of the MS1 monoisotope XIC with the fragment
            // consensus profile — which cleanly separates real precursors (confident-dog
            // median ~0.84) from contaminated/wrong MS1 (decoy/junk median ~0.05). (The
            // theoretical-isotope-pattern SA is ~0 for everyone on this data, so it is NOT
            // used — multiplying by it would zero out ms1_area for real peptides too.)
            if ms1_clean {
                let pw: f32 = ms1_clean_pow;
                let ce = ms1iso.ms1_coelution_base.clamp(0.0, 1.0);
                let gate = ce.powf(pw);
                ms1iso.ms1_area *= gate;
                ms1iso.ms1_apex *= gate;
                ms1iso.ms1_total *= gate;
            }
        }
        let ms1_arr = ms1iso.as_array();

        // Create and return candidate feature
        let mut __feat = CandidateFeature::new(
            candidate.precursor_idx,
            candidate.rank,
            candidate.score,
            mean_correlation,
            median_correlation,
            correlation_std,
            intensity_correlation,
            num_fragments as f32,
            num_scans as f32,
            num_over_95 as f32,
            num_over_90 as f32,
            num_over_80 as f32,
            num_over_50 as f32,
            num_over_0 as f32,
            num_over_0_rank_0_5 as f32,
            num_over_0_rank_6_11 as f32,
            num_over_0_rank_12_17 as f32,
            num_over_0_rank_18_23 as f32,
            num_over_50_rank_0_5 as f32,
            num_over_50_rank_6_11 as f32,
            num_over_50_rank_12_17 as f32,
            num_over_50_rank_18_23 as f32,
            hyperscore_intensity_observation,
            hyperscore_intensity_library,
            hyperscore_inverse_mass_error,
            rt_observed,
            delta_rt,
            longest_b_series as f32,
            longest_y_series as f32,
            precursor.naa as f32,
            weighted_mass_error,
            weighted_mass_error_signed,
            log10_b_ion_intensity,
            log10_y_ion_intensity,
            fwhm_rt,
            idf_hyperscore,
            idf_xic_dot_product,
            log_idf_intensity_dot_product,
            median_profile_sum,
            median_profile_sum_filtered,
            num_profiles as f32,
            num_profiles_filtered as f32,
            num_over_0_top6_idf as f32,
            num_over_50_top6_idf as f32,
            mobility_observed,
            delta_mobility,
            spectral_entropy_similarity,
            weighted_spectral_entropy_similarity,
            spectral_angle,
            matched_frag_fraction,
            ms1_arr[0],
            ms1_arr[1],
            ms1_arr[2],
            ms1_arr[3],
            ms1_arr[4],
            ms1_arr[5],
            ms1_arr[6],
            ms1_arr[7],
            ms1_arr[8],
            ms1_arr[9],
            ms1_arr[10],
            ms1_arr[11],
            ms1_arr[12],
            ms1_arr[13],
            ms1_arr[14],
            ms1_arr[15],
            ms1_arr[16],
            ms1_arr[17],
            ms1_arr[18],
            ms1_arr[19],
        );
        // empirical-refinement: carry the observed per-fragment intensities (aligned to library
        // fragment order) out of scoring so the refined-library builder can rebuild observed spectra.
        __feat.observed_frag_intensities = observation_intensities.to_vec();
        Some(__feat)
    }
}
