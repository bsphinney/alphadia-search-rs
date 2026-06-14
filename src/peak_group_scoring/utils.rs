use crate::constants::FragmentType;
use numpy::ndarray::{Array1, Array2};
use std::f32;

/// Filter out rows that contain only zeros
/// Returns a new Array2 with only non-zero rows
pub fn filter_non_zero(array: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = array.dim();

    // Find indices of rows that have at least one non-zero value
    let non_zero_rows: Vec<usize> = (0..rows)
        .filter(|&i| {
            let row = array.row(i);
            row.iter().any(|&val| val != 0.0)
        })
        .collect();

    if non_zero_rows.is_empty() {
        // Return empty array with same number of columns if all rows are zero
        Array2::zeros((0, cols))
    } else {
        // Create new array with only non-zero rows
        let mut filtered = Array2::zeros((non_zero_rows.len(), cols));
        for (new_idx, &old_idx) in non_zero_rows.iter().enumerate() {
            filtered.row_mut(new_idx).assign(&array.row(old_idx));
        }
        filtered
    }
}

/// Calculate the median along axis 0 (first axis) of a 2D array
/// Works with any input array - caller can filter using filter_non_zero if needed
/// Returns zeros for all columns if array has no rows
/// Similar to np.median(array, axis=0) in NumPy
pub fn median_axis_0(array: &Array2<f32>) -> Vec<f32> {
    let (rows, cols) = array.dim();

    // If no rows exist, return zeros
    if rows == 0 {
        return vec![0.0; cols];
    }

    let mut result = Vec::with_capacity(cols);

    for col in 0..cols {
        let mut column_values: Vec<f32> = Vec::with_capacity(rows);
        for row in 0..rows {
            column_values.push(array[[row, col]]);
        }

        // Sort the column values to find median
        column_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if rows % 2 == 0 {
            // Even number of elements: average of two middle values
            let mid = rows / 2;
            (column_values[mid - 1] + column_values[mid]) / 2.0
        } else {
            // Odd number of elements: middle value
            column_values[rows / 2]
        };

        result.push(median);
    }

    result
}

/// Calculate normalized intensity profiles from dense array.
/// Similar to normalize_profiles in Python
pub fn normalize_profiles(intensity_slice: &Array2<f32>, center_dilations: usize) -> Array2<f32> {
    let (rows, cols) = intensity_slice.dim();
    let center_idx = cols / 2;

    // Calculate mean center intensity for each row
    let mut center_intensity = Vec::with_capacity(rows);

    for i in 0..rows {
        let start_idx = center_idx.saturating_sub(center_dilations);
        let end_idx = std::cmp::min(center_idx + center_dilations + 1, cols);

        let mut sum = 0.0;
        let mut count = 0;

        for j in start_idx..end_idx {
            sum += intensity_slice[[i, j]];
            count += 1;
        }

        let mean_intensity = if count > 0 { sum / count as f32 } else { 0.0 };
        center_intensity.push(mean_intensity);
    }

    // Create normalized output array, initialized to zeros
    let mut normalized = Array2::zeros((rows, cols));

    // Only normalize profiles where center intensity > 0
    for i in 0..rows {
        if center_intensity[i] > 0.0 {
            for j in 0..cols {
                normalized[[i, j]] = intensity_slice[[i, j]] / center_intensity[i];
            }
        }
    }

    normalized
}

/// Calculate correlation between median profile and each row of dense_xic
/// Returns zero where no safe correlation can be calculated
pub fn correlation_axis_0(median_profile: &[f32], dense_xic: &Array2<f32>) -> Vec<f32> {
    let (rows, _cols) = dense_xic.dim();
    let mut correlations = Vec::with_capacity(rows);

    for row in 0..rows {
        let row_data: Vec<f32> = dense_xic.row(row).to_vec();
        let correlation = calculate_correlation_safe(median_profile, &row_data);
        correlations.push(correlation);
    }

    correlations
}

/// Calculate correlation between two arrays safely
/// Returns 0.0 if correlation cannot be calculated safely
pub fn calculate_correlation_safe(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    // Check for all zeros or constant values
    let x_sum: f32 = x.iter().sum();
    let y_sum: f32 = y.iter().sum();

    if x_sum == 0.0 || y_sum == 0.0 {
        return 0.0;
    }

    // Check for constant values (zero variance)
    let x_mean = x_sum / x.len() as f32;
    let y_mean = y_sum / y.len() as f32;

    let mut x_variance = 0.0;
    let mut y_variance = 0.0;
    let mut covariance = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;

        x_variance += x_diff * x_diff;
        y_variance += y_diff * y_diff;
        covariance += x_diff * y_diff;
    }

    // Check for zero variance (constant values)
    if x_variance == 0.0 || y_variance == 0.0 {
        return 0.0;
    }

    // Calculate correlation coefficient
    let correlation = covariance / (f32::sqrt(x_variance) * f32::sqrt(y_variance));

    // Check for NaN or infinite values
    if correlation.is_nan() || correlation.is_infinite() {
        panic!("correlation.is_nan() || correlation.is_infinite()");
    }

    // Clamp to valid range [-1, 1]
    correlation.clamp(-1.0, 1.0)
}

/// Calculate correlation between two f32 slices
/// Returns 0.0 if correlation cannot be calculated safely
pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
    calculate_correlation_safe(x, y)
}

/// Calculate hyperscore with optional per-fragment weights
///
/// hyperscore = log(Nb! * Ny! * sum(Ib,i * w_i) * sum(Iy,i * w_i))
/// where:
/// - Nb = number of matched b-ions
/// - Ny = number of matched y-ions
/// - Ib,i = intensities of matched b-ions
/// - Iy,i = intensities of matched y-ions
/// - w_i = optional weight for each fragment (if None, uses 1.0)
///
/// Fragment types are encoded as ASCII values:
/// - b-ion = 98 (ASCII 'b')
/// - y-ion = 121 (ASCII 'y')
///   (other fragment types exist but are not used in hyperscore)
pub fn calculate_hyperscore_weighted(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    weights: Option<&[f32]>,
) -> f32 {
    if fragment_types.len() != fragment_intensities.len()
        || fragment_types.len() != matched_mask.len()
    {
        return 0.0;
    }

    if let Some(w) = weights {
        if w.len() != fragment_types.len() {
            return 0.0;
        }
    }

    let mut n_b = 0u32;
    let mut n_y = 0u32;
    let mut weighted_sum_b = 0.0f32;
    let mut weighted_sum_y = 0.0f32;

    for i in 0..fragment_types.len() {
        if !matched_mask[i] || fragment_intensities[i] == 0.0 {
            continue;
        }

        let weight = weights.map(|w| w[i]).unwrap_or(1.0);
        let weighted_intensity = fragment_intensities[i] * weight;

        match fragment_types[i] {
            FragmentType::B => {
                // b-ion
                n_b += 1;
                weighted_sum_b += weighted_intensity;
            }
            FragmentType::Y => {
                // y-ion
                n_y += 1;
                weighted_sum_y += weighted_intensity;
            }
            _ => {
                // Other fragment types not used in hyperscore
            }
        }
    }

    if n_b == 0 && n_y == 0 {
        return 0.0;
    }

    // Calculate factorial using gamma function: n! = Γ(n+1)
    let factorial_b = if n_b > 0 {
        gamma_ln(n_b as f32 + 1.0)
    } else {
        0.0
    };
    let factorial_y = if n_y > 0 {
        gamma_ln(n_y as f32 + 1.0)
    } else {
        0.0
    };

    // Calculate hyperscore: log(Nb! * Ny! * weighted_sum_b * weighted_sum_y)
    // Don't use .max(0.0) on ln() as it can make valid small values become 0
    let ln_sum_b = if weighted_sum_b > 0.0 {
        weighted_sum_b.ln()
    } else {
        0.0
    };
    let ln_sum_y = if weighted_sum_y > 0.0 {
        weighted_sum_y.ln()
    } else {
        0.0
    };

    let hyperscore = factorial_b + factorial_y + ln_sum_b + ln_sum_y;

    if hyperscore.is_finite() {
        hyperscore
    } else {
        0.0
    }
}

/// Calculate standard hyperscore similar to X! Tandem and MSFragger
///
/// This is a wrapper around calculate_hyperscore_weighted with no weights
pub fn calculate_hyperscore(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
) -> f32 {
    calculate_hyperscore_weighted(fragment_types, fragment_intensities, matched_mask, None)
}

/// Natural logarithm of gamma function using Stirling's approximation
/// For factorial calculation: ln(n!) = ln(Γ(n+1))
/// Always uses approximation as requested, except for special cases
fn gamma_ln(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    if (x - 1.0).abs() < 1e-6 {
        return 0.0; // ln(Γ(1)) = ln(0!) = ln(1) = 0
    }

    if (x - 2.0).abs() < 1e-6 {
        return 0.0; // ln(Γ(2)) = ln(1!) = ln(1) = 0
    }

    // Stirling's approximation: ln(Γ(x)) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π)
    let ln_2pi = 1.837_877_f32;
    (x - 0.5) * x.ln() - x + 0.5 * ln_2pi
}

/// Calculate longest continuous b and y ion series scores
/// Returns (longest_b_series, longest_y_series) based on fragment_number values
/// Handles fragment numbers in any order by sorting internally
pub fn calculate_longest_ion_series(
    fragment_types: &[u8],
    fragment_numbers: &[u8],
    matched_mask: &[bool],
) -> (u8, u8) {
    if fragment_types.len() != matched_mask.len() || fragment_types.len() != fragment_numbers.len()
    {
        return (0, 0);
    }

    // Collect matched b and y ions with their fragment numbers
    let mut b_ions: Vec<u8> = Vec::new();
    let mut y_ions: Vec<u8> = Vec::new();

    for i in 0..fragment_types.len() {
        if matched_mask[i] {
            match fragment_types[i] {
                FragmentType::B => b_ions.push(fragment_numbers[i]),
                FragmentType::Y => y_ions.push(fragment_numbers[i]),
                _ => {}
            }
        }
    }

    // Helper function to find longest continuous sequence
    let find_longest_sequence = |mut ions: Vec<u8>| -> u8 {
        if ions.is_empty() {
            return 0;
        }

        ions.sort_unstable();

        let mut max_length = 1u8;
        let mut current_length = 1u8;

        for i in 1..ions.len() {
            if ions[i] == ions[i - 1] + 1 {
                current_length += 1;
                max_length = max_length.max(current_length);
            } else {
                current_length = 1;
            }
        }

        max_length
    };

    let longest_b = find_longest_sequence(b_ions);
    let longest_y = find_longest_sequence(y_ions);

    (longest_b, longest_y)
}

/// Calculate hyperscore with inverse mass error weighting
///
/// Similar to standard hyperscore but weights each matched fragment by 1/(|mass_error| + 0.1)
/// Excludes fragments with zero observed intensity (sum across all cycles)
///
/// hyperscore = log(Nb! * Ny! * sum(Ib,i * w_i) * sum(Iy,i * w_i))
/// where w_i = 1/(|mass_error_i| + 0.1)
pub fn calculate_hyperscore_inverse_mass_error(
    fragment_types: &[u8],
    fragment_intensities: &[f32], // Observed intensities (sum across cycles)
    matched_mask: &[bool],
    mass_errors: &[f32], // Mass errors in ppm
) -> f32 {
    if fragment_types.len() != mass_errors.len() {
        return 0.0;
    }

    // Calculate inverse mass error weights: 1/(|mass_error| + 0.1)
    let weights: Vec<f32> = mass_errors
        .iter()
        .map(|&error| 1.0 / (error.abs() + 0.1))
        .collect();

    calculate_hyperscore_weighted(
        fragment_types,
        fragment_intensities,
        matched_mask,
        Some(&weights),
    )
}

/// Calculate total intensity for a specific ion series
///
/// Sums all observed intensities for fragments of the specified type
/// that have a matched intensity (intensity > 0 and matched_mask = true)
pub fn intensity_ion_series(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    target_fragment_type: u8,
) -> f32 {
    let n_fragments = fragment_types.len();
    if n_fragments != fragment_intensities.len() || n_fragments != matched_mask.len() {
        return 0.0;
    }

    let mut total_intensity = 0.0;

    for i in 0..n_fragments {
        if matched_mask[i]
            && fragment_intensities[i] > 0.0
            && fragment_types[i] == target_fragment_type
        {
            total_intensity += fragment_intensities[i];
        }
    }

    total_intensity
}

/// Calculate dot product between two slices of equal length
///
/// Returns the sum of element-wise products: sum(a_i * b_i)
/// Returns 0.0 if slices have different lengths or are empty
pub fn calculate_dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Calculate Full Width at Half Maximum (FWHM) for retention time from an XIC profile
///
/// Finds the maximum peak in the XIC slice and calculates the FWHM by finding points
/// where the intensity is half of the maximum. The slice should be centered at the maximum
/// and have an odd number of elements.
///
/// Parameters:
/// - xic_profile: Intensity profile (median profile across fragments)
/// - cycle_start_idx: Starting cycle index in the RT array
/// - cycle_stop_idx: Ending cycle index in the RT array (exclusive)
/// - rt_values: Array of retention time values
///
/// Returns:
/// - FWHM in retention time units, or 0.0 if cannot be calculated
pub fn calculate_fwhm_rt(
    xic_profile: &[f32],
    cycle_start_idx: usize,
    rt_values: &Array1<f32>,
) -> f32 {
    if xic_profile.is_empty() {
        return 0.0;
    }

    let half_size = xic_profile.len() / 2;
    let center_intensity = xic_profile[half_size];

    for i in 0..half_size {
        let mean_intensity = (xic_profile[half_size - i] + xic_profile[half_size + i]) / 2.0;

        if mean_intensity <= center_intensity / 2.0 {
            let left_rt = rt_values[cycle_start_idx + half_size - i];
            let right_rt = rt_values[cycle_start_idx + half_size + i];
            return right_rt - left_rt;
        }
    }

    0.0
}

// ============================================================================
// Spectral-entropy + MS2 similarity panel (predicted-vs-observed fragment ints)
// Added 2026-06-13 to recover the "scored-out" precursor gap (RUST_DIA_ENGINE §13).
//
// All operate on the two already-matched, index-aligned per-fragment intensity
// vectors the scorer already computes: `obs` = summed observed XIC intensity per
// library fragment, `lib` = library/predicted fragment intensity. A fragment with
// obs==0 is unmatched. These add NO new extraction — pure post-match scalars.
// ============================================================================

/// L1-normalize a slice into a probability vector. Returns None if total<=0.
fn l1_normalize(v: &[f32]) -> Option<Vec<f64>> {
    let total: f64 = v.iter().map(|&x| x.max(0.0) as f64).sum();
    if total <= 0.0 {
        return None;
    }
    Some(v.iter().map(|&x| (x.max(0.0) as f64) / total).collect())
}

/// Shannon entropy (nats) of an L1-normalized probability vector.
fn shannon_entropy(p: &[f64]) -> f64 {
    let mut h = 0.0;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * pi.ln();
        }
    }
    h
}

/// Spectral entropy similarity (Li et al., Nat Methods 2021) between observed and
/// library fragment intensities. Defined as
///   S = 1 - (2*H(p_merged) - H(p_obs) - H(p_lib)) / ln(4)
/// where p_merged is the L1-normalized element-wise mean of the two normalized
/// spectra. Returns a value in [0,1]; 1 = identical, 0 = maximally divergent.
/// Returns 0.0 if either spectrum has no signal.
pub fn calculate_spectral_entropy_similarity(obs: &[f32], lib: &[f32]) -> f32 {
    if obs.len() != lib.len() || obs.is_empty() {
        return 0.0;
    }
    let p = match l1_normalize(obs) {
        Some(v) => v,
        None => return 0.0,
    };
    let q = match l1_normalize(lib) {
        Some(v) => v,
        None => return 0.0,
    };
    let merged: Vec<f64> = p.iter().zip(q.iter()).map(|(&a, &b)| 0.5 * (a + b)).collect();
    let h_merged = shannon_entropy(&merged);
    let h_p = shannon_entropy(&p);
    let h_q = shannon_entropy(&q);
    // Jensen-Shannon entropy distance, normalized to [0,1] by ln(4) = 2*ln(2).
    let sim = 1.0 - ((2.0 * h_merged - h_p - h_q) / (2.0 * std::f64::consts::LN_2));
    sim.clamp(0.0, 1.0) as f32
}

/// Weighted spectral entropy similarity (Li et al., Nat Methods 2021). Each
/// spectrum is first entropy-weighted: if its Shannon entropy H < 3 nats, every
/// intensity is raised to the power w = 0.25 + 0.25*H (down-weighting noisy/low-
/// entropy spectra), then re-normalized, before computing the entropy similarity.
/// This is the MSBooster "weighted spectral entropy" variant.
pub fn calculate_weighted_spectral_entropy_similarity(obs: &[f32], lib: &[f32]) -> f32 {
    if obs.len() != lib.len() || obs.is_empty() {
        return 0.0;
    }
    let reweight = |v: &[f32]| -> Option<Vec<f32>> {
        let p = l1_normalize(v)?;
        let h = shannon_entropy(&p);
        let w = if h < 3.0 { 0.25 + 0.25 * h } else { 1.0 };
        let pw: Vec<f32> = p.iter().map(|&x| (x.powf(w)) as f32).collect();
        Some(pw)
    };
    let ow = match reweight(obs) {
        Some(v) => v,
        None => return 0.0,
    };
    let lw = match reweight(lib) {
        Some(v) => v,
        None => return 0.0,
    };
    calculate_spectral_entropy_similarity(&ow, &lw)
}

/// Normalized spectral contrast angle (Prosit / MSBooster). SA = 1 - 2*acos(cos)/pi
/// where cos is the cosine similarity of the two L1-normalized intensity vectors.
/// Returns [0,1]; 1 = identical direction.
pub fn calculate_spectral_angle(obs: &[f32], lib: &[f32]) -> f32 {
    if obs.len() != lib.len() || obs.is_empty() {
        return 0.0;
    }
    let p = match l1_normalize(obs) {
        Some(v) => v,
        None => return 0.0,
    };
    let q = match l1_normalize(lib) {
        Some(v) => v,
        None => return 0.0,
    };
    let mut dot = 0.0;
    let mut np = 0.0;
    let mut nq = 0.0;
    for i in 0..p.len() {
        dot += p[i] * q[i];
        np += p[i] * p[i];
        nq += q[i] * q[i];
    }
    if np <= 0.0 || nq <= 0.0 {
        return 0.0;
    }
    let cos = (dot / (np.sqrt() * nq.sqrt())).clamp(-1.0, 1.0);
    let sa = 1.0 - 2.0 * cos.acos() / std::f64::consts::PI;
    sa.clamp(0.0, 1.0) as f32
}

/// Fraction of library fragments (intensity>0) that have an observed match (obs>0).
/// Captures matched-peak coverage independent of intensity agreement.
pub fn calculate_matched_frag_fraction(obs: &[f32], lib: &[f32]) -> f32 {
    if obs.len() != lib.len() || obs.is_empty() {
        return 0.0;
    }
    let mut n_lib = 0u32;
    let mut n_matched = 0u32;
    for i in 0..lib.len() {
        if lib[i] > 0.0 {
            n_lib += 1;
            if obs[i] > 0.0 {
                n_matched += 1;
            }
        }
    }
    if n_lib == 0 {
        return 0.0;
    }
    n_matched as f32 / n_lib as f32
}

/// The 14 DIA-NN MS1/isotopologue co-elution features (Demichev 2020 Suppl Note 1).
/// All default to 0.0 — that is the value emitted when MS1 signal is absent
/// (`has_ms1()` false / old caches), making the whole panel a no-op for the NN.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ms1IsotopeFeatures {
    /// MS1 co-elution: corr(ref smoothed best-fragment profile, MS1 precursor XIC)
    /// at base / 0.45*base / 0.2*base mass accuracy (3 scores).
    pub ms1_coelution_base: f32,
    pub ms1_coelution_045: f32,
    pub ms1_coelution_02: f32,
    /// Isotopologue co-elution: corr(ref profile, MS1 XIC at precursor + 1/2/3 C13)
    /// at base mass accuracy (3 scores).
    pub iso_coelution_c13_1: f32,
    pub iso_coelution_c13_2: f32,
    pub iso_coelution_c13_3: f32,
    /// Sum over the top-6 fragments of corr(ref profile, fragment XIC shifted by
    /// +1 C13 / fragment_charge) — i.e. each fragment's +1 isotopologue (1 score).
    pub iso_frag_plus_c13_sum: f32,
    /// Per-fragment ANTI-features: corr(ref profile, fragment XIC shifted by
    /// -(C13-C12)/charge) for the top-6 fragments (6 scores). High values flag a
    /// fragment that is actually a heavy isotopologue of a LOWER-mass peak of
    /// another peptide (interference). Padded with 0.0 if fewer than 6 fragments.
    pub iso_frag_minus_c13: [f32; 6],
    /// Sum of the 6 anti-feature correlations (1 score).
    pub iso_frag_minus_c13_sum: f32,
}

impl Ms1IsotopeFeatures {
    /// Flatten to the 14 feature values in the canonical order used by
    /// `FEATURE_NAMES` / `CandidateFeature`.
    pub fn as_array(&self) -> [f32; 14] {
        [
            self.ms1_coelution_base,
            self.ms1_coelution_045,
            self.ms1_coelution_02,
            self.iso_coelution_c13_1,
            self.iso_coelution_c13_2,
            self.iso_coelution_c13_3,
            self.iso_frag_plus_c13_sum,
            self.iso_frag_minus_c13[0],
            self.iso_frag_minus_c13[1],
            self.iso_frag_minus_c13[2],
            self.iso_frag_minus_c13[3],
            self.iso_frag_minus_c13[4],
            self.iso_frag_minus_c13[5],
            self.iso_frag_minus_c13_sum,
        ]
    }
}
