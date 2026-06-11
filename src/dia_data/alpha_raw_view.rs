//! AlphaRawView: zero-copy view over AlphaRaw spectrum and peak arrays
//!
//! This module provides the `AlphaRawView` struct, a lightweight view over
//! NumPy-backed arrays passed from Python via PyO3 and the `numpy` crate.
//! It is designed to interoperate with Python code and provides a zero-copy view over the data.

use numpy::ndarray::{ArrayBase, Dim, ViewRepr};

type View1<'py, T> = ArrayBase<ViewRepr<&'py T>, Dim<[usize; 1]>>;

/// Zero-copy view over AlphaRaw arrays borrowed from Python
pub struct AlphaRawView<'py> {
    pub spectrum_delta_scan_idx: View1<'py, i64>,
    pub isolation_lower_mz: View1<'py, f32>,
    pub isolation_upper_mz: View1<'py, f32>,
    pub spectrum_peak_start_idx: View1<'py, i64>,
    pub spectrum_peak_stop_idx: View1<'py, i64>,
    pub spectrum_cycle_idx: View1<'py, i64>,
    pub spectrum_rt: View1<'py, f32>,
    pub peak_mz: View1<'py, f32>,
    pub peak_intensity: View1<'py, f32>,
    pub cycle: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 4]>>,

    /// Optional per-peak ion-mobility scan index (timsTOF / dia-PASEF).
    /// `None` for mobility-agnostic data (Thermo/Orbitrap).
    pub peak_scan_idx: Option<View1<'py, i64>>,

    /// Number of ion-mobility scans (1 when no mobility dimension).
    pub num_scans: usize,
}

impl<'py> AlphaRawView<'py> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spectrum_delta_scan_idx: View1<'py, i64>,
        isolation_lower_mz: View1<'py, f32>,
        isolation_upper_mz: View1<'py, f32>,
        spectrum_peak_start_idx: View1<'py, i64>,
        spectrum_peak_stop_idx: View1<'py, i64>,
        spectrum_cycle_idx: View1<'py, i64>,
        spectrum_rt: View1<'py, f32>,
        peak_mz: View1<'py, f32>,
        peak_intensity: View1<'py, f32>,
        cycle: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 4]>>,
    ) -> Self {
        Self {
            spectrum_delta_scan_idx,
            isolation_lower_mz,
            isolation_upper_mz,
            spectrum_peak_start_idx,
            spectrum_peak_stop_idx,
            spectrum_cycle_idx,
            spectrum_rt,
            peak_mz,
            peak_intensity,
            cycle,
            peak_scan_idx: None,
            num_scans: 1,
        }
    }

    /// IM-aware constructor carrying per-peak scan indices.
    #[allow(clippy::too_many_arguments)]
    pub fn new_im(
        spectrum_delta_scan_idx: View1<'py, i64>,
        isolation_lower_mz: View1<'py, f32>,
        isolation_upper_mz: View1<'py, f32>,
        spectrum_peak_start_idx: View1<'py, i64>,
        spectrum_peak_stop_idx: View1<'py, i64>,
        spectrum_cycle_idx: View1<'py, i64>,
        spectrum_rt: View1<'py, f32>,
        peak_mz: View1<'py, f32>,
        peak_intensity: View1<'py, f32>,
        cycle: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 4]>>,
        peak_scan_idx: View1<'py, i64>,
        num_scans: usize,
    ) -> Self {
        Self {
            spectrum_delta_scan_idx,
            isolation_lower_mz,
            isolation_upper_mz,
            spectrum_peak_start_idx,
            spectrum_peak_stop_idx,
            spectrum_cycle_idx,
            spectrum_rt,
            peak_mz,
            peak_intensity,
            cycle,
            peak_scan_idx: Some(peak_scan_idx),
            num_scans,
        }
    }

    pub fn has_mobility(&self) -> bool {
        self.peak_scan_idx.is_some()
    }
}
