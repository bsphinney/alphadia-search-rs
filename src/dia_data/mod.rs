use numpy::ndarray::{Array1, Array4};
use numpy::{PyArray1, PyArray4, PyReadonlyArray1, PyReadonlyArray4};
use pyo3::{prelude::*, Bound};
mod alpha_raw_view;
use crate::dia_data_builder::DIADataBuilder;
use crate::ms1_observation::Ms1Observation;
use crate::mz_index::MZIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::rt_index::RTIndex;
pub use alpha_raw_view::AlphaRawView;

/// DIAData structure using optimized memory layout
///
/// This structure achieves >99.9% memory overhead reduction compared to the original
/// by using consolidated arrays instead of millions of individual allocations.
///
/// Ion-mobility (timsTOF / dia-PASEF): when built from IM data, `has_mobility` is
/// true and `num_scans > 1`; each `QuadrupoleObservation` then carries per-event
/// scan indices enabling mobility-windowed and 3D extraction.
#[pyclass]
pub struct DIAData {
    pub rt_index: RTIndex,
    pub quadrupole_observations: Vec<QuadrupoleObservation>,
    pub rt_values: Array1<f32>,
    pub cycle: Array4<f32>,
    pub num_scans: usize,
    pub has_mobility: bool,
    /// Per-scan ion-mobility (1/K0) values, length == num_scans (timsTOF). Empty
    /// for mobility-agnostic data. Set via `set_mobility_values` after construction.
    pub mobility_per_scan: Vec<f32>,

    /// MS1 survey-frame peak store (A1). `None` for caches without MS1 arrays
    /// (then `has_ms1()` is false and scoring is byte-identical to before). Set
    /// via `set_ms1_arrays` after construction.
    pub ms1: Option<Ms1Observation>,
}

impl Default for DIAData {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl DIAData {
    #[new]
    pub fn new() -> Self {
        Self {
            rt_index: RTIndex::new(),
            quadrupole_observations: Vec::new(),
            rt_values: Array1::zeros((0,)),
            cycle: Array4::zeros((0, 0, 0, 0)),
            num_scans: 1,
            has_mobility: false,
            mobility_per_scan: Vec::new(),
            ms1: None,
        }
    }

    /// Set the per-scan ion-mobility (1/K0) values (length should equal num_scans).
    /// Used by the timsTOF feeder to provide real mobility for mobility-error scoring.
    pub fn set_mobility_values(&mut self, mobility_values: Vec<f32>) {
        self.mobility_per_scan = mobility_values;
    }

    /// Attach the MS1 survey-frame arrays (A1: MS1/isotope scoring features).
    /// `ms1_mz/ms1_intensity` are parallel peak arrays; `ms1_cycle/ms1_scan` are
    /// the RT(cycle) and IM(scan) indices (parallel). Builds an m/z-sorted
    /// `Ms1Observation`. Additive: callers using old caches simply never call
    /// this, leaving `has_ms1()` false and scoring unchanged.
    pub fn set_ms1_arrays(
        &mut self,
        ms1_mz: PyReadonlyArray1<'_, f32>,
        ms1_intensity: PyReadonlyArray1<'_, f32>,
        ms1_cycle: PyReadonlyArray1<'_, i64>,
        ms1_scan: PyReadonlyArray1<'_, i64>,
    ) {
        let mz = ms1_mz.as_array().to_vec();
        let intensity = ms1_intensity.as_array().to_vec();
        let cycle: Vec<u32> = ms1_cycle.as_array().iter().map(|&c| c.max(0) as u32).collect();
        let scan: Vec<u32> = ms1_scan.as_array().iter().map(|&s| s.max(0) as u32).collect();
        if mz.is_empty() {
            self.ms1 = None;
            return;
        }
        self.ms1 = Some(Ms1Observation::from_arrays(
            mz,
            intensity,
            cycle,
            scan,
            self.num_scans,
        ));
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays<'py>(
        spectrum_delta_scan_idx: PyReadonlyArray1<'py, i64>,
        isolation_lower_mz: PyReadonlyArray1<'py, f32>,
        isolation_upper_mz: PyReadonlyArray1<'py, f32>,
        spectrum_peak_start_idx: PyReadonlyArray1<'py, i64>,
        spectrum_peak_stop_idx: PyReadonlyArray1<'py, i64>,
        spectrum_cycle_idx: PyReadonlyArray1<'py, i64>,
        spectrum_rt: PyReadonlyArray1<'py, f32>,
        peak_mz: PyReadonlyArray1<'py, f32>,
        peak_intensity: PyReadonlyArray1<'py, f32>,
        cycle: PyReadonlyArray4<'py, f32>,
        _py: Python<'py>,
    ) -> PyResult<Self> {
        let alpha_raw_view = AlphaRawView::new(
            spectrum_delta_scan_idx.as_array(),
            isolation_lower_mz.as_array(),
            isolation_upper_mz.as_array(),
            spectrum_peak_start_idx.as_array(),
            spectrum_peak_stop_idx.as_array(),
            spectrum_cycle_idx.as_array(),
            spectrum_rt.as_array(),
            peak_mz.as_array(),
            peak_intensity.as_array(),
            cycle.as_array(),
        );

        // Use optimized builder
        let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        Ok(dia_data)
    }

    /// IM-aware constructor for timsTOF / dia-PASEF.
    ///
    /// Identical to `from_arrays` plus a per-peak `peak_scan_idx` (ion-mobility
    /// scan index, parallel to `peak_mz`/`peak_intensity`) and `num_scans`
    /// (total number of mobility scans per frame).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays_im<'py>(
        spectrum_delta_scan_idx: PyReadonlyArray1<'py, i64>,
        isolation_lower_mz: PyReadonlyArray1<'py, f32>,
        isolation_upper_mz: PyReadonlyArray1<'py, f32>,
        spectrum_peak_start_idx: PyReadonlyArray1<'py, i64>,
        spectrum_peak_stop_idx: PyReadonlyArray1<'py, i64>,
        spectrum_cycle_idx: PyReadonlyArray1<'py, i64>,
        spectrum_rt: PyReadonlyArray1<'py, f32>,
        peak_mz: PyReadonlyArray1<'py, f32>,
        peak_intensity: PyReadonlyArray1<'py, f32>,
        cycle: PyReadonlyArray4<'py, f32>,
        peak_scan_idx: PyReadonlyArray1<'py, i64>,
        num_scans: usize,
        _py: Python<'py>,
    ) -> PyResult<Self> {
        let alpha_raw_view = AlphaRawView::new_im(
            spectrum_delta_scan_idx.as_array(),
            isolation_lower_mz.as_array(),
            isolation_upper_mz.as_array(),
            spectrum_peak_start_idx.as_array(),
            spectrum_peak_stop_idx.as_array(),
            spectrum_cycle_idx.as_array(),
            spectrum_rt.as_array(),
            peak_mz.as_array(),
            peak_intensity.as_array(),
            cycle.as_array(),
            peak_scan_idx.as_array(),
            num_scans,
        );

        let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
        Ok(dia_data)
    }

    #[getter]
    pub fn num_observations(&self) -> usize {
        self.quadrupole_observations.len()
    }

    pub fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize> {
        let mut valid_observations = Vec::new();
        for (i, obs) in self.quadrupole_observations.iter().enumerate() {
            if obs.isolation_window[0] <= precursor_mz && obs.isolation_window[1] >= precursor_mz {
                valid_observations.push(i);
            }
        }
        valid_observations
    }

    /// Returns the memory footprint of the optimized DIAData structure in bytes
    pub fn memory_footprint_bytes(&self) -> usize {
        let mut total_size = 0;

        // Size of RTIndex (MZIndex is global and not owned by this struct)
        total_size += self.rt_index.rt.len() * std::mem::size_of::<f32>();

        // Size of quadrupole_observations Vec overhead
        total_size += std::mem::size_of::<Vec<QuadrupoleObservation>>();

        // Size of each optimized QuadrupoleObservation
        for obs in &self.quadrupole_observations {
            total_size += obs.memory_footprint_bytes();
        }

        if let Some(ms1) = &self.ms1 {
            total_size += ms1.memory_footprint_bytes();
        }

        total_size
    }

    /// Returns the memory footprint in megabytes for easier reading
    pub fn memory_footprint_mb(&self) -> f64 {
        self.memory_footprint_bytes() as f64 / (1024.0 * 1024.0)
    }

    #[getter]
    pub fn has_mobility(&self) -> bool {
        self.has_mobility
    }

    #[getter]
    pub fn has_ms1(&self) -> bool {
        self.ms1.as_ref().map(|m| !m.is_empty()).unwrap_or(false)
    }

    #[getter]
    pub fn num_scans(&self) -> usize {
        self.num_scans
    }

    #[getter]
    pub fn mobility_values(&self) -> Vec<f32> {
        // Returns the real per-scan 1/K0 values when provided (via
        // set_mobility_values); otherwise a scan-index ramp for IM data or the
        // historical 2-element placeholder for mobility-agnostic data.
        if !self.mobility_per_scan.is_empty() {
            self.mobility_per_scan.clone()
        } else if self.has_mobility {
            (0..self.num_scans).map(|s| s as f32).collect()
        } else {
            vec![1e-6, 0.0]
        }
    }

    #[getter]
    pub fn rt_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_array(py, &self.rt_values)
    }

    #[getter]
    pub fn cycle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f32>> {
        PyArray4::from_array(py, &self.cycle)
    }
}

// Implement the DIADataTrait for DIAData
impl crate::traits::DIADataTrait for DIAData {
    type QuadrupoleObservation = crate::quadrupole_observation::QuadrupoleObservation;

    fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize> {
        self.get_valid_observations(precursor_mz)
    }

    fn mz_index(&self) -> &crate::mz_index::MZIndex {
        MZIndex::global()
    }

    fn rt_index(&self) -> &crate::rt_index::RTIndex {
        &self.rt_index
    }

    fn quadrupole_observations(&self) -> &[Self::QuadrupoleObservation] {
        &self.quadrupole_observations
    }

    fn memory_footprint_bytes(&self) -> usize {
        self.memory_footprint_bytes()
    }

    fn has_mobility(&self) -> bool {
        self.has_mobility
    }

    fn num_scans(&self) -> usize {
        self.num_scans
    }

    fn mobility_of_scan(&self, scan_idx: usize) -> f32 {
        if scan_idx < self.mobility_per_scan.len() {
            self.mobility_per_scan[scan_idx]
        } else {
            0.0
        }
    }

    fn has_ms1(&self) -> bool {
        self.has_ms1()
    }

    fn ms1(&self) -> Option<&crate::ms1_observation::Ms1Observation> {
        self.ms1.as_ref()
    }
}

#[cfg(test)]
mod tests;
