use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;
use numpy::ndarray::{ArrayViewMut1, ArrayViewMut2};

/// Trait for DIA data structures that support peak group scoring
pub trait DIADataTrait {
    type QuadrupoleObservation: QuadrupoleObservationTrait;

    fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize>;
    fn mz_index(&self) -> &MZIndex;
    fn rt_index(&self) -> &RTIndex;
    fn quadrupole_observations(&self) -> &[Self::QuadrupoleObservation];

    // Common functionality that both implementations have
    fn num_observations(&self) -> usize {
        self.quadrupole_observations().len()
    }

    fn memory_footprint_bytes(&self) -> usize;

    fn memory_footprint_mb(&self) -> f64 {
        self.memory_footprint_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Whether this data carries an ion-mobility dimension (timsTOF / dia-PASEF).
    /// Defaults to false for mobility-agnostic backends.
    fn has_mobility(&self) -> bool {
        false
    }

    /// Number of ion-mobility scans (1 when no mobility dimension).
    fn num_scans(&self) -> usize {
        1
    }

    /// Map an ion-mobility scan index to its 1/K0 value. Returns 0.0 when no
    /// per-scan mobility is available (mobility-agnostic data or not provided).
    fn mobility_of_scan(&self, _scan_idx: usize) -> f32 {
        0.0
    }

    /// Whether MS1 survey-frame signal is available (A1: MS1/isotope features).
    /// Defaults to false; backends with an MS1 store override this.
    fn has_ms1(&self) -> bool {
        false
    }

    /// Access the MS1 survey-frame peak store, if present. Default `None`.
    fn ms1(&self) -> Option<&crate::ms1_observation::Ms1Observation> {
        None
    }
}

/// Trait for quadrupole observation types that support XIC slice filling
pub trait QuadrupoleObservationTrait {
    fn fill_xic_slice(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    );

    fn fill_xic_and_mz_slice(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        dense_mz: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    );

    /// IM-aware 1D extraction restricted to the mobility window
    /// `[scan_start, scan_stop)`. Default falls back to `fill_xic_slice`.
    #[allow(clippy::too_many_arguments)]
    fn fill_xic_slice_mobility_windowed(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        _scan_start: usize,
        _scan_stop: usize,
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
        );
    }

    /// IM-aware 3D extraction into a `[scan, cycle]` matrix. Default is a no-op
    /// (mobility-agnostic data has no scan dimension to fill).
    #[allow(clippy::too_many_arguments)]
    fn fill_dense_xic_3d_trait(
        &self,
        _mz_index: &MZIndex,
        _dense_xic_2d: &mut ArrayViewMut2<f32>,
        _cycle_start_idx: usize,
        _cycle_stop_idx: usize,
        _scan_start: usize,
        _scan_stop: usize,
        _mass_tolerance: f32,
        _mz: f32,
    ) {
    }

    /// IM-aware extraction restricted to the mobility window `[scan_start, scan_stop)`.
    /// Default falls back to the full-range `fill_xic_and_mz_slice` (correct for
    /// mobility-agnostic data, where the scan window is meaningless).
    #[allow(clippy::too_many_arguments)]
    fn fill_xic_and_mz_slice_mobility_windowed(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        dense_mz: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        _scan_start: usize,
        _scan_stop: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        self.fill_xic_and_mz_slice(
            mz_index,
            dense_xic,
            dense_mz,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            mz,
        );
    }
}
