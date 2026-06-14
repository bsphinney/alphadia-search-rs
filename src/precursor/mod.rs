pub struct Precursor {
    pub precursor_idx: usize,
    pub mz: f32,
    pub mz_library: f32,
    pub rt: f32,
    pub rt_library: f32,
    /// Predicted/library ion mobility (1/K0). 0.0 when not provided.
    pub mobility: f32,
    /// Precursor charge state. 0 when not provided by the library (then the
    /// isotope-spacing features are skipped — the C13 m/z step is C13_C12/charge).
    pub charge: u8,
    pub naa: u8,
    pub fragment_mz: Vec<f32>,
    pub fragment_mz_library: Vec<f32>,
    pub fragment_intensity: Vec<f32>,
    pub fragment_cardinality: Vec<u8>,
    pub fragment_charge: Vec<u8>,
    pub fragment_loss_type: Vec<u8>,
    pub fragment_number: Vec<u8>,
    pub fragment_position: Vec<u8>,
    pub fragment_type: Vec<u8>,
}

#[cfg(test)]
mod tests;
