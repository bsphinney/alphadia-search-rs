//! Spectrum-centric fragment-ion index (optional, opt-in via SelectionParameters.use_fragment_index +
//! min_cofrag). Sage/DIA-NN style: instead of extracting a dense XIC for every ~4M precursors, iterate
//! observed peaks per cycle and scatter matches to precursors via an inverted (m/z-bin -> precursors)
//! index; a precursor is a candidate only if >= min_cofrag of its fragments CO-OCCUR in the same cycle
//! (within its isolation window, RT window, IM band). The existing per-precursor scoring then runs ONLY
//! on that shortlist. Default OFF => byte-identical to the legacy loop.
use crate::dia_data::DIAData;
use crate::mz_index::MZIndex;
use crate::speclib_flat::SpecLibFlat;
use std::collections::HashMap;

pub struct FragmentIndex {
    bin_offsets: Vec<u32>,
    prec_ids: Vec<u32>,
    pub prec_mz: Vec<f32>,
    pub prec_rt: Vec<f32>,
    pub prec_im: Vec<f32>,
    n_prec: usize,
}

impl FragmentIndex {
    pub fn build(lib: &SpecLibFlat) -> Self {
        let mz = MZIndex::global();
        let n_bins = mz.len();
        let n_prec = lib.num_precursors();
        let mut prec_mz = vec![0.0f32; n_prec];
        let mut prec_rt = vec![0.0f32; n_prec];
        let mut prec_im = vec![0.0f32; n_prec];
        let mut bin_counts = vec![0u32; n_bins + 1];
        let mut entries: Vec<(u32, u32)> = Vec::new();
        for i in 0..n_prec {
            let p = lib.get_precursor(i);
            prec_mz[i] = p.mz;
            prec_rt[i] = p.rt;
            prec_im[i] = p.mobility;
            for &fmz in p.fragment_mz.iter() {
                if fmz <= 0.0 {
                    continue;
                }
                let b = mz.find_closest_index(fmz) as u32;
                entries.push((b, i as u32));
                bin_counts[b as usize + 1] += 1;
            }
        }
        for b in 0..n_bins {
            bin_counts[b + 1] += bin_counts[b];
        }
        let bin_offsets = bin_counts.clone();
        let mut cursor = bin_offsets.clone();
        let mut prec_ids = vec![0u32; entries.len()];
        for (b, pid) in entries {
            let pos = cursor[b as usize] as usize;
            prec_ids[pos] = pid;
            cursor[b as usize] += 1;
        }
        FragmentIndex { bin_offsets, prec_ids, prec_mz, prec_rt, prec_im, n_prec }
    }

    #[inline]
    fn precursors_in_bin(&self, bin: usize) -> &[u32] {
        let s = self.bin_offsets[bin] as usize;
        let e = self.bin_offsets[bin + 1] as usize;
        &self.prec_ids[s..e]
    }

    /// Returns a bool mask (len = n_prec): true if the precursor has >= min_cofrag fragments
    /// co-occurring in some cycle within its RT window / IM band / isolation window.
    pub fn shortlist(
        &self,
        dia_data: &DIAData,
        rt_tolerance: f32,
        im_tolerance: f32,
        kernel_size: usize,
        min_cofrag: usize,
    ) -> Vec<bool> {
        let has_mob = dia_data.has_mobility;
        let mut keep = vec![false; self.n_prec];
        let mut cyc_lo = vec![0u32; self.n_prec];
        let mut cyc_hi = vec![0u32; self.n_prec];
        for i in 0..self.n_prec {
            let (a, b) = dia_data
                .rt_index
                .get_cycle_idx_limits(self.prec_rt[i], rt_tolerance, kernel_size);
            cyc_lo[i] = a as u32;
            cyc_hi[i] = b as u32;
        }
        let mut count = vec![0u8; self.n_prec];
        for obs in dia_data.quadrupole_observations.iter() {
            let iw = obs.isolation_window;
            // bucket peaks by cycle: cycle -> Vec<(bin, scan)>
            let mut per_cycle: HashMap<u16, Vec<(u32, u16)>> = HashMap::new();
            let nb = obs.slice_starts.len().saturating_sub(1);
            for b in 0..nb {
                let s = obs.slice_starts[b] as usize;
                let e = obs.slice_starts[b + 1] as usize;
                for k in s..e {
                    let cy = obs.cycle_indices[k];
                    let sc = if has_mob && k < obs.scan_indices.len() {
                        obs.scan_indices[k]
                    } else {
                        0u16
                    };
                    per_cycle.entry(cy).or_default().push((b as u32, sc));
                }
            }
            for (cy, peaks) in per_cycle.iter() {
                let cyc = *cy as u32;
                let mut touched: Vec<u32> = Vec::new();
                for &(bin, scan) in peaks.iter() {
                    for &pid in self.precursors_in_bin(bin as usize) {
                        let pi = pid as usize;
                        if keep[pi] {
                            continue;
                        }
                        if self.prec_mz[pi] < iw[0] || self.prec_mz[pi] > iw[1] {
                            continue;
                        }
                        if cyc < cyc_lo[pi] || cyc >= cyc_hi[pi] {
                            continue;
                        }
                        if has_mob && im_tolerance > 0.0 && self.prec_im[pi] > 0.0 {
                            let mobv = dia_data.mobility_per_scan.get(scan as usize).copied().unwrap_or(0.0);
                            if (mobv - self.prec_im[pi]).abs() > im_tolerance {
                                continue;
                            }
                        }
                        if count[pi] == 0 {
                            touched.push(pid);
                        }
                        if count[pi] < 255 {
                            count[pi] += 1;
                        }
                    }
                }
                for &pid in touched.iter() {
                    let pi = pid as usize;
                    if count[pi] as usize >= min_cofrag {
                        keep[pi] = true;
                    }
                    count[pi] = 0;
                }
            }
        }
        keep
    }
}
