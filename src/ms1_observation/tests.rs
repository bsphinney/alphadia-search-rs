#[cfg(test)]
mod ms1_tests {
    use crate::ms1_observation::Ms1Observation;

    #[test]
    fn extract_xic_basic_window_and_tolerance() {
        // peaks: (mz, int, cycle, scan)
        let mz = vec![500.0, 500.02, 700.0, 500.01];
        let intensity = vec![10.0, 20.0, 99.0, 5.0];
        let cycle = vec![2u32, 3, 2, 5];
        let scan = vec![100u32, 101, 100, 100];
        let obs = Ms1Observation::from_arrays(mz, intensity, cycle, scan, 800);

        // target 500.0, 50 ppm -> band ~ +/-0.025 -> includes 500.0/500.01/500.02
        // cycle window [2,4) excludes cycle 5; scan window [90,110)
        let xic = obs.extract_xic(500.0, 2, 4, 90, 110, 50.0);
        assert_eq!(xic.len(), 2);
        // cycle 2 (rel 0): 500.0 -> 10.0 (700 excluded by m/z)
        assert!((xic[0] - 10.0).abs() < 1e-4, "cycle2 = {}", xic[0]);
        // cycle 3 (rel 1): 500.02 -> 20.0
        assert!((xic[1] - 20.0).abs() < 1e-4, "cycle3 = {}", xic[1]);
    }

    #[test]
    fn extract_xic_scan_window_excludes() {
        let obs = Ms1Observation::from_arrays(
            vec![500.0, 500.0],
            vec![10.0, 20.0],
            vec![2u32, 2],
            vec![100u32, 500],
            800,
        );
        // scan window only includes scan 100
        let xic = obs.extract_xic(500.0, 2, 3, 90, 110, 20.0);
        assert!((xic[0] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn empty_store_is_all_zero() {
        let obs = Ms1Observation::from_arrays(vec![], vec![], vec![], vec![], 800);
        let xic = obs.extract_xic(500.0, 0, 5, 0, 800, 20.0);
        assert_eq!(xic.len(), 5);
        assert!(xic.iter().all(|&v| v == 0.0));
    }
}
