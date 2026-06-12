# Run the converter once and cache the per-spectrum + per-peak arrays to .npy
import numpy as np, sys, time
sys.path.insert(0, "/quobyte/proteomics-grp/brett/glendon")
from alphadia.raw_data.bruker import TimsTOFBase
from build_converter import convert
d = "/nfs/lssc0/flinders/proteomics/Data/raw_data/tTOF_HT/may26/Ameer_Taha/08May2026_DIA_60spd_VER_10_S2-B2_1_21552.d"
t=time.time(); dd = TimsTOFBase(d); print("LOAD_SEC", round(time.time()-t,1))
fpc = int(np.asarray(dd.cycle).shape[1])
t=time.time(); out = convert(dd, fpc); print("CONVERT_SEC", round(time.time()-t,1))
outdir = "/quobyte/proteomics-grp/brett/glendon/ng_cache"
import os; os.makedirs(outdir, exist_ok=True)
np.save(f"{outdir}/ev_mz.npy", out["ev_mz"])
np.save(f"{outdir}/ev_int.npy", out["ev_int"])
np.save(f"{outdir}/ev_scan.npy", out["ev_scan"].astype(np.int32))
np.save(f"{outdir}/cycle_idx.npy", out["cycle_idx"].astype(np.int32))
np.save(f"{outdir}/delta_scan_idx.npy", out["delta_scan_idx"].astype(np.int32))
np.save(f"{outdir}/iso_lo.npy", out["iso_lo"])
np.save(f"{outdir}/iso_hi.npy", out["iso_hi"])
np.save(f"{outdir}/cycle_arr.npy", np.asarray(dd.cycle).astype(np.float32))
np.save(f"{outdir}/rt_values.npy", np.asarray(dd.rt_values).astype(np.float32))
np.save(f"{outdir}/meta.npy", np.array([fpc, out["n_scans"], out["n_frames"]], dtype=np.int64))
print("CACHED to", outdir, "n_events", out["n_events"])
