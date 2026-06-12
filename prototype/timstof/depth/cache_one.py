import numpy as np, sys, time, os
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from alphadia.raw_data.bruker import TimsTOFBase
from build_converter import convert
dpath=sys.argv[1]; outdir=sys.argv[2]
os.makedirs(outdir, exist_ok=True)
if os.path.exists(f"{outdir}/meta.npy"):
    print("already cached", outdir); sys.exit(0)
t=time.time(); dd=TimsTOFBase(dpath); print("LOAD",round(time.time()-t,1))
fpc=int(np.asarray(dd.cycle).shape[1])
t=time.time(); out=convert(dd,fpc); print("CONVERT",round(time.time()-t,1))
np.save(f"{outdir}/ev_mz.npy",out["ev_mz"]); np.save(f"{outdir}/ev_int.npy",out["ev_int"])
np.save(f"{outdir}/ev_scan.npy",out["ev_scan"].astype(np.int32))
np.save(f"{outdir}/cycle_idx.npy",out["cycle_idx"].astype(np.int32))
np.save(f"{outdir}/delta_scan_idx.npy",out["delta_scan_idx"].astype(np.int32))
np.save(f"{outdir}/iso_lo.npy",out["iso_lo"]); np.save(f"{outdir}/iso_hi.npy",out["iso_hi"])
np.save(f"{outdir}/cycle_arr.npy",np.asarray(dd.cycle).astype(np.float32))
np.save(f"{outdir}/rt_values.npy",np.asarray(dd.rt_values).astype(np.float32))
np.save(f"{outdir}/mobility_values.npy",np.asarray(dd.mobility_values).astype(np.float32))
np.save(f"{outdir}/meta.npy",np.array([fpc,out["n_scans"],out["n_frames"]],np.int64))
print("CACHED",outdir,"n_events",out["n_events"])
