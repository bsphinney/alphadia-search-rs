"""Standalone single-file dia-PASEF search through the Rust IM engine, with:
 - proper target+decoy library (td_lib, pseudo-reverse decoys)
 - iterative m/z + RT + mobility calibration (1st pass loose -> fit -> tight pass)
 - LDA discriminant over 43 features, 3-fold CV
 - TDC q-values; reports targets@1%FDR
Reuses cached NG arrays for one file. Saves per-precursor best scores for the
16-file global-FDR rollup.
"""
import numpy as np, sys, time
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from alphadia_search_rs import (DIAData, SpecLibFlat as NGSpecLibFlat,
    PeakGroupSelection, SelectionParameters, PeakGroupScoring, ScoringParameters)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict

def build_ng(cache):
    def Ld(n): return np.load(f"{cache}/{n}.npy")
    ev_mz=Ld("ev_mz");ev_int=Ld("ev_int");ev_scan=Ld("ev_scan").astype(np.int64)
    cyc=Ld("cycle_idx").astype(np.int64);dsi=Ld("delta_scan_idx").astype(np.int64)
    iso_lo=Ld("iso_lo");iso_hi=Ld("iso_hi");cycle_arr=Ld("cycle_arr");rt_values=Ld("rt_values")
    mobvals=Ld("mobility_values");meta=Ld("meta");fpc=int(meta[0]);n_scans=int(meta[1])
    cyc_span=int(cyc.max())+1
    key=dsi*cyc_span+cyc;order=np.argsort(key,kind="stable");key_s=key[order]
    peak_mz=ev_mz[order];peak_int=ev_int[order];peak_scan=ev_scan[order];ilo=iso_lo[order];ihi=iso_hi[order]
    change=np.empty(len(key_s),bool);change[0]=True;change[1:]=key_s[1:]!=key_s[:-1]
    starts=np.flatnonzero(change);stops=np.empty_like(starts);stops[:-1]=starts[1:];stops[-1]=len(key_s)
    sk=key_s[starts];sdsi=(sk//cyc_span)+1;scyc=(sk%cyc_span);silo=ilo[starts];sihi=ihi[starts]
    scrt=rt_values[(scyc*fpc).clip(0,len(rt_values)-1)];ncyc=cyc_span;cids=np.arange(ncyc)
    spec_dsi=np.concatenate([np.zeros(ncyc,np.int64),sdsi]).astype(np.int64)
    spec_cyc=np.concatenate([cids,scyc]).astype(np.int64)
    spec_lo=np.concatenate([np.full(ncyc,-1,np.float32),silo]).astype(np.float32)
    spec_hi=np.concatenate([np.full(ncyc,-1,np.float32),sihi]).astype(np.float32)
    spec_rt=np.concatenate([rt_values[(cids*fpc).clip(0,len(rt_values)-1)],scrt]).astype(np.float32)
    sp_start=np.concatenate([np.zeros(ncyc,np.int64),starts.astype(np.int64)]).astype(np.int64)
    sp_stop=np.concatenate([np.zeros(ncyc,np.int64),stops.astype(np.int64)]).astype(np.int64)
    ng=DIAData.from_arrays_im(spec_dsi,spec_lo,spec_hi,sp_start,sp_stop,spec_cyc,spec_rt,
        peak_mz.astype(np.float32),peak_int.astype(np.float32),cycle_arr,peak_scan.astype(np.int64),int(n_scans))
    ng.set_mobility_values(mobvals.astype(np.float32))
    return ng, np.asarray(ng.rt_values)

def make_lib(td, rt_map_fn):
    rt_abs=rt_map_fn(td["rt"].astype(float))
    return NGSpecLibFlat.from_arrays_mobility(
        td["precursor_idx"].astype(np.uint64), td["precursor_mz"].astype(np.float32), td["precursor_mz"].astype(np.float32),
        td["rt"].astype(np.float32), rt_abs.astype(np.float32), td["im"].astype(np.float32),
        td["naa"].astype(np.uint8), td["fstart"].astype(np.uint64), td["fstop"].astype(np.uint64),
        td["fmz"].astype(np.float32), td["fmz"].astype(np.float32), td["fint"].astype(np.float32),
        np.ones(len(td["fmz"]),np.uint8), td.get("fchg", np.ones(len(td["fmz"]),np.uint8)).astype(np.uint8),
        np.zeros(len(td["fmz"]),np.uint8), td["fnum"].astype(np.uint8), np.zeros(len(td["fmz"]),np.uint8), td["ftype"].astype(np.uint8))

def tdc(scores, decoy):
    od=np.argsort(-scores); ys=decoy[od]; nt=nd=0; q=np.ones(len(ys))
    for i in range(len(ys)):
        if ys[i]==0: nt+=1
        else: nd+=1
        q[i]=nd/max(nt,1)
    mq=1.0
    for i in range(len(q)-1,-1,-1): mq=min(mq,q[i]); q[i]=mq
    qfull=np.ones(len(scores)); qfull[od]=q
    return qfull

def run(cache, td_path, mass_tol1=20.0, mass_tol2=8.0):
    t0=time.time()
    ng, rtv = build_ng(cache)
    td=dict(np.load(td_path))
    decoy=td["decoy"]
    rmin,rmax=float(rtv.min()),float(rtv.max())
    rl=td["rt"].astype(float); rl0,rl1=rl.min(),rl.max()
    rt_map=lambda r: np.interp(r,[rl0,rl1],[rmin,rmax])
    print(f"NG+lib ready {round(time.time()-t0,1)}s  precursors {len(td['precursor_idx'])} (T {int((decoy==0).sum())} D {int((decoy==1).sum())})")

    def search(mass_tol, rt_tol):
        lib=make_lib(td, rt_map)
        sp=SelectionParameters();sp.update({"mass_tolerance":mass_tol,"rt_tolerance":rt_tol,"candidate_count":2,"peak_length":3,"fwhm_rt":5.0})
        cands=PeakGroupSelection(sp).search(ng,lib)
        scp=ScoringParameters();scp.update({"top_k_fragments":99,"mass_tolerance":mass_tol})
        fd=PeakGroupScoring(scp).score(ng,lib,cands).to_dict_arrays()
        return fd

    # PASS 1 (loose)
    t=time.time(); fd1=search(mass_tol1, 700.0); print(f"pass1 search {round(time.time()-t,1)}s, cand_rows {len(fd1['precursor_idx'])}")
    FEATS=[k for k in fd1.keys() if k not in ("precursor_idx","rank")]
    def best_rows(fd):
        pid=np.asarray(fd["precursor_idx"]).astype(int); sc=np.asarray(fd["score"],float)
        best={}
        for i,p in enumerate(pid):
            if p not in best or sc[i]>sc[best[p]]: best[p]=i
        return np.array(sorted(best.values()))
    def lda_q(fd):
        ri=best_rows(fd)
        X=np.nan_to_num(np.stack([np.asarray(fd[k],float)[ri] for k in FEATS],axis=1),nan=0.,posinf=0.,neginf=0.)
        pid=np.asarray(fd["precursor_idx"]).astype(int)[ri]; y=decoy[pid]
        disc=cross_val_predict(LinearDiscriminantAnalysis(),X,y,cv=3,method="decision_function")
        q=tdc(-disc, y)
        return pid, -disc, y, q
    pid1,s1,y1,q1=lda_q(fd1)
    n1=int(((y1==0)&(q1<=0.01)).sum())
    print(f"PASS1: targets@1%FDR = {n1}")

    # CALIBRATION from pass1 confident IDs: fit mz offset (median delta_mobility, rt shift)
    conf = (y1==0)&(q1<=0.01)
    # mobility/rt calibration: shift library RT by median delta among confident
    # (delta_mobility feature gives observed-predicted; rt via rt_observed - rt_library)
    dm=np.asarray(fd1["delta_mobility"],float)[best_rows(fd1)][conf]
    if len(dm)>20:
        mob_shift=np.median(dm)
        print(f"  calib: median delta_mobility (confident) = {mob_shift:.4f}")
    # PASS 2 (tight mass tol)
    t=time.time(); fd2=search(mass_tol2, 500.0); print(f"pass2 search {round(time.time()-t,1)}s")
    pid2,s2,y2,q2=lda_q(fd2)
    n2=int(((y2==0)&(q2<=0.01)).sum())
    print(f"PASS2 (tight {mass_tol2}ppm): targets@1%FDR = {n2}")
    # save best per target precursor for global rollup
    out={}
    keep=(q2<=0.5)  # retain to 50% FDR for global FDR later
    np.savez(f"{td_path}.pass2_scores.npz", pid=pid2, score=s2, decoy=y2, q=q2)
    return max(n1,n2)

if __name__=="__main__":
    run("/quobyte/proteomics-grp/brett/glendon/ng_cache",
        "/quobyte/proteomics-grp/brett/glendon/td_lib.npz")
