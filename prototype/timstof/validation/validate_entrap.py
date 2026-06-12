# Entrapment-style validation: targets = real dog precursors (correct quad window);
# "entrapment/decoy" = SAME fragments assigned to a precursor m/z shifted by +300 Da
# (a DIFFERENT, distant isolation window). These extract from the wrong quad window
# -> a proper NULL that cannot match real co-eluting signal. Clean target/decoy.
import numpy as np, sys, time, h5py
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from alphadia_search_rs import (DIAData, SpecLibFlat as NGSpecLibFlat,
    PeakGroupSelection, SelectionParameters, PeakGroupScoring, ScoringParameters)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
C="/quobyte/proteomics-grp/brett/glendon/ng_cache"
def Ld(n): return np.load(f"{C}/{n}.npy")
ev_mz=Ld("ev_mz");ev_int=Ld("ev_int");ev_scan=Ld("ev_scan").astype(np.int64)
cyc=Ld("cycle_idx").astype(np.int64);dsi=Ld("delta_scan_idx").astype(np.int64)
iso_lo=Ld("iso_lo");iso_hi=Ld("iso_hi");cycle_arr=Ld("cycle_arr");rt_values=Ld("rt_values")
meta=Ld("meta");fpc=int(meta[0]);n_scans=int(meta[1])
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
rtv=np.asarray(ng.rt_values)
f=h5py.File("/quobyte/proteomics-grp/brett/glendon/alphadia_v2_bigdog/out_poc3_py/speclib.hdf","r")
g=f["library"]["precursor_df"]
pmz=np.asarray(g["precursor_mz"][:]);fs=np.asarray(g["frag_start_idx"][:]);fe=np.asarray(g["frag_stop_idx"][:])
rt_pred=np.asarray(g["rt_pred"][:]) if "rt_pred" in g else np.asarray(g["rt_norm_pred"][:])
cols=["b_z1","b_z2","y_z1","y_z2"]
FMZ=np.stack([np.asarray(f["library"]["fragment_mz_df"][c][:]) for c in cols],axis=-1)
FINT=np.stack([np.asarray(f["library"]["fragment_intensity_df"][c][:]) for c in cols],axis=-1)
rt_abs=np.interp(rt_pred,[rt_pred.min(),rt_pred.max()],[float(rtv.min()),float(rtv.max())])
N=int(sys.argv[1]) if len(sys.argv)>1 else 40000
selidx=np.flatnonzero((pmz>=400)&(pmz<=900))  # keep room for +300 shift <=1200
rng=np.random.default_rng(3);selidx=rng.choice(selidx,size=min(N,len(selidx)),replace=False)
pidx=[];pmz_l=[];rt_l=[];decoy_l=[];fstart=[];fstop=[];fmz_l=[];fint_l=[];cur=0;pid=0
for pi in selidx:
    a,b=int(fs[pi]),int(fe[pi]);mzr=FMZ[a:b].reshape(-1);inr=FINT[a:b].reshape(-1)
    keep=mzr>0;mzr=mzr[keep].astype(np.float32);inr=inr[keep].astype(np.float32)
    if len(mzr)<4:continue
    # TARGET: correct precursor mz
    pidx.append(pid);pmz_l.append(float(pmz[pi]));rt_l.append(float(rt_abs[pi]));decoy_l.append(0)
    fstart.append(cur);fstop.append(cur+len(mzr));fmz_l.append(mzr);fint_l.append(inr);cur+=len(mzr);pid+=1
    # NULL: SAME fragments but precursor mz +300 (different quad window) => wrong-window extraction
    pidx.append(pid);pmz_l.append(float(pmz[pi])+300.0);rt_l.append(float(rt_abs[pi]));decoy_l.append(1)
    fstart.append(cur);fstop.append(cur+len(mzr));fmz_l.append(mzr.copy());fint_l.append(inr.copy());cur+=len(mzr);pid+=1
n=len(pidx);fmz_a=np.concatenate(fmz_l);fint_a=np.concatenate(fint_l);decoy_arr=np.array(decoy_l)
print("flat lib precursors",n,"T",int((decoy_arr==0).sum()),"NULL",int((decoy_arr==1).sum()))
lib=NGSpecLibFlat.from_arrays(np.array(pidx,np.uint64),np.array(pmz_l,np.float32),np.array(pmz_l,np.float32),
    np.array(rt_l,np.float32),np.array(rt_l,np.float32),np.full(n,12,np.uint8),
    np.array(fstart,np.uint64),np.array(fstop,np.uint64),fmz_a,fmz_a,fint_a,
    np.ones(len(fmz_a),np.uint8),np.ones(len(fmz_a),np.uint8),np.zeros(len(fmz_a),np.uint8),
    np.ones(len(fmz_a),np.uint8),np.zeros(len(fmz_a),np.uint8),np.zeros(len(fmz_a),np.uint8))
sp=SelectionParameters();sp.update({"mass_tolerance":15.0,"rt_tolerance":700.0,"candidate_count":2,"peak_length":3,"fwhm_rt":5.0})
t=time.time();cands=PeakGroupSelection(sp).search(ng,lib);print("SELECT",round(time.time()-t,2),"n_cand",cands.len())
scp=ScoringParameters();scp.update({"top_k_fragments":99,"mass_tolerance":15.0})
t=time.time();feats=PeakGroupScoring(scp).score(ng,lib,cands);print("SCORE",round(time.time()-t,2))
fd=feats.to_dict_arrays();pid_a=np.asarray(fd["precursor_idx"]).astype(int)
FEATS=[k for k in fd.keys() if k not in ("precursor_idx","rank")]
X=np.nan_to_num(np.stack([np.asarray(fd[k],float) for k in FEATS],axis=1),nan=0.,posinf=0.,neginf=0.)
y=decoy_arr[pid_a];sc=np.asarray(fd["score"],float)
bestrow={}
for i,pi in enumerate(pid_a):
    if pi not in bestrow or sc[i]>sc[bestrow[pi]]: bestrow[pi]=i
ridx=np.array(sorted(bestrow.values()));Xb=X[ridx];yb=y[ridx];pidb=pid_a[ridx]
print("scored precursors",len(pidb),"T",int((yb==0).sum()),"NULL",int((yb==1).sum()))
# raw score separation
tv=sc[ridx][yb==0];dv=sc[ridx][yb==1]
print(f"raw score: T_med={np.median(tv):.2f} NULL_med={np.median(dv):.2f} T_p90={np.percentile(tv,90):.2f} NULL_p90={np.percentile(dv,90):.2f}")
def tdc(score_t):
    od=np.argsort(-score_t);ys=yb[od];nt=nd=0;q=np.ones(len(ys))
    for i in range(len(ys)):
        if ys[i]==0:nt+=1
        else:nd+=1
        q[i]=nd/max(nt,1)
    mq=1.
    for i in range(len(q)-1,-1,-1):mq=min(mq,q[i]);q[i]=mq
    return int(((ys==0)&(q<=0.01)).sum())
print("[raw score TDC] targets@1%FDR =", tdc(sc[ridx]))
lda=LinearDiscriminantAnalysis();disc=cross_val_predict(lda,Xb,yb,cv=3,method="decision_function")
print("[LDA+3CV TDC] targets@1%FDR =", tdc(-disc))
np.savez("/quobyte/proteomics-grp/brett/glendon/entrap_results.npz",pid=pidb,score=sc[ridx],lda=-disc,decoy=yb)
print("saved entrap_results.npz")
