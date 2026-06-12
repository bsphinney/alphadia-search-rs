"""16-file dia-PASEF search through the Rust IM engine + GLOBAL cross-file
precursor FDR + protein-group rollup. Reports PG count at 1% FDR vs DIA-NN/FragPipe.
Uses per-file ng caches in cache16/. Library = td_lib (dog target+decoy).
"""
import numpy as np, sys, time, os, glob
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from search_pipeline import build_ng, tdc
from alphadia_search_rs import (SpecLibFlat as NGSpecLibFlat, PeakGroupSelection,
    SelectionParameters, PeakGroupScoring, ScoringParameters)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
import pandas as pd

LIB="/quobyte/proteomics-grp/brett/glendon/td_lib.npz"
td=dict(np.load(LIB)); decoy=td["decoy"]
# precursor_idx -> protein group mapping from DIA-NN lib (target precursors only)
lib_df=pd.read_parquet("/quobyte/proteomics-grp/brett/glendon/diann251_clean16/report-lib.parquet")
lib_df=lib_df[lib_df["Decoy"]==0]
prec_first=lib_df.groupby("Precursor.Id",sort=False).first().reset_index()
# td_lib built target precursors in the SAME order as DIA-NN grouping (z=1 filtered);
# but td_lib's precursor_idx are 0,2,4,... for targets. Reconstruct mapping by matching
# precursor_mz+rt+im is fragile; instead rebuild td-target order = DIA-NN z1 group order.
# Simpler: re-read td build assigned target pid=0,2,4.. in order of DIA-NN keys (z=1).
# We'll map td target index k -> DIA-NN precursor k (same iteration order).
lib_z1=lib_df[lib_df["Fragment.Charge"]==1]
keys_z1=list(lib_z1.groupby("Precursor.Id",sort=False).groups.keys())
pg_by_key={}
gg=lib_z1.groupby("Precursor.Id",sort=False)["Protein.Group"].first()
for k in keys_z1: pg_by_key[k]=gg[k]
# td targets that survived (>=4 frags). We saved nothing about which survived; recompute counts.
# Build target_pid -> PG by walking same filter as build_lib_td (>=4 z1 frags).
target_pid_to_pg={}
tpid=0
for k in keys_z1:
    sub=lib_z1[lib_z1["Precursor.Id"]==k] if False else None
# (vectorized) count z1 frags per key
cnt=lib_z1.groupby("Precursor.Id",sort=False).size()
tpid=0
for k in keys_z1:
    if cnt[k]>=4:
        target_pid_to_pg[tpid]=pg_by_key[k]
        tpid+=1
print("target precursors with PG", len(target_pid_to_pg), "expected td targets", int((decoy==0).sum()))

def run_file(cache_dir):
    ng,rtv=build_ng(cache_dir)
    rl=td["rt"].astype(float); rt_abs=np.interp(rl,[rl.min(),rl.max()],[float(rtv.min()),float(rtv.max())])
    lib=NGSpecLibFlat.from_arrays_mobility(
        td["precursor_idx"].astype(np.uint64),td["precursor_mz"].astype(np.float32),td["precursor_mz"].astype(np.float32),
        td["rt"].astype(np.float32),rt_abs.astype(np.float32),td["im"].astype(np.float32),
        td["naa"].astype(np.uint8),td["fstart"].astype(np.uint64),td["fstop"].astype(np.uint64),
        td["fmz"].astype(np.float32),td["fmz"].astype(np.float32),td["fint"].astype(np.float32),
        np.ones(len(td["fmz"]),np.uint8),np.ones(len(td["fmz"]),np.uint8),np.zeros(len(td["fmz"]),np.uint8),
        td["fnum"].astype(np.uint8),np.zeros(len(td["fmz"]),np.uint8),td["ftype"].astype(np.uint8))
    sp=SelectionParameters();sp.update({"mass_tolerance":8.0,"rt_tolerance":600.0,"candidate_count":2,"peak_length":3,"fwhm_rt":5.0})
    cands=PeakGroupSelection(sp).search(ng,lib)
    scp=ScoringParameters();scp.update({"top_k_fragments":99,"mass_tolerance":8.0})
    fd=PeakGroupScoring(scp).score(ng,lib,cands).to_dict_arrays()
    return fd

FEATS=None
all_rows=[]  # (precursor_idx, feature_vector, decoy)
caches=sorted(glob.glob("/quobyte/proteomics-grp/brett/glendon/cache16/*/"))
print("caches found", len(caches))
for cd in caches:
    if not os.path.exists(cd+"meta.npy"): 
        print("skip (no cache)", cd); continue
    t=time.time(); fd=run_file(cd)
    if FEATS is None: FEATS=[k for k in fd.keys() if k not in ("precursor_idx","rank")]
    pid=np.asarray(fd["precursor_idx"]).astype(int); sc=np.asarray(fd["score"],float)
    best={}
    for i,p in enumerate(pid):
        if p not in best or sc[i]>sc[best[p]]: best[p]=i
    ri=np.array(sorted(best.values()))
    X=np.nan_to_num(np.stack([np.asarray(fd[k],float)[ri] for k in FEATS],axis=1),nan=0.,posinf=0.,neginf=0.)
    pb=pid[ri]
    for j in range(len(pb)):
        all_rows.append((pb[j], X[j], int(decoy[pb[j]])))
    print(f"  {os.path.basename(cd.rstrip('/'))}: best-rows {len(pb)} in {round(time.time()-t,1)}s", flush=True)

print("total file-level best rows", len(all_rows))
Xall=np.stack([r[1] for r in all_rows]); pidall=np.array([r[0] for r in all_rows]); yall=np.array([r[2] for r in all_rows])
# train ONE classifier across all files, 3-fold CV discriminant
disc=cross_val_predict(LinearDiscriminantAnalysis(),Xall,yall,cv=3,method="decision_function")
score_t=-disc
# GLOBAL cross-file precursor FDR: best score per precursor across ALL files
bestp={}
for i in range(len(pidall)):
    p=pidall[i]
    if p not in bestp or score_t[i]>bestp[p][0]: bestp[p]=(score_t[i], yall[i])
items=sorted(bestp.items(), key=lambda kv:-kv[1][0])
ps=np.array([k for k,_ in items]); ss=np.array([v[0] for _,v in items]); ys=np.array([v[1] for _,v in items])
q=tdc(ss, ys)
sel=(ys==0)&(q<=0.01)
prec_ids=ps[sel]
n_prec=int(sel.sum())
print(f"[GLOBAL 16-file] precursors @1% FDR (cross-file) = {n_prec}")
# rollup to protein groups
pgs=set()
pgs_multi={}
for pid_ in prec_ids:
    pg=target_pid_to_pg.get(int(pid_))
    if pg is not None:
        pgs.add(pg); pgs_multi[pg]=pgs_multi.get(pg,0)+1
n_pg=len(pgs)
n_pg_multi=sum(1 for pg,c in pgs_multi.items() if c>=2)
print(f"[GLOBAL 16-file] PROTEIN GROUPS @1% FDR = {n_pg}  (multi-peptide PG = {n_pg_multi})")
print(f"  vs DIA-NN 1306 PG / 17770 precursors ; FragPipe 1337 PG")
np.savez("/quobyte/proteomics-grp/brett/glendon/global16_result.npz", prec_ids=prec_ids, n_prec=n_prec, n_pg=n_pg, n_pg_multi=n_pg_multi)
