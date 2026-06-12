import numpy as np, sys, time
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from search_pipeline import build_ng, tdc
from alphadia_search_rs import (SpecLibFlat as NGSpecLibFlat, PeakGroupSelection,
    SelectionParameters, PeakGroupScoring, ScoringParameters)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict

ng, rtv = build_ng("/quobyte/proteomics-grp/brett/glendon/ng_cache")
td=dict(np.load("/quobyte/proteomics-grp/brett/glendon/td_entrap.npz"))
decoy=td["decoy"]; species=td["species"]  # 0=dog,1=yeast entrapment
rl=td["rt"].astype(float); rt_abs=np.interp(rl,[rl.min(),rl.max()],[float(rtv.min()),float(rtv.max())])
print("combined precursors", len(td["precursor_idx"]),
      "dog_T", int(((species==0)&(decoy==0)).sum()), "yeast_T", int(((species==1)&(decoy==0)).sum()))
def make_lib():
    return NGSpecLibFlat.from_arrays_mobility(
        td["precursor_idx"].astype(np.uint64), td["precursor_mz"].astype(np.float32), td["precursor_mz"].astype(np.float32),
        td["rt"].astype(np.float32), rt_abs.astype(np.float32), td["im"].astype(np.float32),
        td["naa"].astype(np.uint8), td["fstart"].astype(np.uint64), td["fstop"].astype(np.uint64),
        td["fmz"].astype(np.float32), td["fmz"].astype(np.float32), td["fint"].astype(np.float32),
        np.ones(len(td["fmz"]),np.uint8), np.ones(len(td["fmz"]),np.uint8), np.zeros(len(td["fmz"]),np.uint8),
        td["fnum"].astype(np.uint8), np.zeros(len(td["fmz"]),np.uint8), td["ftype"].astype(np.uint8))
def search(mt,rtt):
    lib=make_lib()
    sp=SelectionParameters();sp.update({"mass_tolerance":mt,"rt_tolerance":rtt,"candidate_count":2,"peak_length":3,"fwhm_rt":5.0})
    cands=PeakGroupSelection(sp).search(ng,lib)
    scp=ScoringParameters();scp.update({"top_k_fragments":99,"mass_tolerance":mt})
    return PeakGroupScoring(scp).score(ng,lib,cands).to_dict_arrays()
t=time.time(); fd=search(8.0,500.0); print("search",round(time.time()-t,1),"cand_rows",len(fd["precursor_idx"]))
FEATS=[k for k in fd.keys() if k not in ("precursor_idx","rank")]
pid=np.asarray(fd["precursor_idx"]).astype(int); sc=np.asarray(fd["score"],float)
best={}
for i,p in enumerate(pid):
    if p not in best or sc[i]>sc[best[p]]: best[p]=i
ri=np.array(sorted(best.values()))
X=np.nan_to_num(np.stack([np.asarray(fd[k],float)[ri] for k in FEATS],axis=1),nan=0.,posinf=0.,neginf=0.)
pidb=pid[ri]; yb=decoy[pidb]; spb=species[pidb]
disc=cross_val_predict(LinearDiscriminantAnalysis(),X,yb,cv=3,method="decision_function")
score_t=-disc
q=tdc(score_t, yb)
# reported 1% FDR target precursors (decoy-based)
sel=(yb==0)&(q<=0.01)
n_rep=int(sel.sum())
n_dog=int((sel&(spb==0)).sum())
n_yeast=int((sel&(spb==1)).sum())
# entrapment empirical FDR: yeast targets are all FALSE. With equal dog/yeast target DB sizes,
# empirical FDR ~= 2*n_yeast / (n_dog + n_yeast)  (paired-DB; yeast estimates equal dog false count)
dogTDB=int(((species==0)&(decoy==0)).sum()); yeastTDB=int(((species==1)&(decoy==0)).sum())
ratio=dogTDB/max(yeastTDB,1)
emp_fdr = (n_yeast*ratio) / max(n_dog,1)
print(f"[ENTRAP @1% reported] total_target_IDs={n_rep} dog={n_dog} yeast_entrap={n_yeast}")
print(f"  dog_TDB={dogTDB} yeast_TDB={yeastTDB} ratio={ratio:.2f}")
print(f"  EMPIRICAL FDR (entrapment) = {emp_fdr*100:.2f}%  (reported 1%)")
# also report at a stricter decoy q to find where empirical FDR ~1%
for qt in [0.001,0.005,0.01,0.02,0.05]:
    s=(yb==0)&(q<=qt)
    nd=int((s&(spb==0)).sum()); ny=int((s&(spb==1)).sum())
    ef=(ny*ratio)/max(nd,1)
    print(f"  q<={qt}: dog={nd} yeast={ny} empFDR={ef*100:.2f}%")
