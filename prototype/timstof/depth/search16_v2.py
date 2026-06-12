"""16-file search + Radiant-recipe NN classifier + global MixMax FDR + NSP PG scoring.
Caches per-file features so classifier iterations are fast.
"""
import numpy as np, sys, time, os, glob, pandas as pd
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from search_pipeline import build_ng, tdc
from alphadia_search_rs import (SpecLibFlat as NGSpecLibFlat, PeakGroupSelection,
    SelectionParameters, PeakGroupScoring, ScoringParameters)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

LIB=os.environ.get("TDLIB","/quobyte/proteomics-grp/brett/glendon/td_lib.npz")
FEATCACHE=os.environ.get("FEATCACHE","/quobyte/proteomics-grp/brett/glendon/feat16.npz")
td=dict(np.load(LIB)); decoy=td["decoy"]

# PG mapping (target pid -> protein group), z1 frags >=4 filter mirrors build_lib_td
lib_df=pd.read_parquet("/quobyte/proteomics-grp/brett/glendon/diann251_clean16/report-lib.parquet")
lib_df=lib_df[lib_df["Decoy"]==0]; lib_z1=lib_df[lib_df["Fragment.Charge"]==1]
keys_z1=list(lib_z1.groupby("Precursor.Id",sort=False).groups.keys())
gg=lib_z1.groupby("Precursor.Id",sort=False)["Protein.Group"].first()
cnt=lib_z1.groupby("Precursor.Id",sort=False).size()
target_pid_to_pg={}; tpid=0
for k in keys_z1:
    if cnt[k]>=4: target_pid_to_pg[tpid]=gg[k]; tpid+=1

def run_file(cd):
    ng,rtv=build_ng(cd)
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
    return PeakGroupScoring(scp).score(ng,lib,cands).to_dict_arrays()

if os.path.exists(FEATCACHE):
    Z=np.load(FEATCACHE,allow_pickle=True)
    Xall=Z["X"]; pidall=Z["pid"]; yall=Z["y"]; fileall=Z["fidx"]; FEATS=list(Z["feats"])
    print("loaded feat cache", Xall.shape, flush=True)
else:
    caches=sorted(glob.glob("/quobyte/proteomics-grp/brett/glendon/cache16/*/"))
    FEATS=None; rows_X=[]; rows_pid=[]; rows_y=[]; rows_fidx=[]
    for fi,cd in enumerate(caches):
        if not os.path.exists(cd+"meta.npy"): continue
        t=time.time(); fd=run_file(cd)
        if FEATS is None: FEATS=[k for k in fd.keys() if k not in ("precursor_idx","rank")]
        pid=np.asarray(fd["precursor_idx"]).astype(int); sc=np.asarray(fd["score"],float)
        best={}
        for i,p in enumerate(pid):
            if p not in best or sc[i]>sc[best[p]]: best[p]=i
        ri=np.array(sorted(best.values()))
        X=np.nan_to_num(np.stack([np.asarray(fd[k],float)[ri] for k in FEATS],axis=1),nan=0.,posinf=0.,neginf=0.)
        rows_X.append(X); rows_pid.append(pid[ri]); rows_y.append(decoy[pid[ri]]); rows_fidx.append(np.full(len(ri),fi))
        print(f"  file {fi}: {len(ri)} rows {round(time.time()-t,1)}s",flush=True)
    Xall=np.concatenate(rows_X); pidall=np.concatenate(rows_pid); yall=np.concatenate(rows_y); fileall=np.concatenate(rows_fidx)
    np.savez(FEATCACHE, X=Xall, pid=pidall, y=yall, fidx=fileall, feats=np.array(FEATS))
    print("saved feat cache", Xall.shape, flush=True)

nfeat=Xall.shape[1]
# --- Radiant NN: 3 FC layers, hidden width = nfeat//2, ReLU, BCE, Percolator k-fold ---
# Pre-rank with LDA to define the training set (PSMs passing 50% FDR OR top 50k, whichever larger)
sc=StandardScaler().fit(Xall); Xs=sc.transform(Xall)
lda=LinearDiscriminantAnalysis().fit(Xs,yall)
lda_score=-lda.decision_function(Xs)
q_lda=tdc(lda_score,yall)
train_mask = (q_lda<=0.5) | (np.argsort(np.argsort(-lda_score)) < 50000)
# ensure both classes present
print(f"NN train set: {int(train_mask.sum())} PSMs (T {int((yall[train_mask]==0).sum())} D {int((yall[train_mask]==1).sum())})", flush=True)

hidden=max(nfeat//2,8)
oof=np.zeros(len(yall))
skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=0)
folds=list(skf.split(Xs,yall))
for k,(tr,te) in enumerate(folds):
    tr_use=tr[train_mask[tr]]  # train only on 50%FDR/top50k within this fold's train split
    clf=MLPClassifier(hidden_layer_sizes=(hidden,hidden,hidden),activation="relu",
                      alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10,random_state=0)
    clf.fit(Xs[tr_use],yall[tr_use])
    # score the held-out fold with a network NOT trained on it (Percolator-style)
    oof[te]=clf.predict_proba(Xs[te])[:,list(clf.classes_).index(0)]  # prob of target(class 0)
score_t=oof
# global cross-file best per precursor + recompute FDR on NN scores
bestp={}
for i in range(len(pidall)):
    p=pidall[i]
    if p not in bestp or score_t[i]>bestp[p][0]: bestp[p]=(score_t[i], yall[i])
items=sorted(bestp.items(),key=lambda kv:-kv[1][0])
ps=np.array([k for k,_ in items]); ss=np.array([v[0] for _,v in items]); ys=np.array([v[1] for _,v in items])
q=tdc(ss,ys)
for qt in [0.01]:
    sel=(ys==0)&(q<=qt); prec=ps[sel]
    pgs={}; 
    for pid_ in prec:
        pg=target_pid_to_pg.get(int(pid_))
        if pg is not None: pgs[pg]=pgs.get(pg,0)+1
    n_multi=sum(1 for pg,c in pgs.items() if c>=2)
    n_single=sum(1 for pg,c in pgs.items() if c==1)
    print(f"[NN GLOBAL q<={qt}] precursors={int(sel.sum())} PG={len(pgs)} multiPG={n_multi} singlePG={n_single}", flush=True)
print("  vs DIA-NN 1306 PG/17770 pr ; FragPipe 1337 PG", flush=True)
