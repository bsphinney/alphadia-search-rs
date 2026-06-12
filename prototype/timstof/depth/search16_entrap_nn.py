"""16-file entrapment audit with the Radiant NN: dog + yeast entrapment library,
NN classifier, global FDR -> count yeast IDs in the 1%-FDR set = empirical FDR.
"""
import numpy as np, sys, time, os, glob
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from search_pipeline import build_ng, tdc
from alphadia_search_rs import (SpecLibFlat as NGSpecLibFlat, PeakGroupSelection,
    SelectionParameters, PeakGroupScoring, ScoringParameters)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

td=dict(np.load("/quobyte/proteomics-grp/brett/glendon/td_entrap.npz"))
decoy=td["decoy"]; species=td["species"]
FEATCACHE="/quobyte/proteomics-grp/brett/glendon/feat16_entrap.npz"

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
    Z=np.load(FEATCACHE,allow_pickle=True);Xall=Z["X"];pidall=Z["pid"];yall=Z["y"];FEATS=list(Z["feats"])
    print("loaded",Xall.shape,flush=True)
else:
    caches=sorted(glob.glob("/quobyte/proteomics-grp/brett/glendon/cache16/*/"));FEATS=None
    rX=[];rp=[];ry=[]
    for fi,cd in enumerate(caches):
        if not os.path.exists(cd+"meta.npy"):continue
        t=time.time();fd=run_file(cd)
        if FEATS is None:FEATS=[k for k in fd.keys() if k not in ("precursor_idx","rank")]
        pid=np.asarray(fd["precursor_idx"]).astype(int);sc=np.asarray(fd["score"],float)
        best={}
        for i,p in enumerate(pid):
            if p not in best or sc[i]>sc[best[p]]:best[p]=i
        ri=np.array(sorted(best.values()))
        X=np.nan_to_num(np.stack([np.asarray(fd[k],float)[ri] for k in FEATS],axis=1),nan=0.,posinf=0.,neginf=0.)
        rX.append(X);rp.append(pid[ri]);ry.append(decoy[pid[ri]])
        print(f" file {fi}: {len(ri)} {round(time.time()-t,1)}s",flush=True)
    Xall=np.concatenate(rX);pidall=np.concatenate(rp);yall=np.concatenate(ry)
    np.savez(FEATCACHE,X=Xall,pid=pidall,y=yall,feats=np.array(FEATS));print("saved",Xall.shape,flush=True)

nfeat=Xall.shape[1];scaler=StandardScaler().fit(Xall);Xs=scaler.transform(Xall)
lda=LinearDiscriminantAnalysis().fit(Xs,yall);lda_s=-lda.decision_function(Xs);ql=tdc(lda_s,yall)
tm=(ql<=0.5)|(np.argsort(np.argsort(-lda_s))<50000)
hidden=max(nfeat//2,8);oof=np.zeros(len(yall))
for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(Xs,yall):
    tu=tr[tm[tr]]
    clf=MLPClassifier((hidden,hidden,hidden),activation="relu",alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10,random_state=0)
    clf.fit(Xs[tu],yall[tu]);oof[te]=clf.predict_proba(Xs[te])[:,list(clf.classes_).index(0)]
score_t=oof
bestp={}
for i in range(len(pidall)):
    p=pidall[i]
    if p not in bestp or score_t[i]>bestp[p][0]:bestp[p]=(score_t[i],yall[i])
items=sorted(bestp.items(),key=lambda kv:-kv[1][0])
ps=np.array([k for k,_ in items]);ss=np.array([v[0] for _,v in items]);ys=np.array([v[1] for _,v in items])
q=tdc(ss,ys)
dogTDB=int(((species==0)&(decoy==0)).sum());yeastTDB=int(((species==1)&(decoy==0)).sum());ratio=dogTDB/max(yeastTDB,1)
for qt in [0.001,0.005,0.01,0.02]:
    sel=(ys==0)&(q<=qt);prec=ps[sel];spb=species[prec.astype(int)]
    nd=int((spb==0).sum());ny=int((spb==1).sum());ef=(ny*ratio)/max(nd,1)
    print(f"[NN ENTRAP q<={qt}] dog={nd} yeast={ny} EMPIRICAL_FDR={ef*100:.2f}% (reported {qt*100:.1f}%)",flush=True)
print(f"  dogTDB={dogTDB} yeastTDB={yeastTDB} ratio={ratio:.2f}",flush=True)
