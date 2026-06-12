"""Use cached entrap features to find the entrapment-calibrated threshold where
EMPIRICAL (yeast) FDR = 1%, and report dog precursors + PG there (the honest
deepest depth). Also report at the decoy-q 1% point for comparison.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from search_pipeline import tdc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

Z=np.load("/quobyte/proteomics-grp/brett/glendon/feat16_entrap.npz",allow_pickle=True)
Xall=Z["X"];pidall=Z["pid"];yall=Z["y"];FEATS=list(Z["feats"])
td=dict(np.load("/quobyte/proteomics-grp/brett/glendon/td_entrap.npz"))
decoy=td["decoy"];species=td["species"]
nfeat=Xall.shape[1];scaler=StandardScaler().fit(Xall);Xs=scaler.transform(Xall)
lda=LinearDiscriminantAnalysis().fit(Xs,yall);ls=-lda.decision_function(Xs);ql=tdc(ls,yall)
tm=(ql<=0.5)|(np.argsort(np.argsort(-ls))<50000)
hidden=max(nfeat//2,8);oof=np.zeros(len(yall))
for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(Xs,yall):
    tu=tr[tm[tr]]
    clf=MLPClassifier((hidden,hidden,hidden),activation="relu",alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=10,random_state=0)
    clf.fit(Xs[tu],yall[tu]);oof[te]=clf.predict_proba(Xs[te])[:,list(clf.classes_).index(0)]
score_t=oof
# best per precursor across files
bestp={}
for i in range(len(pidall)):
    p=pidall[i]
    if p not in bestp or score_t[i]>bestp[p][0]:bestp[p]=(score_t[i],yall[i])
items=sorted(bestp.items(),key=lambda kv:-kv[1][0])
ps=np.array([k for k,_ in items]);ss=np.array([v[0] for _,v in items]);ys=np.array([v[1] for _,v in items])
spp=species[ps.astype(int)]
q=tdc(ss,ys)
dogTDB=int(((species==0)&(decoy==0)).sum());yeastTDB=int(((species==1)&(decoy==0)).sum());ratio=dogTDB/max(yeastTDB,1)
# sweep over target precursors ranked by score; empirical FDR from yeast among targets
istgt=(ys==0)
order=np.argsort(-ss)
cum_dog=0;cum_yeast=0
best_at_1pct=None
rows=[]
for idx in order:
    if ys[idx]!=0: continue  # only targets count toward IDs
    if spp[idx]==0: cum_dog+=1
    else: cum_yeast+=1
    emp=(cum_yeast*ratio)/max(cum_dog,1)
    rows.append((cum_dog,cum_yeast,emp,ss[idx]))
    if emp<=0.01: best_at_1pct=(cum_dog,cum_yeast,emp,ss[idx])
# best_at_1pct = deepest point with empirical FDR <= 1%
if best_at_1pct:
    nd,ny,ef,thr=best_at_1pct
    print(f"[ENTRAP-CALIBRATED 1% empirical] dog_precursors={nd} yeast={ny} empFDR={ef*100:.2f}% thr={thr:.4f}")
    # PG rollup at this threshold
    lib_df=pd.read_parquet("/quobyte/proteomics-grp/brett/glendon/diann251_clean16/report-lib.parquet")
    lib_df=lib_df[lib_df["Decoy"]==0];lz=lib_df[lib_df["Fragment.Charge"]==1]
    keys=list(lz.groupby("Precursor.Id",sort=False).groups.keys())
    gg=lz.groupby("Precursor.Id",sort=False)["Protein.Group"].first();cnt=lz.groupby("Precursor.Id",sort=False).size()
    pidpg={};tp=0
    for k in keys:
        if cnt[k]>=4: pidpg[tp]=gg[k];tp+=1
    sel=istgt&(spp==0)&(ss>=thr)
    prec=ps[sel]
    pgc={}
    for pid_ in prec:
        pg=pidpg.get(int(pid_))
        if pg is not None: pgc[pg]=pgc.get(pg,0)+1
    nm=sum(1 for pg,c in pgc.items() if c>=2);nsg=sum(1 for pg,c in pgc.items() if c==1)
    print(f"[ENTRAP-CALIBRATED] dog precursors={int(sel.sum())} PG={len(pgc)} multiPG={nm} singlePG={nsg}")
    print(f"  vs DIA-NN 1306 PG/17770 pr ; FragPipe 1337 PG  (ratio dogTDB/yeastTDB={ratio:.2f})")
