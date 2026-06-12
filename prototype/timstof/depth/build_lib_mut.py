"""Build target+MUTATED-decoy library (Radiant recipe): mutate 2nd AND penultimate
residue via the fixed substitution map, adjust b/y fragment masses accordingly.
Better-calibrated decoys than pseudo-reverse.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from fraglib import fragments, AA

# Radiant mutation map: ACDEFGHIKLMNPQRSTUVWY -> LSEDLLSVLVLQLNLTSSLLS
SRC="ACDEFGHIKLMNPQRSTUVWY"
DST="LSEDLLSVLVLQLNLTSSLLS"
MUT={s:d for s,d in zip(SRC,DST)}

def mutate(seq):
    if len(seq)<4: return seq
    s=list(seq)
    s[1]=MUT.get(s[1],s[1])           # 2nd residue
    s[-2]=MUT.get(s[-2],s[-2])        # penultimate
    return "".join(s)

def build(out, max_prec=None):
    df=pd.read_parquet("/quobyte/proteomics-grp/brett/glendon/diann251_clean16/report-lib.parquet")
    df=df[(df["Decoy"]==0)&(df["Fragment.Charge"]==1)].reset_index(drop=True)
    keys=list(df.groupby("Precursor.Id",sort=False).groups.keys())
    if max_prec: keys=keys[:max_prec]
    groups=df.groupby("Precursor.Id",sort=False).groups
    PM=df["Precursor.Mz"].values;RTv=df["RT"].values;IMv=df["IM"].values
    SEQ=df["Stripped.Sequence"].values;FMZ=df["Product.Mz"].values;FINT=df["Relative.Intensity"].values
    FTY=df["Fragment.Type"].values;FNU=df["Fragment.Series.Number"].values
    ftmap={"b":0,"y":121}
    pidx=[];pmz=[];rt=[];im=[];naa=[];decoy=[];fstart=[];fstop=[];fmz=[];fint=[];ftype=[];fnum=[];cur=0;pid=0
    for k in keys:
        idxs=np.asarray(groups[k]);mzr=FMZ[idxs].astype(np.float32);inr=FINT[idxs].astype(np.float32)
        typ=FTY[idxs];num=FNU[idxs].astype(np.uint8);keep=mzr>0
        if keep.sum()<4: continue
        mzr=mzr[keep];inr=inr[keep];typ=typ[keep];num=num[keep]
        tcode=np.array([ftmap.get(t,0) for t in typ],np.uint8)
        seq=SEQ[idxs[0]];pm=float(PM[idxs[0]]);rtv=float(RTv[idxs[0]]);imval=float(IMv[idxs[0]])
        pidx.append(pid);pmz.append(pm);rt.append(rtv);im.append(imval);naa.append(len(seq));decoy.append(0)
        fstart.append(cur);fstop.append(cur+len(mzr));fmz.append(mzr);fint.append(inr);ftype.append(tcode);fnum.append(num);cur+=len(mzr);pid+=1
        # MUTATED decoy: regenerate b/y z1 masses with carbamidomethyl C
        dseq=mutate(seq);cm={i:57.021464 for i,a in enumerate(dseq) if a=='C'}
        gen=[(m,t,n) for (m,t,n,z) in fragments(dseq,1,cm) if z==1]
        gmap={(t,n):m for m,t,n in gen}
        dmz=[];din=[];dty=[];dnu=[]
        for j in range(len(mzr)):
            t=int(tcode[j]);n=int(num[j]);key2=(0 if t==0 else 121,n)
            if key2 in gmap: dmz.append(gmap[key2]);din.append(float(inr[j]));dty.append(t);dnu.append(n)
        if len(dmz)<4: dmz=(mzr+13.0).tolist();din=inr.tolist();dty=tcode.tolist();dnu=num.tolist()
        dmz=np.array(dmz,np.float32);din=np.array(din,np.float32);dty=np.array(dty,np.uint8);dnu=np.array(dnu,np.uint8)
        pidx.append(pid);pmz.append(pm);rt.append(rtv);im.append(imval);naa.append(len(seq));decoy.append(1)
        fstart.append(cur);fstop.append(cur+len(dmz));fmz.append(dmz);fint.append(din);ftype.append(dty);fnum.append(dnu);cur+=len(dmz);pid+=1
    fmz=np.concatenate(fmz).astype(np.float32);fint=np.concatenate(fint).astype(np.float32)
    ftype=np.concatenate(ftype).astype(np.uint8);fnum=np.concatenate(fnum).astype(np.uint8)
    np.savez(out, precursor_idx=np.array(pidx,np.uint64),precursor_mz=np.array(pmz,np.float32),
        rt=np.array(rt,np.float32),im=np.array(im,np.float32),naa=np.array(naa,np.uint8),
        decoy=np.array(decoy,np.int8),fstart=np.array(fstart,np.uint64),fstop=np.array(fstop,np.uint64),
        fmz=fmz,fint=fint,ftype=ftype,fnum=fnum)
    print(f"built {out}: {len(pidx)} prec (T={sum(1 for d in decoy if d==0)} D={sum(1 for d in decoy if d==1)})")
if __name__=="__main__": build("/quobyte/proteomics-grp/brett/glendon/td_lib_mut.npz")
