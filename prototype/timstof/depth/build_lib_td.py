"""Build a proper target+decoy flat library (numpy arrays) from the DIA-NN MBR
library: targets use real DIA-NN fragments; decoys = pseudo-reverse SEQUENCE
fragments (genuinely different masses). Saves arrays for fast reuse.
Restricts to charge-1 b/y fragments. Carries IM (1/K0) and iRT.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from fraglib import fragments, pseudo_reverse

def build(out_prefix, max_prec=None):
    df=pd.read_parquet("/quobyte/proteomics-grp/brett/glendon/diann251_clean16/report-lib.parquet")
    df=df[df["Decoy"]==0].reset_index(drop=True)
    # only z=1 fragments (the data; keep it simple/robust)
    df=df[df["Fragment.Charge"]==1].reset_index(drop=True)
    gp=df.groupby("Precursor.Id",sort=False)
    keys=list(gp.groups.keys())
    if max_prec: keys=keys[:max_prec]
    pidx=[];pmz=[];rt=[];im=[];naa=[];decoy=[]
    fstart=[];fstop=[];fmz=[];fint=[];ftype=[];fnum=[]
    cur=0;pid=0
    ftmap={"b":0,"y":121}
    rows_by_key=gp.groups
    PM=df["Precursor.Mz"].values; RTv=df["RT"].values; IMv=df["IM"].values
    SEQ=df["Stripped.Sequence"].values; MODSEQ=df["Modified.Sequence"].values
    FMZ=df["Product.Mz"].values; FINT=df["Relative.Intensity"].values
    FTY=df["Fragment.Type"].values; FNU=df["Fragment.Series.Number"].values
    for k in keys:
        idxs=np.asarray(rows_by_key[k])
        mzr=FMZ[idxs].astype(np.float32); inr=FINT[idxs].astype(np.float32)
        typ=FTY[idxs]; num=FNU[idxs].astype(np.uint8)
        keep=mzr>0
        if keep.sum()<4: continue
        mzr=mzr[keep];inr=inr[keep];typ=typ[keep];num=num[keep]
        tcode=np.array([ftmap.get(t,0) for t in typ],np.uint8)
        seq=SEQ[idxs[0]]; modseq=str(MODSEQ[idxs[0]]); pm=float(PM[idxs[0]])
        rtv=float(RTv[idxs[0]]); imval=float(IMv[idxs[0]]); na=len(seq)
        # TARGET (real fragments)
        pidx.append(pid);pmz.append(pm);rt.append(rtv);im.append(imval);naa.append(na);decoy.append(0)
        fstart.append(cur);fstop.append(cur+len(mzr));fmz.append(mzr);fint.append(inr);ftype.append(tcode);fnum.append(num);cur+=len(mzr);pid+=1
        # DECOY: pseudo-reverse sequence, regenerate b/y z=1 masses
        dseq=pseudo_reverse(seq)
        modmap={i:57.021464 for i,a in enumerate(dseq) if a=='C'}
        dgen=fragments(dseq, max_charge=1, mods=modmap)
        # match the target's (type, number) set so decoy has same #frags & intensities
        gmap={(t,n):mz for mz,t,n,z in dgen if z==1}
        dmz=[];din=[];dty=[];dnu=[]
        for j in range(len(mzr)):
            t=int(tcode[j]); n=int(num[j])
            tt='b' if t==0 else 'y'
            key2=(0 if tt=='b' else 121, n)
            if key2 in gmap:
                dmz.append(gmap[key2]); din.append(float(inr[j])); dty.append(t); dnu.append(n)
        if len(dmz)<4: 
            # fallback: shift target fragments by +13.0 (still a null)
            dmz=(mzr+13.0).tolist(); din=inr.tolist(); dty=tcode.tolist(); dnu=num.tolist()
        dmz=np.array(dmz,np.float32);din=np.array(din,np.float32);dty=np.array(dty,np.uint8);dnu=np.array(dnu,np.uint8)
        pidx.append(pid);pmz.append(pm);rt.append(rtv);im.append(imval);naa.append(na);decoy.append(1)
        fstart.append(cur);fstop.append(cur+len(dmz));fmz.append(dmz);fint.append(din);ftype.append(dty);fnum.append(dnu);cur+=len(dmz);pid+=1
    fmz=np.concatenate(fmz).astype(np.float32);fint=np.concatenate(fint).astype(np.float32)
    ftype=np.concatenate(ftype).astype(np.uint8);fnum=np.concatenate(fnum).astype(np.uint8)
    np.savez(out_prefix,
        precursor_idx=np.array(pidx,np.uint64), precursor_mz=np.array(pmz,np.float32),
        rt=np.array(rt,np.float32), im=np.array(im,np.float32), naa=np.array(naa,np.uint8),
        decoy=np.array(decoy,np.int8), fstart=np.array(fstart,np.uint64), fstop=np.array(fstop,np.uint64),
        fmz=fmz, fint=fint, ftype=ftype, fnum=fnum)
    print(f"built {out_prefix}: {len(pidx)} precursors (T={sum(1 for d in decoy if d==0)} D={sum(1 for d in decoy if d==1)}) frags={len(fmz)}")

if __name__=="__main__":
    build("/quobyte/proteomics-grp/brett/glendon/td_lib", max_prec=int(sys.argv[1]) if len(sys.argv)>1 else None)
