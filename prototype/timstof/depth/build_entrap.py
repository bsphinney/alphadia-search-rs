"""Build an ENTRAPMENT target+decoy library: dog targets (real, from td_lib) PLUS
yeast entrapment targets (tryptic peptides from yeast proteome, foreign to dog).
Yeast precursors get plausible IM/RT drawn from the dog distribution so they ARE
searchable. After search+FDR, yeast IDs in the 1%-FDR set estimate the TRUE FDR
(entrapment method, paired-DB ratio scaling).
"""
import numpy as np, pandas as pd, re, sys
sys.path.insert(0,"/quobyte/proteomics-grp/brett/glendon")
from fraglib import fragments, pseudo_reverse, AA, PROTON, H2O

def digest(seq, missed=1, lo=7, hi=30):
    # trypsin: cut after K/R not before P
    sites=[0]+[m.end() for m in re.finditer(r'[KR](?!P)', seq)]+[len(seq)]
    sites=sorted(set(sites))
    peps=set()
    for i in range(len(sites)-1):
        for j in range(i+1, min(i+2+missed, len(sites))):
            p=seq[sites[i]:sites[j]]
            if lo<=len(p)<=hi and all(a in AA for a in p):
                peps.add(p)
    return peps

def pep_mz(seq, charge, cmod=57.021464):
    m=sum(AA[a] for a in seq)+H2O + sum(cmod for a in seq if a=='C')
    return (m + charge*PROTON)/charge

def main(n_entrap=20000):
    # read yeast
    seqs=[]
    cur=[]
    for line in open("/quobyte/proteomics-grp/brett/glendon/yeast.fasta"):
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur=[]
        else: cur.append(line.strip())
    if cur: seqs.append("".join(cur))
    print("yeast proteins", len(seqs))
    peps=set()
    for s in seqs:
        peps|=digest(s)
        if len(peps)>n_entrap*3: break
    peps=list(peps)
    print("yeast tryptic peptides", len(peps))
    # dog IM/RT distribution to sample from
    td=dict(np.load("/quobyte/proteomics-grp/brett/glendon/td_lib.npz"))
    dog_t = td["decoy"]==0
    dog_im=td["im"][dog_t]; dog_rt=td["rt"][dog_t]
    rng=np.random.default_rng(0)
    # build entrapment precursors (charge 2 or 3), mz in 300-1200
    ent=[]
    for p in peps:
        for z in (2,3):
            mz=pep_mz(p,z)
            if 300<=mz<=1200:
                ent.append((p,z,mz))
    rng.shuffle(ent); ent=ent[:n_entrap]
    print("entrapment precursors", len(ent))
    # assemble combined library: dog targets+decoys (from td_lib) + yeast entrap targets + yeast decoys
    # We append yeast as new precursor_idx after dog's max.
    base=int(td["precursor_idx"].max())+1
    pidx=list(td["precursor_idx"]); pmz=list(td["precursor_mz"]); rt=list(td["rt"]); im=list(td["im"])
    naa=list(td["naa"]); decoy=list(td["decoy"]); spec=list(np.zeros(len(td["precursor_idx"]),np.int8))  # species 0=dog
    fstart=list(td["fstart"]); fstop=list(td["fstop"])
    fmz=[td["fmz"]]; fint=[td["fint"]]; ftype=[td["ftype"]]; fnum=[td["fnum"]]
    cur=int(td["fstop"][-1])
    pid=base
    for (p,z,mz) in ent:
        cm={i:57.021464 for i,a in enumerate(p) if a=='C'}
        gen=[(m,t,n) for (m,t,n,zz) in fragments(p,1,cm) if zz==1 and 200<=m<=1800]
        if len(gen)<4: continue
        gen=sorted(gen,key=lambda x:-x[0])[:12]
        mzr=np.array([g[0] for g in gen],np.float32); tcr=np.array([g[1] for g in gen],np.uint8); nur=np.array([g[2] for g in gen],np.uint8)
        inr=np.ones(len(mzr),np.float32)
        rim=float(rng.choice(dog_im)); rrt=float(rng.choice(dog_rt))
        # entrap TARGET
        pidx.append(pid);pmz.append(float(mz));rt.append(rrt);im.append(rim);naa.append(len(p));decoy.append(0);spec.append(1)
        fstart.append(cur);fstop.append(cur+len(mzr));fmz.append(mzr);fint.append(inr);ftype.append(tcr);fnum.append(nur);cur+=len(mzr);pid+=1
        # entrap DECOY (pseudo-reverse)
        dp=pseudo_reverse(p); cm2={i:57.021464 for i,a in enumerate(dp) if a=='C'}
        gen2=[(m,t,n) for (m,t,n,zz) in fragments(dp,1,cm2) if zz==1 and 200<=m<=1800]
        gen2=sorted(gen2,key=lambda x:-x[0])[:len(mzr)]
        if len(gen2)<4: gen2=[(g[0]+13.0,g[1],g[2]) for g in gen]
        dmz=np.array([g[0] for g in gen2],np.float32); dtc=np.array([g[1] for g in gen2],np.uint8); dnu=np.array([g[2] for g in gen2],np.uint8)
        din=np.ones(len(dmz),np.float32)
        pidx.append(pid);pmz.append(float(mz));rt.append(rrt);im.append(rim);naa.append(len(p));decoy.append(1);spec.append(1)
        fstart.append(cur);fstop.append(cur+len(dmz));fmz.append(dmz);fint.append(din);ftype.append(dtc);fnum.append(dnu);cur+=len(dmz);pid+=1
    fmz=np.concatenate(fmz).astype(np.float32);fint=np.concatenate(fint).astype(np.float32)
    ftype=np.concatenate(ftype).astype(np.uint8);fnum=np.concatenate(fnum).astype(np.uint8)
    np.savez("/quobyte/proteomics-grp/brett/glendon/td_entrap",
        precursor_idx=np.array(pidx,np.uint64),precursor_mz=np.array(pmz,np.float32),
        rt=np.array(rt,np.float32),im=np.array(im,np.float32),naa=np.array(naa,np.uint8),
        decoy=np.array(decoy,np.int8),species=np.array(spec,np.int8),
        fstart=np.array(fstart,np.uint64),fstop=np.array(fstop,np.uint64),
        fmz=fmz,fint=fint,ftype=ftype,fnum=fnum)
    sp=np.array(spec)
    print(f"combined lib: dog_T+D={int((sp==0).sum())} entrap_T+D={int((sp==1).sum())} total_prec={len(pidx)}")

if __name__=="__main__": main(int(sys.argv[1]) if len(sys.argv)>1 else 20000)
