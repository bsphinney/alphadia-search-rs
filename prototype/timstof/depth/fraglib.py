"""Generate b/y fragment m/z from peptide sequence + build pseudo-reverse decoys.
Used to make a proper target+decoy flat library from the DIA-NN MBR library
(which gives us the empirical sequences, RT(iRT), IM, and precursor charge).
"""
import numpy as np

# monoisotopic residue masses (Da)
AA = {
 'G':57.02146,'A':71.03711,'S':87.03203,'P':97.05276,'V':99.06841,
 'T':101.04768,'C':103.00919,'L':113.08406,'I':113.08406,'N':114.04293,
 'D':115.02694,'Q':128.05858,'K':128.09496,'E':129.04259,'M':131.04049,
 'H':137.05891,'F':147.06841,'R':156.10111,'Y':163.06333,'W':186.07931,
}
H2O=18.010565; PROTON=1.007276; NH3=17.026549; CO=27.994915

def fragments(seq, max_charge=1, mods=None):
    """Return list of (mz, ftype_code(0=b,121=y), series_number, charge) for b/y ions.
    mods: optional dict pos->massdelta (0-based residue). Carbamidomethyl C handled by caller via mods."""
    n=len(seq)
    masses=[AA.get(a,0.0) for a in seq]
    if mods:
        for pos,dm in mods.items():
            if 0<=pos<n: masses[pos]+=dm
    # cumulative for b ions (N-term), y ions (C-term)
    frags=[]
    bsum=0.0
    for i in range(n-1):  # b1..b(n-1)
        bsum+=masses[i]
        for z in range(1,max_charge+1):
            mz=(bsum + z*PROTON)/z
            frags.append((mz,0,i+1,z))
    ysum=0.0
    for i in range(n-1):  # y1..y(n-1)
        ysum+=masses[n-1-i]
        for z in range(1,max_charge+1):
            mz=(ysum + H2O + z*PROTON)/z
            frags.append((mz,121,i+1,z))
    return frags

def pseudo_reverse(seq):
    """Pseudo-reverse decoy: reverse all but the C-terminal residue (Pyteomics/Sage style)."""
    if len(seq)<2: return seq
    return seq[:-1][::-1] + seq[-1]
