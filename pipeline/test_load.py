# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_io as io, config as C

frames = io.load_all()
reg = io.build_brain_registry(frames)
print("brain metrics:", reg.shape, reg["klass"].value_counts().to_dict())
master, wlog = io.build_master(frames)
print("master:", master.shape)
print("age notna:", master["age"].notna().sum(), "gender notna:", master["gender"].notna().sum())
dcols = io.disease_columns(frames)
print("n disease cols:", len(dcols))
dc = io.disease_counts(master, dcols)
print(dc.head(40).to_string())
inc = io.included_diseases(master, dcols)
print("included (>=50):", len(inc), inc)
# sample sizes per module
import pandas as pd
gapc = [C.gap_col(o) for o in C.ORGANS]
has_gap = master[gapc].notna().all(axis=1)
print("has all organ gaps:", has_gap.sum())
print("gap+demo:", (has_gap & master["age"].notna() & master["gender"].notna()).sum())
print("gap+BrainSegVol:", (has_gap & master["BrainSegVol"].notna()).sum())
print("gap+eTIV:", (has_gap & master["eTIV"].notna()).sum())
sub0 = [c for c in master.columns if "Thalamus-Proper (left" in c][0]
print("gap+subcort:", (has_gap & master[sub0].notna()).sum())
print("gap+FluidIntelligence:", (has_gap & master["FluidIntelligence"].notna()).sum())
print(wlog.to_string())
