import sys,numpy as np,pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from fgt8_lib import *
noise=float(sys.argv[1]); budget=int(sys.argv[2]); rows=[]
for tr in [0,.25,.5,.75,1.0]:
 for prev in [.25,.5,.75,1.0]:
  for seed in range(9201,9207):
   reg=Regime8(transfer=tr,entangle=.25,noise=noise,horizon=8,diversity=.7,transition_noise=.08)
   for level in ['global','capability','family','failure']:
    rng=np.random.default_rng(seed*59+budget+len(level)+int(tr*1000)+int(noise*100)); a=Agent8('linear',rng)
    for _ in range(budget):
     # diagnostic noise routes to wrong target at each abstraction level
     tgt=3 if rng.random()>=noise else int(rng.integers(0,8))
     if level=='global': t=make_task8(rng,reg)
     elif level=='capability': t=make_task8(rng,reg,'capability',CAP_OF[tgt])
     elif level=='family': t=make_task8(rng,reg,'family',FAM_OF[tgt])
     else: t=make_task8(rng,reg,'failure',tgt)
     a.train(t,reg,lr=.024)
    s,_=eval8(a,reg,np.random.default_rng(seed*61+budget+int(prev*100)),300,False,prevalence=prev,target=3)
    rows.append({'seed':seed,'transfer':tr,'prevalence':prev,'budget':budget,'noise':noise,'level':level,'success':s})
pd.DataFrame(rows).to_csv(str(ROOT/'results'/f'fgt8_phase_n{int(noise*10):02d}_b{budget}.csv'),index=False); print('DONE',noise,budget,len(rows))
