import sys,numpy as np,pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from fgt8_lib import *
rows=[]
for axis,vals in [('entangle',[0,.25,.5,.75,1.0]),('horizon',[2,4,8,12,16]),('transition_noise',[0,.1,.2,.3,.4])]:
 for val in vals:
  for seed in range(9301,9311):
   reg=Regime8(transfer=.5,entangle=.25,noise=.15,horizon=8,diversity=.7,transition_noise=.08)
   setattr(reg,axis,val)
   for level in ['global','capability','family','failure']:
    rng=np.random.default_rng(seed*71+len(level)+int(float(val)*100 if axis!='horizon' else val)); a=Agent8('linear',rng)
    for _ in range(80):
     if level=='global': t=make_task8(rng,reg)
     elif level=='capability': t=make_task8(rng,reg,'capability',0)
     elif level=='family': t=make_task8(rng,reg,'family',1)
     else: t=make_task8(rng,reg,'failure',3)
     a.train(t,reg,lr=.024)
    s,_=eval8(a,reg,np.random.default_rng(seed+131),300,False,prevalence=.5,target=3)
    rows.append({'axis':axis,'value':val,'seed':seed,'level':level,'success':s})
pd.DataFrame(rows).to_csv(str(ROOT/'results'/'fgt8_stress_axes.csv'),index=False); print('STRESS8',len(rows))
