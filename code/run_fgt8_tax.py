import sys,math,numpy as np,pandas as pd
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from fgt8_lib import *
levels=['global','capability','family','failure']; rows=[]
for seed in range(9101,9201):
 reg=Regime8(transfer=.5,entangle=.25,noise=0,horizon=8,diversity=.7,transition_noise=.05)
 for level in levels:
  rng=np.random.default_rng(seed*43+len(level)); a=Agent8('linear',rng); before=a.skill.copy(); masks=[]
  for _ in range(100):
   if level=='global': t=make_task8(rng,reg)
   elif level=='capability': t=make_task8(rng,reg,'capability',0) # state is in cap0
   elif level=='family': t=make_task8(rng,reg,'family',1)       # order/state family
   else: t=make_task8(rng,reg,'failure',3)
   masks.append(tuple(t['req'].astype(int))); a.train(t,reg,lr=.024)
  c=Counter(masks); p=np.array(list(c.values()),float); p/=p.sum(); H=-(p*np.log(p+1e-12)).sum()/math.log(2**8-1)
  h,_=eval8(a,reg,np.random.default_rng(seed+11),500,False); x,_=eval8(a,reg,np.random.default_rng(seed+12),500,True)
  target,_=eval8(a,reg,np.random.default_rng(seed+13),500,False,prevalence=1.0,target=3)
  rows.append({'seed':seed,'level':level,'support_entropy':H,'coverage':len(c)/(2**8-1),'state_repair':a.skill[3]-before[3],'heldout':h,'external':x,'target_success':target,'all_skill_gain':(a.skill-before).mean()})
pd.DataFrame(rows).to_csv(str(ROOT/'results'/'fgt8_tax_100.csv'),index=False); print('TAX8',len(rows))
