import sys,math,numpy as np,pandas as pd
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from fgt8_lib import *
levels=['global','capability','family','failure']; masks_all=[tuple(int((m>>i)&1) for i in range(8)) for m in range(1,256)]
def vec(c):
 v=np.array([c.get(m,0) for m in masks_all],float); return v/max(1,v.sum())
rows=[]
for seed in range(9401,9441):
 reg=Regime8(transfer=.5,entangle=.25,noise=0,horizon=8,diversity=.7,transition_noise=.05); tmp={}
 for level in levels:
  rng=np.random.default_rng(seed*73+len(level)); a=Agent8('linear',rng); b=a.skill.copy(); masks=[]
  for _ in range(100):
   if level=='global': t=make_task8(rng,reg)
   elif level=='capability': t=make_task8(rng,reg,'capability',0)
   elif level=='family': t=make_task8(rng,reg,'family',1)
   else:t=make_task8(rng,reg,'failure',3)
   masks.append(tuple(t['req'].astype(int)));a.train(t,reg,lr=.024)
  tmp[level]=(a,b,Counter(masks))
 p=vec(tmp['global'][2])
 for level in levels:
  a,b,c=tmp[level];q=vec(c); nz=q[q>0];H=-(nz*np.log(nz)).sum(); Hn=H/math.log(255); eff=np.exp(H)/255;cov=len(c)/255;m=.5*(p+q);eps=1e-12
  js=.5*(np.sum(np.where(p>0,p*np.log((p+eps)/(m+eps)),0))+np.sum(np.where(q>0,q*np.log((q+eps)/(m+eps)),0)))/math.log(2)
  broad,_=eval8(a,reg,np.random.default_rng(seed+2),400,False,prevalence=.25,target=3)
  rows.append({'seed':seed,'level':level,'entropy_norm':Hn,'effective_support_norm':eff,'coverage':cov,'js_from_global':js,'state_repair':a.skill[3]-b[3],'broad_eval':broad})
pd.DataFrame(rows).to_csv(str(ROOT/'results'/'fgt8_support_alt.csv'),index=False);print('SUP8',len(rows))
