import pandas as pd,numpy as np
from scipy.stats import ttest_rel,wilcoxon
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=str(ROOT/'results');df=pd.read_csv(f'{OUT}/fgt8_main.csv');final=df[df['round']==3]
rows=[]
for kind in final.kind.unique():
 p=final[final.kind==kind].pivot(index='seed',columns='method',values='heldout')
 for a,b in [('difficulty','uniform'),('difficulty','capability'),('difficulty','family'),('difficulty','failure'),('difficulty','random_gran'),('capability','failure'),('family','failure')]:
  d=(p[a]-p[b]).values; rng=np.random.default_rng(77);boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(10000)]);lo,hi=np.percentile(boots,[2.5,97.5])
  rows.append({'kind':kind,'comparison':f'{a}-{b}','delta_pp':100*d.mean(),'ci_lo_pp':100*lo,'ci_hi_pp':100*hi,'p_t':ttest_rel(p[a],p[b]).pvalue,'p_wilcoxon':wilcoxon(p[a],p[b]).pvalue if np.any(d!=0) else 1.0})
st=pd.DataFrame(rows)
# Holm p_t
order=np.argsort(st.p_t.values);m=len(st);adj=np.empty(m);running=0
for rank,idx in enumerate(order):
 v=min(1,st.p_t.iloc[idx]*(m-rank));running=max(running,v);adj[idx]=running
st['p_holm']=adj;st.to_csv(f'{OUT}/fgt8_main_stats.csv',index=False)
print(st.to_string(index=False))
