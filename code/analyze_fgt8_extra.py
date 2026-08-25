import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=str(ROOT/'results');FIG=str(ROOT/'figures')
# support alt
s=pd.read_csv(f'{OUT}/fgt8_support_alt.csv'); sm=s.groupby('level')[['entropy_norm','effective_support_norm','coverage','js_from_global','state_repair','broad_eval']].agg(['mean','std']).reset_index(); sm.columns=['_'.join(x).rstrip('_') for x in sm.columns];sm.to_csv(f'{OUT}/fgt8_support_alt_summary.csv',index=False)
c=[]
for m in ['entropy_norm','effective_support_norm','coverage','js_from_global']:
 for y in ['state_repair','broad_eval']:
  r,p=spearmanr(s[m],s[y]);c.append({'metric':m,'outcome':y,'rho':r,'p':p})
pd.DataFrame(c).to_csv(f'{OUT}/fgt8_support_alt_correlations.csv',index=False)
# stress
st=pd.read_csv(f'{OUT}/fgt8_stress_axes.csv'); ss=st.groupby(['axis','value','level']).success.agg(['mean','std','count']).reset_index();ss.to_csv(f'{OUT}/fgt8_stress_summary.csv',index=False)
for axis in st.axis.unique():
 d=ss[ss.axis==axis].pivot(index='value',columns='level',values='mean');fig,ax=plt.subplots(figsize=(7.5,4.8))
 for col in d.columns:ax.plot(d.index,d[col],marker='o',label=col)
 ax.set_xlabel(axis);ax.set_ylabel('Success at 50% target prevalence');ax.set_title(f'H8 robustness: {axis}');ax.legend();fig.tight_layout();fig.savefig(f'{FIG}/fig_h8_{axis}.png',dpi=200);plt.close(fig)
# support figure
order=['global','capability','family','failure'];x=s.groupby('level').mean(numeric_only=True).loc[order]
fig,ax=plt.subplots(figsize=(7,5));ax.scatter(x.effective_support_norm,x.state_repair,s=90)
for k,r in x.iterrows():ax.annotate(k,(r.effective_support_norm,r.state_repair),xytext=(5,5),textcoords='offset points')
ax.set_xlabel('Effective support / total support');ax.set_ylabel('State repair');ax.set_title('Tax under alternative support metric (H8)');fig.tight_layout();fig.savefig(f'{FIG}/fig_h8_support_alt.png',dpi=200);plt.close(fig)
print('EXTRA8_DONE')
