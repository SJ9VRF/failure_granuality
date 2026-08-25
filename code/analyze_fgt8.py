import glob,math,os,json
import numpy as np,pandas as pd
from scipy.stats import ttest_rel,wilcoxon,spearmanr
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=str(ROOT/'results'); FIG=str(ROOT/'figures'); os.makedirs(FIG,exist_ok=True)
# main
main=pd.read_csv(f'{OUT}/fgt8_main.csv'); final=main[main['round']==3]
ms=final.groupby(['kind','method'])[['heldout','external']].agg(['mean','std','count']).reset_index(); ms.columns=['_'.join(x).rstrip('_') for x in ms.columns]; ms.to_csv(f'{OUT}/fgt8_main_summary.csv',index=False)
# overall summary and AUC
ov=final.groupby('method')[['heldout','external']].agg(['mean','std','count']).reset_index(); ov.columns=['_'.join(x).rstrip('_') for x in ov.columns]; ov.to_csv(f'{OUT}/fgt8_main_overall.csv',index=False)
auc=[]
for (seed,kind,method),g in main.groupby(['seed','kind','method']):
 g=g.sort_values('round'); auc.append({'seed':seed,'kind':kind,'method':method,'heldout_auc':np.trapz(g.heldout,g['round'])/3,'external_auc':np.trapz(g.external,g['round'])/3})
pd.DataFrame(auc).to_csv(f'{OUT}/fgt8_main_auc.csv',index=False)
# tax stats
tax=pd.read_csv(f'{OUT}/fgt8_tax_100.csv'); tsum=tax.groupby('level')[['support_entropy','coverage','state_repair','heldout','external','target_success','all_skill_gain']].agg(['mean','std']).reset_index(); tsum.columns=['_'.join(x).rstrip('_') for x in tsum.columns]; tsum.to_csv(f'{OUT}/fgt8_tax_summary.csv',index=False)
piv=tax.pivot(index='seed',columns='level'); stats=[]
for metric in ['support_entropy','coverage','state_repair','heldout','external','target_success','all_skill_gain']:
 for a,b in [('failure','global'),('capability','global'),('family','global'),('failure','family')]:
  d=(piv[metric][a]-piv[metric][b]).values; rng=np.random.default_rng(81); boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)]); lo,hi=np.percentile(boots,[2.5,97.5])
  stats.append({'metric':metric,'comparison':f'{a}-{b}','delta':d.mean(),'ci_lo':lo,'ci_hi':hi,'p_t':ttest_rel(piv[metric][a],piv[metric][b]).pvalue})
pd.DataFrame(stats).to_csv(f'{OUT}/fgt8_tax_stats.csv',index=False)
# empirical phase
phase=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/fgt8_phase_n*_b*.csv'))],ignore_index=True); phase.to_csv(f'{OUT}/fgt8_phase_raw.csv',index=False)
ps=phase.groupby(['noise','budget','transfer','prevalence','level']).success.agg(['mean','std','count']).reset_index(); ps.to_csv(f'{OUT}/fgt8_phase_summary.csv',index=False)
w=[]
for keys,g in ps.groupby(['noise','budget','transfer','prevalence']):
 gg=g.sort_values('mean',ascending=False); w.append(dict(zip(['noise','budget','transfer','prevalence'],keys))|{'winner':gg.iloc[0].level,'winner_mean':gg.iloc[0]['mean'],'runner_up':gg.iloc[1].level,'margin':gg.iloc[0]['mean']-gg.iloc[1]['mean']})
w=pd.DataFrame(w); w.to_csv(f'{OUT}/fgt8_phase_winners.csv',index=False)
wf=w.groupby(['noise','budget','winner']).size().rename('cells').reset_index(); wf['fraction']=wf.groupby(['noise','budget']).cells.transform(lambda x:x/x.sum()); wf.to_csv(f'{OUT}/fgt8_phase_win_fractions.csv',index=False)
# phase plot noise .2
levels=['global','capability','family','failure']; code={l:i for i,l in enumerate(levels)}
fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
for ax,b in zip(axes,[40,80,160]):
 d=w[(w.noise==.2)&(w.budget==b)]; arr=np.full((4,5),np.nan)
 for _,r in d.iterrows(): arr[[.25,.5,.75,1].index(r.prevalence),[0,.25,.5,.75,1].index(r.transfer)]=code[r.winner]
 im=ax.imshow(arr,aspect='auto',origin='lower',vmin=0,vmax=3); ax.set_xticks(range(5),[0,.25,.5,.75,1]); ax.set_yticks(range(4),[.25,.5,.75,1]); ax.set_xlabel('Cross-skill transfer'); ax.set_title(f'Budget={b}')
axes[0].set_ylabel('Target prevalence'); cb=fig.colorbar(im,ax=axes.ravel().tolist(),ticks=range(4),fraction=.025,pad=.04); cb.ax.set_yticklabels(levels); fig.suptitle('GranularityBench-H8: empirical optimal granularity (noise=0.2)'); fig.savefig(f'{FIG}/fig_h8_phase.png',dpi=200,bbox_inches='tight'); plt.close(fig)
# tax plot
s=tax.groupby('level').mean(numeric_only=True).loc[levels]
fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(s.support_entropy,s.state_repair,s=90)
for k,r in s.iterrows(): ax.annotate(k,(r.support_entropy,r.state_repair),xytext=(5,5),textcoords='offset points')
ax.set_xlabel('Normalized experience-support entropy'); ax.set_ylabel('Targeted state repair'); ax.set_title('Failure Granularity Tax (proper 8-leaf hierarchy, 100 seeds)'); fig.tight_layout(); fig.savefig(f'{FIG}/fig_h8_tax.png',dpi=200); plt.close(fig)
# target vs broad
fig,ax=plt.subplots(figsize=(7,5)); ax.plot(range(4),s.target_success,marker='o',label='target-only evaluation'); ax.plot(range(4),s.heldout,marker='o',label='broad held-out'); ax.set_xticks(range(4),levels); ax.set_ylabel('Success'); ax.set_title('Specificity helps local target but hurts broad transfer'); ax.legend(); fig.tight_layout(); fig.savefig(f'{FIG}/fig_h8_local_global.png',dpi=200); plt.close(fig)
print('ANALYZE8_DONE')
