import pandas as pd, numpy as np
from pathlib import Path
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; O=ROOT
main=pd.read_csv(R/'fgt8_main_overall.csv').sort_values('heldout_mean',ascending=False)
tax=pd.read_csv(R/'fgt8_tax_summary.csv').set_index('level').loc[['global','capability','family','failure']]
stats=pd.read_csv(R/'fgt8_tax_stats.csv')
win=pd.read_csv(R/'fgt8_phase_win_fractions.csv')
sup=pd.read_csv(R/'fgt8_support_alt_summary.csv').set_index('level').loc[['global','capability','family','failure']]
stress=pd.read_csv(R/'fgt8_stress_summary.csv')

def pct(x): return f'{100*x:.2f}%'
lines=['# Auto-generated Results','', '> Generated directly from CSV outputs. Do not edit numerical values by hand.','', '## Main heterogeneous-distribution study','', '| Method | Held-out | External shift |','|---|---:|---:|']
for _,r in main.iterrows(): lines.append(f"| {r['method']} | {pct(r.heldout_mean)} | {pct(r.external_mean)} |")
lines += ['', '## Direct causal granularity intervention (100 untouched seeds)','', '| Granularity | Support entropy | Coverage | State repair | Target-only success | Broad held-out | External |','|---|---:|---:|---:|---:|---:|---:|']
for lev,r in tax.iterrows(): lines.append(f"| {lev} | {r.support_entropy_mean:.3f} | {r.coverage_mean:.3f} | {r.state_repair_mean:.3f} | {pct(r.target_success_mean)} | {pct(r.heldout_mean)} | {pct(r.external_mean)} |")
for metric in ['support_entropy','state_repair','heldout','external','target_success']:
 r=stats[(stats.metric==metric)&(stats.comparison=='failure-global')].iloc[0]
 unit=' pp' if metric in ['heldout','external','target_success'] else ''
 mult=100 if unit else 1
 lines.append(f"- Failure − Global, {metric}: {r.delta*mult:.3f}{unit}, 95% bootstrap CI [{r.ci_lo*mult:.3f}, {r.ci_hi*mult:.3f}]{unit}; paired t p={r.p_t:.3g}.")
lines += ['', '## Empirical phase map: fraction of 20 transfer × prevalence cells won','', '| Diagnostic noise | Budget | Global | Capability | Family | Failure |','|---:|---:|---:|---:|---:|---:|']
for (n,b),g in win.groupby(['noise','budget']):
 d={r.winner:r.fraction for _,r in g.iterrows()}; lines.append(f"| {n:.1f} | {int(b)} | {d.get('global',0):.2f} | {d.get('capability',0):.2f} | {d.get('family',0):.2f} | {d.get('failure',0):.2f} |")
lines += ['', '## Alternative support definitions','', '| Granularity | Normalized entropy | Effective support | Coverage | JS from global | Broad eval |','|---|---:|---:|---:|---:|---:|']
for lev,r in sup.iterrows(): lines.append(f"| {lev} | {r.entropy_norm_mean:.3f} | {r.effective_support_norm_mean:.3f} | {r.coverage_mean:.3f} | {r.js_from_global_mean:.3f} | {pct(r.broad_eval_mean)} |")
(O/'AUTO_RESULTS.md').write_text('\n'.join(lines))
print('WROTE',O/'AUTO_RESULTS.md')
