import sys,pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from fgt8_lib import *
rows=[]; reg=Regime8(transfer=.5,entangle=.3,noise=.1,horizon=8,diversity=.7,transition_noise=.08)
for kind in ['tabular','linear','nonlinear']:
 for seed in range(9001,9021):
  for method in ['uniform','difficulty','capability','family','failure','random_gran']:
   rows+=run8(seed,reg,method,kind,3,100)
pd.DataFrame(rows).to_csv(str(ROOT/'results'/'fgt8_main.csv'),index=False); print('MAIN8',len(rows))
