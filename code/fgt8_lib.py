import numpy as np, math, json
from dataclasses import dataclass
K=8
FAILS=['tool','arg','order','state','termination','recovery','constraint','verification']
FIDX={f:i for i,f in enumerate(FAILS)}
CAP_GROUPS={0:[0,1,2,3],1:[4,5,6,7]}
FAM_GROUPS={0:[0,1],1:[2,3],2:[4,5],3:[6,7]}
CAP_OF={i:(0 if i<4 else 1) for i in range(K)}
FAM_OF={0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3}
@dataclass
class Regime8:
    transfer: float=.5
    entangle: float=.25
    noise: float=.1
    horizon: int=8
    diversity: float=.7
    transition_noise: float=0.0

def transfer_affinity(i,j):
    if i==j:return 1.0
    if FAM_OF[i]==FAM_OF[j]: return 1.0
    if CAP_OF[i]==CAP_OF[j]: return .55
    return .18

class Agent8:
    def __init__(self,kind,rng):
        self.kind=kind; self.rng=rng
        self.skill=np.clip(rng.normal(.38,.04,K),.15,.6)
        self.shared=0 if kind=='tabular' else np.clip(rng.normal(.10 if kind=='linear' else .14,.015),.04,.22)
    def prob(self,t,reg):
        req=t['req']; d=t['difficulty']; h=t['horizon']; inds=np.where(req>0)[0]
        base=(self.skill*req).sum()/(req.sum()+1e-9)
        if self.kind!='tabular' and len(inds)>1: base += self.shared*reg.transfer*(len(inds)-1)/len(inds)
        logit=5.4*(base-d)-.023*max(0,h-4)
        return float(np.clip(1/(1+np.exp(-logit)),.02,.98))
    def train(self,t,reg,lr=.026):
        req=t['req']; d=t['difficulty']; p=self.prob(t,reg); learn=(1-p)*p*4
        inds=np.where(req>0)[0]
        for i in inds:
            self.skill[i]+=lr*learn*(.85+.28*d)
            for j in range(K):
                if j==i: continue
                if self.kind=='tabular': cross=0
                elif self.kind=='linear': cross=.12
                else: cross=.17*max(0,1-self.skill[j])**.5
                self.skill[j]+=lr*learn*cross*reg.transfer*transfer_affinity(i,j)*(.5+.5*reg.entangle)
        if self.kind!='tabular': self.shared+=lr*learn*(.06 if self.kind=='linear' else .09)*reg.transfer*len(inds)/K
        self.skill=np.clip(self.skill,.02,.97); self.shared=np.clip(self.shared,0,.5)

def make_task8(rng,reg,level='global',target=None):
    req=np.zeros(K)
    if level=='failure' and target is not None:
        req[int(target)]=1
        if rng.random()<reg.entangle:
            j=int(rng.choice([x for x in range(K) if x!=target])); req[j]=1
    elif level=='family' and target is not None:
        mem=FAM_GROUPS[int(target)]; req[int(rng.choice(mem))]=1
        if rng.random()<.58+.22*reg.entangle: req[int(rng.choice(mem))]=1
        if rng.random()<.18*reg.entangle: req[int(rng.choice([x for x in range(K) if x not in mem]))]=1
    elif level=='capability' and target is not None:
        mem=CAP_GROUPS[int(target)]; req[int(rng.choice(mem))]=1
        if rng.random()<.62+.20*reg.entangle: req[int(rng.choice(mem))]=1
        if rng.random()<.28+.18*reg.entangle: req[int(rng.choice(mem))]=1
        if rng.random()<.12*reg.entangle: req[int(rng.choice([x for x in range(K) if x not in mem]))]=1
    else:
        kk=1+int(rng.random()<.42)+int(rng.random()<.23*reg.entangle)
        req[rng.choice(K,int(kk),replace=False)]=1
    if rng.random()<reg.transition_noise: req[int(rng.integers(0,K))]=1
    diff=float(np.clip(rng.beta(2.4,2.1)*(.58+.36*reg.diversity)+.1,.07,.95))
    h=int(max(2,rng.poisson(max(2,reg.horizon-1))+1))
    return {'req':req,'difficulty':diff,'horizon':h}

def eval8(a,reg,rng,n=250,external=False,prevalence=None,target=3):
    reg2=Regime8(**reg.__dict__)
    if external: reg2.diversity=min(1,reg.diversity+.22); reg2.horizon+=3; reg2.transition_noise=min(.4,reg.transition_noise+.08)
    succ=[]; fc=np.zeros(K); tc=np.zeros(K)
    for _ in range(n):
        if prevalence is not None and rng.random()<prevalence: t=make_task8(rng,reg2,'failure',target)
        else: t=make_task8(rng,reg2)
        p=a.prob(t,reg2); y=rng.random()<p; succ.append(y)
        inds=np.where(t['req']>0)[0]; tc[inds]+=1
        if not y: fc[inds]+=1
    return float(np.mean(succ)),np.divide(fc,tc,out=np.zeros_like(fc),where=tc>0)

def diagnose8(a,reg,rng,n=100):
    c=np.ones(K)*1e-3
    for _ in range(n):
        t=make_task8(rng,reg); p=a.prob(t,reg)
        if rng.random()>=p:
            inds=np.where(t['req']>0)[0]
            if len(inds):
                true=int(rng.choice(inds)); obs=int(rng.integers(0,K)) if rng.random()<reg.noise else true; c[obs]+=1
    return c/c.sum()

def curriculum8(a,method,reg,rng,budget):
    fd=diagnose8(a,reg,rng)
    for _ in range(budget):
        if method=='uniform': t=make_task8(rng,reg)
        elif method=='difficulty':
            cand=[make_task8(rng,reg) for __ in range(5)]; ps=[a.prob(x,reg) for x in cand]; t=cand[int(np.argmin(np.abs(np.array(ps)-.5)))]
        elif method=='capability':
            mass=np.array([fd[CAP_GROUPS[g]].sum() for g in range(2)]); tg=int(rng.choice(2,p=mass/mass.sum())); t=make_task8(rng,reg,'capability',tg)
        elif method=='family':
            mass=np.array([fd[FAM_GROUPS[g]].sum() for g in range(4)]); tg=int(rng.choice(4,p=mass/mass.sum())); t=make_task8(rng,reg,'family',tg)
        elif method=='failure':
            tg=int(rng.choice(K,p=fd/fd.sum())); t=make_task8(rng,reg,'failure',tg)
        elif method=='random_gran':
            g=str(rng.choice(['global','capability','family','failure']))
            if g=='global': t=make_task8(rng,reg)
            elif g=='capability': t=make_task8(rng,reg,g,int(rng.integers(0,2)))
            elif g=='family': t=make_task8(rng,reg,g,int(rng.integers(0,4)))
            else: t=make_task8(rng,reg,g,int(rng.integers(0,K)))
        a.train(t,reg)

def run8(seed,reg,method,kind='linear',rounds=3,budget=100):
    rng=np.random.default_rng(seed*1009+sum(map(ord,method+kind))); a=Agent8(kind,rng); rows=[]
    for r in range(rounds+1):
        h,fr=eval8(a,reg,np.random.default_rng(seed*2003+r),300,False); x,xfr=eval8(a,reg,np.random.default_rng(seed*3001+r),300,True)
        rows.append({'seed':seed,'kind':kind,'method':method,'round':r,'heldout':h,'external':x,**{f'fail_{FAILS[i]}':fr[i] for i in range(K)}})
        if r<rounds: curriculum8(a,method,reg,rng,budget)
    return rows
