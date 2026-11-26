import copy
import heapq
import io
import json
import math
import platform
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field, asdict
from itertools import product

import numpy as np
import pandas as pd
import scipy
from scipy.special import log_ndtr

ITINERARIOS = {
    '011': ('São Gonçalo R1', 76, 76), '012': ('São Gonçalo R2', 76, 73),
    '013-CMEI': ('Cohab VI vía CMEI', 69, 81), '013-Distr.': ('Cohab VI vía Distrito', 67, 69),
    '014': ('Jardim Guararapes', 77, 80), '015': ('Residencial Pomares', 63, 69),
    '031-Unif.': ('Pedra Linda unificada', 72, 66), '031-N.T.': ('Pedra Linda vía Novo Tempo', 68, 61),
    '041-J.M.': ('Dom Avelar vía José e Maria', 80, 67), '041-Univ.': ('Dom Avelar universitaria', 44, 36),
    '041-S.L.': ('Dom Avelar vía Santa Luzia', 95, 77),
}
CENARIOS = {
    'Base': (1., 1., False), 'Demanda +20%': (1.2, 1., False),
    'Demanda +40%': (1.4, 1., False), 'Tempo de percurso +25%': (1., 1.25, False),
    'Indisponibilidade de veículo': (1., 1., True),
    'Perturbações combinadas': (1.2, 1.25, True),
}
POLITICAS = ('Sem controle', 'Controle reativo', 'MPC nominal', 'RMPC')
VERSION = '2.0-temporal-experimental'

@dataclass(frozen=True)
class Config:
    h_min: int = 10
    horas: float = 4.
    aquecimento_min: int = 60
    inicio_choque_min: int = 90
    duracao_choque_min: int = 30
    reserva_min: float = 10.
    capacidade: int = 80
    tau0: float = 5.
    theta_b: float = 2.5
    theta_d: float = 1.8
    servico_nominal_extra: float = 6.
    cv: float = .15
    demanda: float = .12
    pico: float = 1.25
    passo_ctrl: int = 6
    ganho: float = .5
    dt: int = 60
    H: int = 5
    u_max: int = 180
    passo_u: int = 60
    alpha: float = .50
    beta: float = .35
    gamma: float = .15
    W_ref: float = 5000.
    n_omega: int = 5
    raio: float = .20
    max_avaliacoes: int = 24
    max_s: float = 60.
    tolerancia_rec: float = .20
    n_rec: int = 3
    n_boot: int = 2000

    @property
    def T(self): return int(self.horas * 3600)
    @property
    def warm(self): return self.aquecimento_min * 60
    @property
    def shock(self): return self.inicio_choque_min * 60
    @property
    def shock_end(self): return self.shock + self.duracao_choque_min * 60
    @property
    def h(self): return self.h_min * 60

    def validate(self):
        if not (0 <= self.warm <= self.shock < self.shock_end < self.T):
            raise ValueError('Exigir 0 <= aquecimento <= início < fim do choque < T.')
        if self.dt != 60 or self.H < 1 or self.h <= 0 or self.T % self.dt:
            raise ValueError('Passo de 60 s, H positivo e T múltiplo de 60 s.')
        if min(self.alpha, self.beta, self.gamma) < 0 or not np.isclose(self.alpha+self.beta+self.gamma, 1):
            raise ValueError('Pesos não negativos devem somar 1.')
        if not (self.n_omega >= 3 and self.max_avaliacoes >= 2 and self.max_s > 0 and 0 <= self.raio < 1):
            raise ValueError('Parâmetros da otimização inválidos.')
        if not (self.cv > 0 and self.capacidade > 0 and self.passo_ctrl > 0 and self.W_ref > 0 and self.passo_u > 0 and self.u_max > 0 and self.demanda >= 0):
            raise ValueError('Parâmetros físicos inválidos.')

@dataclass(frozen=True)
class Route:
    code: str
    n: int
    duration: float
    m_arc: float
    fleet: int
    ctrl: frozenset
    lam: np.ndarray
    slots: int
    refs: tuple


def make_route(code, p):
    _, n, duration = ITINERARIOS[code]
    pure = max(duration*60*.65, duration*60 - n*(p.tau0+p.servico_nominal_extra))
    lam = p.demanda/60 * (.5+np.sin(np.linspace(0,np.pi,n)))
    lam[-1] = 0.
    return Route(code,n,duration,pure/(n-1),math.ceil((duration+p.reserva_min)/p.h_min),
                 frozenset(range(p.passo_ctrl,n-1,p.passo_ctrl)),lam,
                 math.ceil(p.T/p.h), (n//2,n-1))

@dataclass(frozen=True)
class World:

    arrivals: tuple
    arcs: np.ndarray
    demand_mult: float
    travel_mult: float
    outage: bool
    start: float
    end: float
    forecast: bool = False


def in_shock(t, start, end): return start <= t < end


def arrival_stream(route,p,rng,begin,end,dm,shock_start,shock_end,peak_factor=1.):

    cuts=sorted(set([float(begin),float(end)]+[float(x) for x in (.3*p.T,.7*p.T,shock_start,shock_end) if begin<x<end]))
    all_stops=[]
    for s in range(route.n):
        rows=[]
        for a,b in zip(cuts,cuts[1:]):
            mid=(a+b)/2
            rate=route.lam[s]*peak_factor*(p.pico if .3*p.T<=mid<.7*p.T else 1.)*(dm if in_shock(mid,shock_start,shock_end) else 1.)
            count=int(rng.poisson(rate*(b-a)))
            if count:
                ts=np.sort(rng.uniform(a,b,count))
                dest=rng.integers(s+1,route.n,count)
                offset=len(rows)
                rows.extend((float(t),int(d),offset+i) for i,(t,d) in enumerate(zip(ts,dest)))
        all_stops.append(tuple(rows))
    return tuple(all_stops)


def make_world(route,p,scenario,seed):
    dm,tm,out=CENARIOS[scenario]
    children=np.random.SeedSequence(seed).spawn(2)
    arrivals=arrival_stream(route,p,np.random.default_rng(children[0]),0,p.T,dm,p.shock,p.shock_end)
    s2=math.log1p(p.cv**2)
    arcs=np.random.default_rng(children[1]).lognormal(-s2/2,math.sqrt(s2),(route.slots,route.n-1))
    return World(arrivals,arcs,dm,tm,out,p.shock,p.shock_end)

@dataclass
class State:
    events: list
    queues: list
    idx: list
    synced: list
    last_depart: list
    recent: dict
    free: set
    pending: deque
    onboard: dict = field(default_factory=dict)
    assignment: dict = field(default_factory=dict)
    moving: dict = field(default_factory=dict)
    seq: int = 0
    now: float = 0.
    area: float = 0.
    head: list = field(default_factory=list)
    waits: list = field(default_factory=list)
    denied: int = 0
    denied_ids: set = field(default_factory=set)
    holds: list = field(default_factory=list)
    control_slots: set = field(default_factory=set)
    used_slots: set = field(default_factory=set)
    dispatched: int = 0
    completed: int = 0
    recovery: float = math.nan
    solve_records: list = field(default_factory=list)


def push(st,t,kind,*args):
    st.seq+=1

    heapq.heappush(st.events,(float(t),kind,st.seq,args))


def initialize(route,p):
    st=State([], [deque() for _ in range(route.n)], [0]*route.n, [0.]*route.n,
             [None]*route.n,{s:deque(maxlen=p.n_rec) for s in route.refs},
             set(range(route.fleet)),deque())
    for b in range(route.slots): push(st,b*p.h,1,b)
    push(st,p.shock_end,0,-1)
    return st


def sync_stop(st,w,s,t):
    q=st.queues[s]; old=st.synced[s]
    if t < old-1e-7: raise AssertionError('Tempo não monotônico.')
    st.area += len(q)*max(0.,t-old)
    arr=w.arrivals[s]; i=st.idx[s]
    while i<len(arr) and arr[i][0]<=t:
        a,d,uid=arr[i]; q.append((a,d,uid)); st.area+=max(0.,t-a); i+=1
    st.idx[s]=i; st.synced[s]=t


def sync_all(st,w,t):
    for s in range(len(st.queues)): sync_stop(st,w,s,t)


def dispatch(st,w,route,p,t):


    avail=st.free-({0} if w.outage and in_shock(t,w.start,w.end) else set())
    while st.pending and avail:
        trip=st.pending.popleft(); vehicle=min(avail); avail.remove(vehicle); st.free.remove(vehicle)
        st.assignment[trip]=vehicle; st.onboard[trip]=[]; st.dispatched+=1
        push(st,t,2,trip,0)


def reactive(st,route,p,trip,s,t):
    prev=st.last_depart[s]
    return 0. if prev is None else float(np.clip(p.ganho*(p.h-(t-prev)),0,p.u_max))


def advance(st,w,route,p,end,policy,plan=None,origin=0.,record=False):
    while st.events and st.events[0][0]<end:
        t,kind,_,args=heapq.heappop(st.events); st.now=t
        if kind==0:
            vehicle=args[0]
            if vehicle>=0: st.free.add(vehicle)
            dispatch(st,w,route,p,t)
        elif kind==1:
            st.pending.append(args[0]); dispatch(st,w,route,p,t)
        elif kind==2:
            trip,s=args; st.moving.pop(trip,None); sync_stop(st,w,s,t)
            riders=st.onboard[trip]
            down=sum(d==s for d in riders)
            riders[:]=[d for d in riders if d!=s]
            q=st.queues[s]; board=min(len(q),p.capacidade-len(riders)) if s<route.n-1 else 0
            for _ in range(board):
                a,d,uid=q.popleft(); riders.append(d)
                if record and p.warm<=a<p.T: st.waits.append(t-a)
            if record and t>=p.warm and s<route.n-1:

                for a,d,uid in q:
                    if p.warm<=a<p.T: st.denied+=1; st.denied_ids.add((s,uid))
            assert 0<=len(riders)<=p.capacidade
            service=p.tau0+p.theta_b*board+p.theta_d*down
            push(st,t+service,3,trip,s)
        elif kind==3:
            trip,s=args; u=0.
            if s in route.ctrl:
                k=int((t-origin)//p.dt)
                key=(trip,k)
                st.control_slots.add(key)
                if key not in st.used_slots:
                    st.used_slots.add(key)
                    if policy=='Controle reativo': u=reactive(st,route,p,trip,s,t)
                    elif policy in ('MPC nominal','RMPC') and plan is not None: u=plan.get(key,0.)
                    assert 0<=u<=p.u_max
                    st.holds.append((t,u))
            push(st,t+u,4,trip,s)
        else:
            trip,s=args; prev=st.last_depart[s]
            if prev is not None:
                h=t-prev
                assert h>=-1e-7
                st.head.append((t,s,h))
                if s in st.recent and prev>=p.shock_end: st.recent[s].append((t,h))
            st.last_depart[s]=t
            if s==route.n-1:
                assert not st.onboard[trip]
                vehicle=st.assignment.pop(trip); del st.onboard[trip]; st.completed+=1
                push(st,t+p.reserva_min*60,0,vehicle)
            else:
                mult=w.travel_mult if in_shock(t,w.start,w.end) else 1.
                travel=route.m_arc*w.arcs[trip,s]*mult
                st.moving[trip]=(t,s+1)
                push(st,t+travel,2,trip,s+1)
    sync_all(st,w,end); st.now=end


def prediction_snapshot(st,world,route,p,t):

    z=copy.copy(st)
    z.events=[]


    sigma2=math.log1p(p.cv**2); sigma=math.sqrt(sigma2)
    for event in st.events:
        et,kind,seq,args=event
        if kind==2 and args[0] in st.moving:
            trip,s=args; departure,next_stop=st.moving[trip]
            assert s==next_stop
            elapsed=max(0.,t-departure)
            mean=route.m_arc*world.arcs[trip,s-1]*(world.travel_mult if in_shock(departure,world.start,world.end) else 1.)
            if elapsed>0 and sigma>0:
                mu=math.log(mean)-sigma2/2
                a=(math.log(elapsed)-mu)/sigma

                conditional=mean*math.exp(float(log_ndtr(sigma-a)-log_ndtr(-a)))
            else: conditional=mean
            et=max(t+1e-6,departure+conditional)
        z.events.append((et,kind,seq,args))
    heapq.heapify(z.events)
    if world.outage and t<world.end and world.end!=p.shock_end:
        push(z,world.end,0,-1)
    z.queues=[deque(q) for q in st.queues]
    z.idx=[0]*len(st.idx); z.synced=list(st.synced); z.last_depart=list(st.last_depart)
    z.recent={s:deque(v,maxlen=v.maxlen) for s,v in st.recent.items()}
    z.free=set(st.free); z.pending=deque(st.pending)
    z.onboard={b:list(v) for b,v in st.onboard.items()}; z.assignment=dict(st.assignment); z.moving=dict(st.moving)
    z.area=0.; z.head=[]; z.waits=[]; z.holds=[]; z.denied_ids=set(); z.denied=0
    z.control_slots=set(); z.used_slots=set(); z.solve_records=[]
    return z


def forecast_worlds(route,p,scenario,seed,t,policy):


    dm,tm,out=CENARIOS[scenario]; end=min(p.T,t+p.H*p.dt)
    nominal=[]
    if policy=='MPC nominal': specs=[(1.,1.,False)]
    else:
        lo=1-p.raio; hi=1+p.raio
        specs=[(1.,1.,False),(hi,hi,True),(hi,lo,False),(lo,hi,False),(lo,lo,False)]
        rg=np.random.default_rng(np.random.SeedSequence([seed,int(t),99]))
        while len(specs)<p.n_omega: specs.append((float(rg.uniform(lo,hi)),float(rg.uniform(lo,hi)),bool(rg.integers(2))))
        specs=specs[:p.n_omega]
    for i,(mt,md,extra) in enumerate(specs):

        rng=np.random.default_rng(np.random.SeedSequence([seed,int(t),12345]))
        arr=arrival_stream(route,p,rng,t,end,dm,p.shock,p.shock_end,md)


        arcs=np.full((route.slots,route.n-1),mt)
        outage_end=p.shock_end+(p.dt if extra and out else 0)
        nominal.append(World(arr,arcs,dm,tm,out,p.shock,outage_end,True))
    return nominal


def forecast_cost(st,world,route,p,t,plan,collect=False):
    z=prediction_snapshot(st,world,route,p,t); cost=0.
    for j in range(p.H):
        end=min(p.T,t+(j+1)*p.dt)
        if end<=z.now: break
        a0=z.area; h0=len(z.head); u0=len(z.holds)
        advance(z,world,route,p,end,'RMPC',plan,t,False)
        hs=[h for _,_,h in z.head[h0:]]; us=[u for _,u in z.holds[u0:]]


        e=float(np.mean([(h/p.h-1)**2 for h in hs])) if hs else 0.
        r=float(np.mean([(u/p.u_max)**2 for u in us])) if us else 0.
        cost+=p.alpha*e+p.beta*(z.area-a0)/p.W_ref+p.gamma*r
    return (cost,z.control_slots) if collect else cost


def optimize(st,route,p,scenario,seed,t,policy):
    start=time.perf_counter(); worlds=forecast_worlds(route,p,scenario,seed,t,policy)


    keys=set(); base=[]
    for w in worlds:
        c,k=forecast_cost(st,w,route,p,t,{},True); base.append(c); keys.update(k)
    if not keys or not any(k==0 for _,k in keys): return {},0.,'sem_oportunidade',1,0
    best={}; val=max(base); evaluations=1; timed_out=False

    alphabet=sorted(set(range(p.passo_u,p.u_max+1,p.passo_u))|{p.u_max})
    for key in sorted(keys,key=lambda x:(x[1],x[0])):
        local_best=best; local_val=val
        for u in alphabet:
            if evaluations>=p.max_avaliacoes: break
            if time.perf_counter()-start>=p.max_s: timed_out=True; break
            candidate=dict(best); candidate[key]=float(u)
            worst=max(forecast_cost(st,w,route,p,t,candidate) for w in worlds); evaluations+=1
            if worst<local_val-1e-9: local_best=candidate; local_val=worst
        best,val=local_best,local_val
        if evaluations>=p.max_avaliacoes or timed_out: break
    elapsed=time.perf_counter()-start
    if timed_out or elapsed>=p.max_s or not math.isfinite(val): return {},elapsed*1000,'fallback',evaluations,len(keys)
    return best,elapsed*1000,'factível_aproximado',evaluations,len(keys)


def recovered(st,route,p,t):
    if t<p.shock_end: return False
    return all(len(st.recent[s])==p.n_rec and t-st.recent[s][-1][0]<=(1+p.tolerancia_rec)*p.h
               and all(abs(h/p.h-1)<=p.tolerancia_rec for _,h in st.recent[s]) for s in route.refs)


def run_replica(code,scenario,policy,seed,p,world=None):
    p.validate(); route=make_route(code,p)
    w=world if world is not None else make_world(route,p,scenario,seed)
    st=initialize(route,p)
    for t in range(0,p.T,p.dt):
        plan={}; active=policy
        st.used_slots=set(); st.control_slots=set()
        if policy in ('MPC nominal','RMPC'):


            if st.onboard or st.pending or (st.events and st.events[0][0]<t+p.dt):
                plan,ms,status,ev,nkeys=optimize(st,route,p,scenario,seed,t,policy)
                if status!='sem_oportunidade' and t>=p.warm:
                    st.solve_records.append((ms,status,ev,nkeys))
                if status=='fallback': active='Controle reativo'
        advance(st,w,route,p,t+p.dt,active,plan,t,True)
        if scenario!='Base' and math.isnan(st.recovery) and recovered(st,route,p,t+p.dt):
            st.recovery=(t+p.dt-p.shock_end)/60

    residual=[p.T-a for q in st.queues for a,_,_ in q if p.warm<=a<p.T]
    cohort=sum(sum(p.warm<=a<p.T for a,_,_ in rows) for rows in w.arrivals)
    assert len(st.waits)+len(residual)==cohort, (len(st.waits),len(residual),cohort)
    hw=np.array([h for t,s,h in st.head if t>=p.warm and s>0],dtype=float)
    waits=np.array(st.waits,dtype=float); restricted=np.array(st.waits+residual,dtype=float)
    def mean(v): return float(np.mean(v)) if len(v) else math.nan
    def pct(v): return float(np.percentile(v,95)) if len(v) else math.nan
    records=st.solve_records
    follow=(p.T-p.shock_end)/60
    rec=st.recovery
    return {'Itinerario':code,'Cenario':scenario,'h_min':p.h_min,'Politica':policy,'Seed':seed,
        'RMSE_min':float(np.sqrt(np.mean((hw-p.h)**2))/60) if len(hw) else math.nan,
        'NRMSE':float(np.sqrt(np.mean((hw/p.h-1)**2))) if len(hw) else math.nan,
        'Bunching_pct':mean(hw<.5*p.h)*100,
        'Espera_media_min':mean(waits)/60,'Espera_p95_min':pct(waits)/60,
        'Espera_restrita_media_min':mean(restricted)/60,'Espera_restrita_p95_min':pct(restricted)/60,
        'Chegadas_coorte':cohort,'Embarcados_coorte':len(waits),'Fila_residual':len(residual),
        'Fila_residual_pct':100*len(residual)/cohort if cohort else math.nan,
        'Denegacoes_eventos':st.denied,'Passageiros_denegados_unicos':len(st.denied_ids),
        'Retencao_acum_min':sum(u for t,u in st.holds if t>=p.warm)/60,
        'Tempo_computo_ms':mean([x[0] for x in records]),'Tempo_computo_p95_ms':pct([x[0] for x in records]),
        'Factibilidade_pct':100*mean([x[1]=='factível_aproximado' for x in records]),
        'Fallbacks':sum(x[1]=='fallback' for x in records),'Otimizacoes':len(records),
        'Recuperou_pct':(100. if math.isfinite(rec) else 0.) if scenario!='Base' else math.nan,
        'Tempo_recuperacao_min':rec,'Tempo_recuperacao_restrito_min':(rec if math.isfinite(rec) else follow) if scenario!='Base' else math.nan,
        'Frota_sintetica':route.fleet,'Viagens_previstas':route.slots,'Viagens_iniciadas':st.dispatched,
        'Viagens_concluidas':st.completed,'Partidas_pendentes':len(st.pending),'Headways_n':len(hw)}


METRICAS={
 'RMSE_min':'RMSE de intervalos (min)','NRMSE':'NRMSE de intervalos', 'Bunching_pct':'Bunching (%)',
 'Espera_media_min':'Espera média dos embarcados (min)','Espera_p95_min':'Espera p95 dos embarcados (min)',
 'Espera_restrita_media_min':'Espera média observada até T (min)','Espera_restrita_p95_min':'Espera p95 observada até T (min)',
 'Fila_residual_pct':'Passageiros ainda em espera (%)','Denegacoes_eventos':'Embarques denegados (ocorrências)',
 'Passageiros_denegados_unicos':'Passageiros distintos com denegação','Retencao_acum_min':'Retenção acumulada (min)',
 'Tempo_computo_ms':'Tempo de cálculo (ms/otimização)','Tempo_computo_p95_ms':'p95 do tempo de cálculo (ms)',
 'Factibilidade_pct':'Planos factíveis retornados (%)','Recuperou_pct':'Recuperação até T (%)',
 'Tempo_recuperacao_restrito_min':'Tempo até recuperação ou fim do seguimento (min)',
 'Viagens_iniciadas':'Viagens iniciadas','Partidas_pendentes':'Partidas ainda pendentes'}


def bootstrap(v,n=2000,seed=42):
    a=np.asarray(v,dtype=float); a=a[np.isfinite(a)]
    if not len(a): return math.nan,math.nan,math.nan,0
    m=float(a.mean())
    if len(a)<2: return m,math.nan,math.nan,len(a)
    rng=np.random.default_rng(seed); means=a[rng.integers(0,len(a),(n,len(a)))].mean(axis=1)
    lo,hi=np.percentile(means,[2.5,97.5]); return m,float(lo),float(hi),len(a)


def summarize(df,p):
    group=['Itinerario','Cenario','h_min']; tables=[]; diffs=[]
    for keys,g in df.groupby(group,sort=False):
        for metric,label in METRICAS.items():
            row=dict(zip(group,keys)); row.update({'Metrica':metric,'Indicador':label})
            for pol in POLITICAS:
                m,lo,hi,n=bootstrap(g.loc[g.Politica==pol,metric],p.n_boot)
                row[pol]=f'{m:.3f} [{lo:.3f}; {hi:.3f}] (n={n})' if n>=2 else (f'{m:.3f} (n=1; IC indisponível)' if n else 'N/A')
            tables.append(row)
            pv=g.pivot(index='Seed',columns='Politica',values=metric)
            for comparator in POLITICAS[:-1]:
                if 'RMPC' not in pv or comparator not in pv: continue
                d=(pv['RMPC']-pv[comparator]).to_numpy(); m,lo,hi,n=bootstrap(d,p.n_boot)
                diffs.append(dict(zip(group,keys))|{'Metrica':metric,'Comparador':comparator,'Diferenca_RMPC_menos_comparador':m,'IC95_inf':lo,'IC95_sup':hi,'Pares_validos':n})
    return pd.DataFrame(tables),pd.DataFrame(diffs)


def run_batch(codes,scenarios,hs,n_rep,p,seed0=1000,progress=None):
    from dataclasses import replace
    records=[]; total=len(codes)*len(scenarios)*len(hs)*n_rep*len(POLITICAS); done=0
    for code,scenario,h in product(codes,scenarios,hs):
        cfg=replace(p,h_min=int(h)); route=make_route(code,cfg)
        for seed in range(seed0,seed0+n_rep):
            w=make_world(route,cfg,scenario,seed)
            cohort=[]
            for pol in POLITICAS:
                r=run_replica(code,scenario,pol,seed,cfg,w); records.append(r); cohort.append(r['Chegadas_coorte']); done+=1
                if progress: progress(done,total,code,scenario,h,seed,pol)
            assert len(set(cohort))==1,'Demanda exógena não pareada.'
    df=pd.DataFrame(records); table,diff=summarize(df,p)
    return df,table,diff


def export_zip(df,table,diff,p,codes,scenarios,hs,n_rep,seed0):
    meta={'version':VERSION,'config':asdict(p),'itinerarios':list(codes),'cenarios':list(scenarios),'intervalos':list(hs),
          'replicas':n_rep,'seed_inicial':seed0,'python':platform.python_version(),'numpy':np.__version__,
          'pandas':pd.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
          'escopo':'itinerários isolados sintéticos; sem validação empírica da operação real'}
    try:
        from importlib.metadata import version
        meta['streamlit']=version('streamlit')
    except Exception:
        meta['streamlit']=None
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in [('replicas.csv',df),('tabela3.csv',table),('diferencas_pareadas.csv',diff)]: z.writestr(name,data.to_csv(index=False))
        z.writestr('parametros.json',json.dumps(meta,ensure_ascii=False,indent=2))
    return b.getvalue()


def app():
    import streamlit as st
    st.set_page_config(page_title='Simulador RMPC — A1',layout='wide')
    st.title('Simulador estocástico RMPC — Petrolina')
    st.caption('Versão 2.0 experimental | Δt = 60 s | itinerários isolados | busca min-max aproximada')
    st.warning('Parâmetros sintéticos. Resultados não são estimativas da operação real. Execute pilotos antes do lote definitivo.')
    with st.sidebar:
        st.header('Escopo experimental')
        codes=st.multiselect('Itinerários',list(ITINERARIOS),default=['041-Univ.'])
        scenarios=st.multiselect('Cenários',list(CENARIOS),default=['Perturbações combinadas'])
        hs=st.multiselect('Intervalos (min)',[10,15,20],default=[10])
        n_rep=st.number_input('Réplicas por combinação',1,2000,2)
        seed0=st.number_input('Semente inicial',0,2000000000,1000)
        horas=st.number_input('Duração da simulação (h)',2.,12.,4.,step=.5)
        warm=st.number_input('Aquecimento (min)',0,180,60)
        shock=st.number_input('Início do choque (min)',0,600,90)
        duration=st.number_input('Duração do choque (min)',1,180,30)
        st.header('Parâmetros operacionais')
        demanda=st.number_input('Demanda (pax/min/parada)',.02,1.,.12,step=.02)
        cv=st.number_input('CV do tempo de arco',.01,.8,.15,step=.05)
        cap=st.number_input('Capacidade',20,160,80)
        reserva=st.number_input('Reserva terminal sintética (min)',0.,60.,10.)
        passo=st.slider('Retenção a cada N paradas',3,15,6)
        ganho=st.slider('Ganho reativo',.1,1.,.5)
        st.header('Controladores preditivos')
        H=st.slider('Horizonte (decisões de 60 s)',1,20,5)
        alpha=st.slider('α — regularidade',0.,1.,.50,.05)
        beta=st.slider('β — espera',0.,1.,.35,.05)
        gamma=1-alpha-beta
        st.write(f'γ — intervenção: {gamma:.2f}')
        wref=st.number_input('W_ref (passageiros-segundo)',100.,100000.,5000.,step=100.)
        nomega=st.slider('Cenários robustos',3,12,5)
        raio=st.slider('Amplitude simétrica de incerteza (%)',5,40,20)/100
        passo_u=st.selectbox('Discretização da retenção (s)',[15,30,60,90],index=2)
        maxeval=st.number_input('Máximo de planos avaliados por decisão',2,200,24)
    p=Config(horas=float(horas),aquecimento_min=int(warm),inicio_choque_min=int(shock),duracao_choque_min=int(duration),
             demanda=demanda,cv=cv,capacidade=int(cap),reserva_min=reserva,passo_ctrl=passo,ganho=ganho,H=H,
             alpha=alpha,beta=beta,gamma=max(0.,gamma),W_ref=wref,n_omega=nomega,raio=raio,passo_u=passo_u,max_avaliacoes=int(maxeval))
    total=len(codes)*len(scenarios)*len(hs)*int(n_rep)*4
    st.write(f'Execuções previstas: {total}. Cada combinação contém quatro políticas e as mesmas realizações exógenas.')
    if st.button('Executar simulação comparativa',type='primary'):
        try:
            if not total: raise ValueError('Selecione itinerários, cenários e intervalos.')
            if gamma < -1e-9: raise ValueError('α + β não pode ultrapassar 1.')
            p.validate()
            bar=st.progress(0.); status=st.empty()
            def update(done,total,c,s,h,seed,pol):
                bar.progress(done/total); status.text(f'{done}/{total} | {c} | {s} | h={h} | seed={seed} | {pol}')
            df,table,diff=run_batch(codes,scenarios,hs,int(n_rep),p,int(seed0),update)
            archive=export_zip(df,table,diff,p,codes,scenarios,hs,int(n_rep),int(seed0))
            st.session_state['resultado_v2']=(df,table,diff,archive)
            status.success('Execução concluída.')
        except Exception as e: st.exception(e)
    if 'resultado_v2' in st.session_state:
        df,table,diff,archive=st.session_state['resultado_v2']
        st.subheader('Tabela 3 — médias e IC 95% bootstrap por combinação')
        st.dataframe(table,use_container_width=True,hide_index=True)
        st.caption('N/A: não aplicável ou sem observação. IC das diferenças pareadas na tabela seguinte. Espera dos embarcados deve ser lida com fila residual e espera observada até T; esta última é truncada, não espera completa.')
        st.subheader('Diferenças pareadas: RMPC menos comparador')
        st.dataframe(diff,use_container_width=True,hide_index=True)
        st.subheader('Comparação visual por combinação')
        combos=list(df[['Itinerario','Cenario','h_min']].drop_duplicates().itertuples(index=False,name=None))
        c,s,h=st.selectbox('Combinação para gráficos',combos,format_func=lambda x:f'{x[0]} | {x[1]} | h={x[2]} min')
        subset=df[(df.Itinerario==c)&(df.Cenario==s)&(df.h_min==h)]
        means=subset.groupby('Politica').mean(numeric_only=True).reindex(POLITICAS)
        for col,(metric,label) in zip(st.columns(3), [('RMSE_min','RMSE (min)'),('Espera_restrita_media_min','Espera observada até T (min)'),('Bunching_pct','Agrupamento (%)')]):
            with col:
                st.markdown(f'**{label}**'); st.bar_chart(means[metric])
        st.subheader('Dados por réplica'); st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button('Baixar pacote de resultados e parâmetros (ZIP)',archive,'RMPC_resultados.zip','application/zip')
        st.download_button('Baixar Tabela 3 (CSV)',table.to_csv(index=False).encode('utf-8-sig'),'tabela3.csv','text/csv')
        st.download_button('Baixar réplicas (CSV)',df.to_csv(index=False).encode('utf-8-sig'),'replicas.csv','text/csv')


def self_test():

    p=Config(horas=.75,aquecimento_min=0,inicio_choque_min=10,duracao_choque_min=10,H=5,max_avaliacoes=5,n_omega=3)
    route=make_route('041-Univ.',p); w=make_world(route,p,'Perturbações combinadas',1000)
    results=[run_replica(route.code,'Perturbações combinadas',pol,1000,p,w) for pol in POLITICAS]
    assert len({r['Chegadas_coorte'] for r in results})==1
    assert all(r['Embarcados_coorte']+r['Fila_residual']==r['Chegadas_coorte'] for r in results)
    assert all(r['Viagens_iniciadas']<=r['Viagens_previstas'] for r in results)
    df=pd.DataFrame(results); table,diff=summarize(df,p)
    assert len(table)==len(METRICAS) and len(diff)==3*len(METRICAS)
    assert all(r['Otimizacoes']>0 for r in results[2:])
    assert results[0]['Retencao_acum_min']==0
    print(df[['Politica','Chegadas_coorte','RMSE_min','Fila_residual','Otimizacoes','Fallbacks']].to_string(index=False))
    print('SELF-TEST OK — não usar como resultados do artigo.')

if __name__=='__main__':
    import sys
    if '--self-test' in sys.argv: self_test()
    else: app()
