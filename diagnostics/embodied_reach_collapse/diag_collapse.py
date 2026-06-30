"""Confirm the collapse mechanism of the motor->cerebellum->sensory forward model.

Predicted trivial fixed point of the local PC rule from scratch:
  - cerebellum settles ~ command-independent (= its bias) during learning,
  - so eps_cb ~= -W_mc.phi(motor)  -> dW_mc = -eta W_mc phi phi^T -> W_mc DECAYS,
  - the sensory bias absorbs the mean reafference (low FE without any hidden code).
Net: a deep generative model that 'predicts the mean', hidden weights ~0.

Track ||W_mc||, ||W_cs||, ||bias_cb||, ||bias_s||, and the cerebellar variance
over babble. Also: does SEEDING W_mc large (like the cortex Gabor seed) rescue it?
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx

from core.backend import DTYPE, split_key, make_key
from core.pc_brain import init_pc_brain, pc_brain_learn_forward
from core.pc_graph import pc_graph_clamp, pc_graph_relax, pc_graph_predictions
from core.pc_module import _phi
from sensory.population_code import gaussian_population_encode, monotonic_population_encode

N=12; L1=L2=0.25; HALF=0.5; JR=2.0
PROP=2*2*N; TIP=2*N; SENS=PROP+TIP
mono=lambda v: monotonic_population_encode(v,N,x_min=-HALF,x_max=HALF)
def fk(c):
    q=JR*c; return jnp.stack([L1*jnp.cos(q[0])+L2*jnp.cos(q[0]+q[1]),
                              L1*jnp.sin(q[0])+L2*jnp.sin(q[0]+q[1])])
def enc(c):
    q=JR*c
    ang=jnp.concatenate([gaussian_population_encode(q[j],N,x_min=-JR,x_max=JR) for j in range(2)])
    vel=jnp.concatenate([gaussian_population_encode(0.0,N,x_min=-JR,x_max=JR) for _ in range(2)])
    t=fk(c); return jnp.concatenate([ang,vel,mono(t[0]),mono(t[1])]).astype(DTYPE)

def edge(params,src,dst):
    for e,(a,b) in enumerate(params.graph.edges):
        if a==src and b==dst: return e
    raise RuntimeError

def make(seed_wmc=None):
    params,state=init_pc_brain(make_key(0),sensory_size=SENS,motor_size=2,eta_w=0.05,n_relax=30)
    if seed_wmc is not None:
        e=edge(params,params.motor_idx,params.cerebellum_idx)
        w=list(state.graph.weights)
        w[e]=w[e]*seed_wmc
        state=eqx.tree_at(lambda s:s.graph.weights,state,tuple(w))
    return params,state

def norms(params,state):
    e_mc=edge(params,params.motor_idx,params.cerebellum_idx)
    e_cs=edge(params,params.cerebellum_idx,params.sensory_idx)
    cb=params.cerebellum_idx; s=params.sensory_idx
    return (float(jnp.linalg.norm(state.graph.weights[e_mc])),
            float(jnp.linalg.norm(state.graph.weights[e_cs])),
            float(jnp.linalg.norm(state.graph.bias[cb])),
            float(jnp.linalg.norm(state.graph.bias[s])))

def cb_var(params,state):
    m=params.motor_idx; cb=params.cerebellum_idx; hold=tuple(params.perceptual_nodes)
    cs=jax.random.uniform(make_key(7),(100,2),DTYPE,-0.9,0.9)
    feats=[]
    for c in cs:
        bel=jnp.arctanh(jnp.clip(c,-0.999,0.999))
        cl=pc_graph_clamp(state.graph,{m:bel})
        rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=120)
        feats.append(_phi(params.graph.act,rl.mu[cb]))
    return float(jnp.mean(jnp.var(jnp.stack(feats),axis=0)))

grid=jnp.linspace(-HALF,HALF,201); codes=jax.vmap(mono)(grid)
def fwd_tip_err(params,state):
    m=params.motor_idx; s=params.sensory_idx; hold=tuple(params.perceptual_nodes)
    cs=jax.random.uniform(make_key(7),(64,2),DTYPE,-0.7,0.7)
    errs=[]
    for c in cs:
        bel=jnp.arctanh(jnp.clip(c,-0.999,0.999))
        cl=pc_graph_clamp(state.graph,{m:bel})
        rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=120)
        pr=pc_graph_predictions(rl,params.graph)[s]
        a=grid[jnp.argmin(jnp.sum((codes-pr[PROP:PROP+N][None])**2,1))]
        b=grid[jnp.argmin(jnp.sum((codes-pr[PROP+N:][None])**2,1))]
        errs.append(float(jnp.linalg.norm(jnp.stack([a,b])-fk(c))))
    return float(np.mean(errs))

def babble_step(params):
    a=np.exp(-1/20.0); g=1.5*np.sqrt(1-a*a)
    @eqx.filter_jit
    def chunk(state,bel,keys):
        def step(c,k):
            st,b=c
            b=a*b+g*jax.random.normal(k,(2,),DTYPE)
            st=pc_brain_learn_forward(st,params,b,enc(jnp.tanh(b)),n_relax=None)
            return (st,b),None
        (state,bel),_=jax.lax.scan(step,(state,bel),keys); return state,bel
    return chunk

for tag,seed in [("default-init",None),("seed W_mc x8",8.0)]:
    params,state=make(seed)
    chunk=babble_step(params)
    bel=jnp.zeros(2,DTYPE)
    print(f"\n=== {tag} ===")
    print(f"  {'babble':>7} | {'|W_mc|':>7} {'|W_cs|':>7} {'|bias_cb|':>9} {'|bias_s|':>8} | cb_var")
    n=0
    for ck in [0,2000,6000,12000,20000]:
        if ck>n:
            key=make_key(100+ck)
            state,bel=chunk(state,bel,jax.random.split(key,ck-n)); n=ck
        wmc,wcs,bcb,bs=norms(params,state); v=cb_var(params,state)
        print(f"  {ck:7d} | {wmc:7.3f} {wcs:7.3f} {bcb:9.3f} {bs:8.3f} | {v:.5f}")
    print(f"  -> forward-model decoded tip err @20k = {fwd_tip_err(params,state):.3f} m")

print("\nIf |W_mc| collapses toward 0 while |bias_s| grows and cb_var->0:")
print("  the deep hidden edge is decayed to a 'predict-the-mean' trivial fixed point.")
print("If seeding W_mc keeps cb_var>0 and a real forward model forms: collapse confirmed,")
print("  and the substrate needs a non-collapse mechanism for deep edges (not a tuning).")
