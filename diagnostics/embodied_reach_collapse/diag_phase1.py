"""Phase-1 end-to-end validation: does FREEZING motor->cerebellum as a fixed
random Marr-Albus expansion (plastic only on cerebellum->sensory) actually make
the *reach inversion* land on target?

Freeze is emulated by restoring W(motor->cb) and bias_cb to their fixed random
init after every learning step (no core change needed for the probe). Then run
the real pc_brain_act inversion and measure FK(inferred command) vs target.

This separates two questions:
  (a) does the forward model now fit?           (acquisition)
  (b) does the inversion through the free wide   (inversion)
      cerebellum recover the right command?
If (a) good but (b) bad -> Phase 1 needs an inversion-side companion (constrain
the cerebellum during inversion, e.g. raise its prior precision / hold on-manifold).
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx

from core.backend import DTYPE, split_key, make_key
from core.pc_brain import init_pc_brain, pc_brain_learn_forward, pc_brain_act
from core.pc_graph import pc_graph_clamp, pc_graph_relax, pc_graph_predictions
from sensory.population_code import gaussian_population_encode, monotonic_population_encode

N=12; L1=L2=0.25; HALF=0.5; JR=2.0; PROP=2*2*N; TIP=2*N; SENS=PROP+TIP
mono=lambda v: monotonic_population_encode(v,N,x_min=-HALF,x_max=HALF)
def fk(c):
    q=JR*c; return jnp.stack([L1*jnp.cos(q[0])+L2*jnp.cos(q[0]+q[1]),
                              L1*jnp.sin(q[0])+L2*jnp.sin(q[0]+q[1])])
def enc(c):
    q=JR*c
    ang=jnp.concatenate([gaussian_population_encode(q[j],N,x_min=-JR,x_max=JR) for j in range(2)])
    vel=jnp.concatenate([gaussian_population_encode(0.0,N,x_min=-JR,x_max=JR) for _ in range(2)])
    t=fk(c); return jnp.concatenate([ang,vel,mono(t[0]),mono(t[1])]).astype(DTYPE)
def tip_pref(t):
    p=jnp.zeros(SENS,DTYPE).at[PROP:PROP+N].set(mono(t[0])).at[PROP+N:].set(mono(t[1])); return p
TIP_MASK=jnp.zeros(SENS,bool).at[PROP:].set(True)
grid=jnp.linspace(-HALF,HALF,201); codes=jax.vmap(mono)(grid)
def dec(s):
    a=grid[jnp.argmin(jnp.sum((codes-s[PROP:PROP+N][None])**2,1))]
    b=grid[jnp.argmin(jnp.sum((codes-s[PROP+N:][None])**2,1))]
    return jnp.stack([a,b])

def edge(params,src,dst):
    for e,(a,b) in enumerate(params.graph.edges):
        if a==src and b==dst: return e
    raise RuntimeError

def build(g_expansion):
    params,state=init_pc_brain(make_key(0),sensory_size=SENS,motor_size=2,eta_w=0.05,n_relax=30)
    m=params.motor_idx; cb=params.cerebellum_idx; s=params.sensory_idx
    e_mc=edge(params,m,cb)
    # Marr-Albus fixed random expansion: gain set so granule pre-activations span
    # their informative tanh range (standard random-feature/ELM scaling), NOT
    # tuned to reach. Diverse random thresholds (bias_cb) = granule threshold spread.
    kw,kb=split_key(make_key(42))
    Wmc=jax.random.normal(kw,(params.graph.node_sizes[cb],2),DTYPE)*g_expansion
    bcb=jax.random.normal(kb,(params.graph.node_sizes[cb],),DTYPE)*g_expansion
    w=list(state.graph.weights); w[e_mc]=Wmc
    bias=list(state.graph.bias); bias[cb]=bcb
    state=eqx.tree_at(lambda st:(st.graph.weights,st.graph.bias),state,(tuple(w),tuple(bias)))
    return params,state,m,cb,s,e_mc,Wmc,bcb

def babble_frozen(params,state,m,cb,s,e_mc,Wmc,bcb,n):
    a=np.exp(-1/20.0); g=1.5*np.sqrt(1-a*a)
    @eqx.filter_jit
    def chunk(state,bel,keys):
        def step(c,k):
            st,b=c
            b=a*b+g*jax.random.normal(k,(2,),DTYPE)
            st=pc_brain_learn_forward(st,params,b,enc(jnp.tanh(b)),n_relax=None)
            # emulate frozen granule expansion: restore W_mc and bias_cb
            w=list(st.graph.weights); w[e_mc]=Wmc
            bias=list(st.graph.bias); bias[cb]=bcb
            st=eqx.tree_at(lambda s_:(s_.graph.weights,s_.graph.bias),st,(tuple(w),tuple(bias)))
            return (st,b),None
        (state,bel),_=jax.lax.scan(step,(state,bel),keys); return state,bel
    return chunk(state,jnp.zeros(2,DTYPE),jax.random.split(make_key(1),n))[0]

def fwd_err(params,state,m,s):
    hold=tuple(params.perceptual_nodes)
    cs=jax.random.uniform(make_key(7),(64,2),DTYPE,-0.7,0.7); e=[]
    for c in cs:
        bel=jnp.arctanh(jnp.clip(c,-0.999,0.999))
        rl=pc_graph_relax(pc_graph_clamp(state.graph,{m:bel}),params.graph,clamp=(m,)+hold,n_steps=120)
        pr=pc_graph_predictions(rl,params.graph)[s]; e.append(float(jnp.linalg.norm(dec(pr)-fk(c))))
    return np.mean(e)

def reach_err(params,state,relax):
    rr=jax.random.uniform(make_key(7),(32,),DTYPE,0.15,0.45)
    th=jax.random.uniform(make_key(8),(32,),DTYPE,-np.pi,np.pi)
    tg=jnp.stack([rr*jnp.cos(th),rr*jnp.sin(th)],1); e=[]; sat=[]
    for t in tg:
        act=pc_brain_act(state,params,tip_pref(t),preference_mask=TIP_MASK,n_relax=relax)
        e.append(float(jnp.linalg.norm(fk(act.joint_command)-t))); sat.append(float(jnp.max(jnp.abs(act.joint_command))))
    e=np.array(e); return e.mean(), float(np.mean(e<0.05)), np.mean(sat)

for g in [1.0, 2.0]:
    params,state,m,cb,s,e_mc,Wmc,bcb=build(g)
    state=babble_frozen(params,state,m,cb,s,e_mc,Wmc,bcb,20000)
    fe=fwd_err(params,state,m,s)
    print(f"\n[expansion gain={g}] frozen Marr-Albus, 20k babble")
    print(f"  (a) forward-model decoded tip err = {fe:.3f} m")
    for relax in [80, 300]:
        re,sr,sat=reach_err(params,state,relax)
        print(f"  (b) reach inversion (n_relax={relax:3d}): FK err={re:.3f} m  success={sr:5.1%}  mean|cmd|max={sat:.2f}")
