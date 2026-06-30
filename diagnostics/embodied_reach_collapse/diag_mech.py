"""Mechanism of the PC forward-model failure.

Controls, all on the SAME synthetic FK + real core graph:
  (1) full PC (current).
  (2) PC but i.i.d. UNIFORM commands instead of correlated OU  -> isolates the
      online non-stationarity / correlation from the rule itself.
  (3) cerebellar code informativeness: across a command grid, how much does the
      forward-pass cerebellum mu vary with the command, and can a LINEAR readout
      fit from the *frozen PC* cerebellar features to the tip code (least squares)?
      -> if even an optimal linear readout off the PC-learned cerebellum can't
         decode the tip, the cerebellum REPRESENTATION (driven by W_mc) is the
         bottleneck; the deep edge motor->cerebellum never learned to encode the
         command. If the optimal readout DOES fit, then cb->sensory (the readout
         edge) is what failed to learn.
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

grid=jnp.linspace(-HALF,HALF,201); codes=jax.vmap(mono)(grid)
def dec_tip(s):
    a=grid[jnp.argmin(jnp.sum((codes-s[PROP:PROP+N][None])**2,1))]
    b=grid[jnp.argmin(jnp.sum((codes-s[PROP+N:][None])**2,1))]
    return jnp.stack([a,b])

def babble(mode, n_babble=20000):
    params,state=init_pc_brain(make_key(0),sensory_size=SENS,motor_size=2,eta_w=0.05,n_relax=30)
    m=params.motor_idx; s=params.sensory_idx; cb=params.cerebellum_idx
    hold=tuple(params.perceptual_nodes)
    a=np.exp(-1/20.0); g=1.5*np.sqrt(1-a*a)
    @eqx.filter_jit
    def chunk(state,bel,keys):
        def step(c,k):
            st,b=c
            if mode=="ou":
                b=a*b+g*jax.random.normal(k,(2,),DTYPE)
            else:  # iid uniform belief in [-2,2]
                b=jax.random.uniform(k,(2,),DTYPE,-2.0,2.0)
            st=pc_brain_learn_forward(st,params,b,enc(jnp.tanh(b)),n_relax=None)
            return (st,b),None
        (state,bel),_=jax.lax.scan(step,(state,bel),keys); return state,bel
    state,_=chunk(state,jnp.zeros(2,DTYPE),jax.random.split(make_key(1),n_babble))
    return params,state,m,s,cb,hold

def fwd_cb_and_pred(params,state,m,s,cb,hold,c):
    bel=jnp.arctanh(jnp.clip(c,-0.999,0.999))
    cl=pc_graph_clamp(state.graph,{m:bel})
    rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=120)
    pred=pc_graph_predictions(rl,params.graph)[s]
    return _phi(params.graph.act, rl.mu[cb]), pred

for mode in ["ou","iid"]:
    params,state,m,s,cb,hold=babble(mode)
    cs=jax.random.uniform(make_key(7),(200,2),DTYPE,-0.9,0.9)
    feats=[]; preds=[]; tgts=[]
    for c in cs:
        f,p=fwd_cb_and_pred(params,state,m,s,cb,hold,c)
        feats.append(f); preds.append(p); tgts.append(enc(c))
    feats=jnp.stack(feats); preds=jnp.stack(preds); tgts=jnp.stack(tgts)
    # PC decoded tip err
    pc_err=np.mean([float(jnp.linalg.norm(dec_tip(preds[i])-fk(cs[i]))) for i in range(len(cs))])
    # cerebellar feature variance (informativeness)
    fvar=float(jnp.mean(jnp.var(feats,axis=0)))
    # OPTIMAL linear readout from PC cerebellar features to the tip block (ridge LS)
    Phi=jnp.concatenate([feats, jnp.ones((feats.shape[0],1))],axis=1)
    Y=tgts[:,PROP:]                       # tip block (24)
    W=jnp.linalg.solve(Phi.T@Phi + 1e-3*jnp.eye(Phi.shape[1]), Phi.T@Y)
    Yhat=Phi@W
    full=jnp.zeros_like(tgts).at[:,PROP:].set(Yhat)
    opt_err=np.mean([float(jnp.linalg.norm(dec_tip(full[i])-fk(cs[i]))) for i in range(len(cs))])
    print(f"[{mode}] PC decoded tip err={pc_err:.3f}m | cerebellar feat var={fvar:.4f} | "
          f"OPTIMAL linear readout off PC cerebellum tip err={opt_err:.3f}m")

print("\nReading:")
print("  opt<<pc  => cerebellum DOES encode the command; the cb->sensory readout EDGE")
print("             failed to learn it (readout plasticity / precision problem).")
print("  opt~=pc (both big) => cerebellum representation itself is uninformative")
print("             (motor->cerebellum deep edge never learned) -> deep credit failure.")
print("  iid<<ou  => online correlation/non-stationarity is a big factor.")
