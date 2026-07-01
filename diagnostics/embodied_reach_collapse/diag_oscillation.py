"""Why does Phase-1 reach OSCILLATE (6-68%) across babble checkpoints with no
trend, while max|W| stays pinned at 3.18?

Phase 1 cured the *collapse* (cb_var stays >0, forward model is learnable). The
remaining symptom is wild checkpoint-to-checkpoint variance. This script
isolates the cause on the REAL core graph (marr_albus default) with a synthetic
2-link FK body, CPU, no MuJoCo.

At each babble checkpoint it measures, with FIXED probe sets (so metric noise is
not target-sampling noise):
  - forward-model decoded tip err  (model accuracy)
  - reach: one-shot active-inference inversion FK err + success<0.05 on 64 targets
  - ||W_cs|| (plastic Purkinje readout) and ||dW_cs|| since the last checkpoint
  - ||bias_s|| and ||d bias_s||
  - mean Pi_sensory
  - cb_var (collapse sanity: must stay >0)
  - max|W| over all edges (the 3.18 the notebook reports == the FROZEN granule edge)

Hypothesis under test: the plastic readout is updated by a CONSTANT-step online
rule on a never-ending stochastic babble stream, so it random-walks inside an
LMS misadjustment ball around the least-squares solution and never settles. The
per-checkpoint snapshot is a random draw from that ball; with the 0.05 m success
threshold sitting at the population-code resolution, readout jitter -> large
success swings with no trend.

Discriminators printed at the end:
  (a) does ||dW_cs|| FAIL to decay (perpetual jitter)?  -> misadjustment
  (b) does an EMA (Polyak-averaged) readout reach BETTER and STABLER than the
      raw online readout?                               -> averaging cures it
  (c) least-squares-ceiling readout reach (best possible on these features).
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx

from core.backend import DTYPE, split_key, make_key
from core.pc_brain import init_pc_brain, pc_brain_learn_forward
from core.pc_graph import pc_graph_clamp, pc_graph_relax, pc_graph_predictions
from core.pc_active import pc_act_infer
from core.pc_module import _phi
from sensory.population_code import gaussian_population_encode, monotonic_population_encode

# ----- synthetic 2-link body (identical to the other diag scripts) -----------
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
def decode_tip(pr):
    a=grid[jnp.argmin(jnp.sum((codes-pr[PROP:PROP+N][None])**2,1))]
    b=grid[jnp.argmin(jnp.sum((codes-pr[PROP+N:][None])**2,1))]
    return jnp.stack([a,b])

NR_FWD = 60          # forward-pass settle for the probes
NR_ACT = 80          # planning relax (matches notebook ACT_RELAX)

def edge(params,src,dst):
    for e,(a,b) in enumerate(params.graph.edges):
        if a==src and b==dst: return e
    raise RuntimeError

# ----- fixed probe sets ------------------------------------------------------
probe_cmds = jax.random.uniform(make_key(7),(64,2),DTYPE,-0.7,0.7)     # forward-model grid
rr = jax.random.uniform(make_key(11),(64,),DTYPE,0.15,0.42)
th = jax.random.uniform(make_key(12),(64,),DTYPE,-np.pi,np.pi)
targets = jnp.stack([rr*jnp.cos(th), rr*jnp.sin(th)],1)                 # reach targets (tip xy)

def build():
    params,state=init_pc_brain(make_key(0),sensory_size=SENS,motor_size=2,eta_w=0.05,n_relax=30)
    return params,state

# ----- babble (online, carrying state; same OU as the notebook) --------------
def babble_chunk_fn(params):
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

# ----- probes (vmapped, jitted) ---------------------------------------------
def make_probes(params):
    m=params.motor_idx; s=params.sensory_idx; cb=params.cerebellum_idx
    hold=tuple(params.perceptual_nodes)
    mask=jnp.zeros(SENS,bool).at[PROP:PROP+TIP].set(True)   # pin the tip channels only

    @eqx.filter_jit
    def fwd_pred(state, bels):                              # bels:(B,2) -> preds:(B,SENS)
        def one(bel):
            cl=pc_graph_clamp(state.graph,{m:bel})
            rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=NR_FWD)
            return pc_graph_predictions(rl,params.graph)[s]
        return jax.vmap(one)(bels)

    @eqx.filter_jit
    def cb_feats(state, bels):                              # granule activity variance
        def one(bel):
            cl=pc_graph_clamp(state.graph,{m:bel})
            rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=NR_FWD)
            return _phi(params.graph.act, rl.mu[cb])
        return jax.vmap(one)(bels)

    @eqx.filter_jit
    def reach_cmd(state, tgts):                             # tgts:(B,2) -> command beliefs
        def one(tgt):
            pref=jnp.zeros(SENS,DTYPE).at[PROP:PROP+N].set(mono(tgt[0])).at[PROP+N:].set(mono(tgt[1]))
            out=pc_act_infer(state.graph,params.graph,motor_idx=m,outcome_idx=s,
                             preference=pref,preference_mask=mask,hold_nodes=hold,n_steps=NR_ACT)
            return out.command
        return jax.vmap(one)(tgts)

    return fwd_pred, cb_feats, reach_cmd

def fwd_err(fwd_pred, state):
    preds=fwd_pred(state, probe_cmds)
    tips=jax.vmap(decode_tip)(preds)
    true=jax.vmap(fk)(probe_cmds)
    return float(jnp.mean(jnp.linalg.norm(tips-true,axis=1)))

def reach_metrics(reach_cmd, state):
    cmds=reach_cmd(state, targets)                          # (B,2) belief
    tips=jax.vmap(lambda b: fk(jnp.tanh(b)))(cmds)
    err=jnp.linalg.norm(tips-targets,axis=1)
    return float(jnp.mean(err)), float(jnp.mean(err<0.05)), float(jnp.median(err))

def cb_var(cb_feats, state):
    f=cb_feats(state, probe_cmds)
    return float(jnp.mean(jnp.var(f,axis=0)))

# =====================================================================
print("=== Phase-1 oscillation diagnostic (real core, synthetic FK body) ===")
params,state=build()
chunk=babble_chunk_fn(params)
fwd_pred, cb_feats, reach_cmd = make_probes(params)
e_cs=edge(params,params.cerebellum_idx,params.sensory_idx)
e_mc=edge(params,params.motor_idx,params.cerebellum_idx)
s=params.sensory_idx

bel=jnp.zeros(2,DTYPE)
prev_Wcs=np.asarray(state.graph.weights[e_cs]); prev_bs=np.asarray(state.graph.bias[s])
Wcs_ema=np.asarray(state.graph.weights[e_cs]).copy(); bs_ema=np.asarray(state.graph.bias[s]).copy()
ema_beta=0.8

hdr=f"  {'babble':>7} | {'fwd_err':>7} | {'reach_err':>9} {'succ':>5} {'med':>5} | " \
    f"{'|W_cs|':>7} {'|dW_cs|':>7} {'|W_mc|':>7} {'maxW':>5} | {'|bias_s|':>8} {'|db_s|':>6} | {'Pi_s':>6} | cb_var"
print(hdr)

STEP=2500; CKPTS=14
ema_rows=[]
n=0
for ck in range(0,CKPTS+1):
    target_n=ck*STEP
    if target_n>n:
        key=make_key(1000+target_n)
        state,bel=chunk(state,bel,jax.random.split(key,target_n-n)); n=target_n
    Wcs=np.asarray(state.graph.weights[e_cs]); bs=np.asarray(state.graph.bias[s])
    Wmc=float(jnp.linalg.norm(state.graph.weights[e_mc]))
    maxW=float(max(jnp.max(jnp.abs(w)) for w in state.graph.weights))
    dWcs=float(np.linalg.norm(Wcs-prev_Wcs)); dbs=float(np.linalg.norm(bs-prev_bs))
    Pi_s=float(jnp.mean(state.graph.pi[s]))
    fe_=fwd_err(fwd_pred,state); rerr,rsucc,rmed=reach_metrics(reach_cmd,state); v=cb_var(cb_feats,state)
    print(f"  {target_n:7d} | {fe_:7.3f} | {rerr:9.3f} {rsucc:5.1%} {rmed:5.3f} | "
          f"{float(jnp.linalg.norm(Wcs)):7.3f} {dWcs:7.3f} {Wmc:7.3f} {maxW:5.2f} | "
          f"{float(jnp.linalg.norm(bs)):8.3f} {dbs:6.3f} | {Pi_s:6.2f} | {v:.5f}")
    prev_Wcs=Wcs; prev_bs=bs
    # Polyak/consolidation EMA of the readout, evaluated on the SAME targets
    if target_n>0:
        Wcs_ema=ema_beta*Wcs_ema+(1-ema_beta)*Wcs; bs_ema=ema_beta*bs_ema+(1-ema_beta)*bs
        st_ema=eqx.tree_at(lambda S:(S.graph.weights[e_cs],S.graph.bias[s]),state,
                           (jnp.asarray(Wcs_ema,DTYPE),jnp.asarray(bs_ema,DTYPE)))
        er,sc,md=reach_metrics(reach_cmd,st_ema)
        ema_rows.append((target_n,er,sc,md))

print("\n-- EMA (consolidated) readout reach, same targets --")
print(f"  {'babble':>7} | {'reach_err':>9} {'succ':>5} {'med':>5}")
for nn,er,sc,md in ema_rows:
    print(f"  {nn:7d} | {er:9.3f} {sc:5.1%} {md:5.3f}")

# ----- least-squares ceiling on the FROZEN granule features (best possible) ---
m=params.motor_idx; hold=tuple(params.perceptual_nodes); cb=params.cerebellum_idx
@eqx.filter_jit
def granule_feats(state, bels):
    def one(bel):
        cl=pc_graph_clamp(state.graph,{m:bel})
        rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=NR_FWD)
        return _phi(params.graph.act, rl.mu[cb])
    return jax.vmap(one)(bels)
tr=jax.random.uniform(make_key(99),(3000,2),DTYPE,-1.2,1.2)
Phi=granule_feats(state,tr); Phi1=jnp.concatenate([Phi,jnp.ones((3000,1),DTYPE)],1)
Y=jax.vmap(enc)(jnp.tanh(tr))
Wls=jnp.linalg.solve(Phi1.T@Phi1+1e-3*jnp.eye(Phi1.shape[1]),Phi1.T@Y)
Wcs_ls=jnp.asarray(Wls[:-1].T,DTYPE); bs_ls=jnp.asarray(Wls[-1],DTYPE)
st_ls=eqx.tree_at(lambda S:(S.graph.weights[e_cs],S.graph.bias[s]),state,(Wcs_ls,bs_ls))
er,sc,md=reach_metrics(reach_cmd,st_ls); fe_ls=fwd_err(fwd_pred,st_ls)
print(f"\n-- least-squares-ceiling readout (best on the fixed granule basis) --")
print(f"  fwd_err={fe_ls:.3f} m | reach_err={er:.3f} m  succ={sc:.1%}  med={md:.3f} m")

print("""
Read:
  * cb_var > 0 and |W_mc| constant  -> collapse is cured (Phase 1 worked); maxW==|W_mc|.
  * |dW_cs| not decaying toward 0    -> readout perpetually jitters (LMS misadjustment).
  * reach succ swinging while fwd_err stays similar -> the wander is readout noise,
    amplified by the 0.05 m threshold near the population-code resolution.
  * EMA readout >= raw and stabler   -> averaging/consolidation cures the oscillation.
  * LS ceiling >> online             -> the basis is fine; the online RULE is the limit.
""")
