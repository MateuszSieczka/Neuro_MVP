"""The forward model is fine; is the INVERSION the bottleneck?

diag_oscillation showed: even the least-squares-ceiling readout (forward err
0.028 m) reaches only ~0.59 m / 0% via pc_act_infer. So test the inversion in
isolation, on ONE fixed good readout (LS ceiling on the frozen granule basis):

  (1) pc_act_infer (the real active-inference relaxation through the whole graph)
      at NR_ACT in {80, 200, 500, 1500}, report FK err + |command| saturation.
  (2) a CLEAN explicit gradient-descent inversion of the SAME readout+granule
      (min_bel ||goal - readout(granule(bel))||^2), the diag_phase1b method.
  (3) inversion using ONLY motor+cerebellum+sensory (no other graph nodes/edges):
      does the rest of the graph interfere?

If (2) << (1): the active-inference relaxation, not the model, loses the reach.
If (1) improves a lot with more steps: it is just under-relaxed (NR_ACT too small).
If command saturates (|c|->1): the goal is outside the model's tanh-reachable set.
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx

from core.backend import DTYPE, make_key
from core.pc_brain import init_pc_brain
from core.pc_graph import pc_graph_clamp, pc_graph_relax, pc_graph_predictions
from core.pc_active import pc_act_infer
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

NR_FWD=60
def edge(params,src,dst):
    for e,(a,b) in enumerate(params.graph.edges):
        if a==src and b==dst: return e
    raise RuntimeError

params,state=init_pc_brain(make_key(0),sensory_size=SENS,motor_size=2,eta_w=0.05,n_relax=30)
m=params.motor_idx; s=params.sensory_idx; cb=params.cerebellum_idx
hold=tuple(params.perceptual_nodes)
e_cs=edge(params,cb,s)

# ---- fit the LS-ceiling readout on the frozen granule basis ------------------
@eqx.filter_jit
def granule_feats(state, bels):
    def one(bel):
        cl=pc_graph_clamp(state.graph,{m:bel})
        rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=NR_FWD)
        return _phi(params.graph.act, rl.mu[cb])
    return jax.vmap(one)(bels)
tr=jax.random.uniform(make_key(99),(4000,2),DTYPE,-1.2,1.2)
Phi=granule_feats(state,tr); Phi1=jnp.concatenate([Phi,jnp.ones((4000,1),DTYPE)],1)
Y=jax.vmap(enc)(jnp.tanh(tr))
Wls=jnp.linalg.solve(Phi1.T@Phi1+1e-3*jnp.eye(Phi1.shape[1]),Phi1.T@Y)
Wcs_ls=jnp.asarray(Wls[:-1].T,DTYPE); bs_ls=jnp.asarray(Wls[-1],DTYPE)
state=eqx.tree_at(lambda S:(S.graph.weights[e_cs],S.graph.bias[s]),state,(Wcs_ls,bs_ls))

# ---- the granule->readout forward map as a plain function (for clean inversion)
Wmc=state.graph.weights[edge(params,m,cb)]; bmc=state.graph.bias[cb]
def granule(bel): return jnp.tanh(Wmc@_phi(params.graph.act,bel)+bmc)
def readout(bel): return Wcs_ls@granule(bel)+bs_ls
grid=jnp.linspace(-HALF,HALF,201); codes=jax.vmap(mono)(grid)
def decode_tip(pr):
    a=grid[jnp.argmin(jnp.sum((codes-pr[PROP:PROP+N][None])**2,1))]
    b=grid[jnp.argmin(jnp.sum((codes-pr[PROP+N:][None])**2,1))]
    return jnp.stack([a,b])

# ---- targets -----------------------------------------------------------------
rr=jax.random.uniform(make_key(11),(64,),DTYPE,0.15,0.42)
th=jax.random.uniform(make_key(12),(64,),DTYPE,-np.pi,np.pi)
targets=jnp.stack([rr*jnp.cos(th),rr*jnp.sin(th)],1)
mask=jnp.zeros(SENS,bool).at[PROP:PROP+TIP].set(True)
def goal_vec(tgt): return jnp.zeros(SENS,DTYPE).at[PROP:PROP+N].set(mono(tgt[0])).at[PROP+N:].set(mono(tgt[1]))

# ---- (1) pc_act_infer inversion at varying relax depth -----------------------
def act_infer_eval(nsteps):
    @eqx.filter_jit
    def one(tgt):
        out=pc_act_infer(state.graph,params.graph,motor_idx=m,outcome_idx=s,
                         preference=goal_vec(tgt),preference_mask=mask,hold_nodes=hold,n_steps=nsteps)
        return out.command
    cmds=jax.vmap(one)(targets)
    tips=jax.vmap(lambda b: fk(jnp.tanh(b)))(cmds)
    err=jnp.linalg.norm(tips-targets,axis=1)
    sat=jnp.mean(jnp.abs(jnp.tanh(cmds))>0.95)
    return float(jnp.mean(err)),float(jnp.mean(err<0.05)),float(jnp.median(err)),float(sat),float(jnp.mean(jnp.abs(cmds)))

print("=== Inversion isolation on the LS-ceiling readout (forward model ~0.028 m) ===")
print(f"  {'method':>22} | {'reach_err':>9} {'succ':>5} {'med':>5} | {'sat>0.95':>8} {'mean|cmd|':>9}")
for ns in [80,200,500,1500]:
    er,sc,md,sat,mc=act_infer_eval(ns)
    print(f"  {('pc_act_infer n='+str(ns)):>22} | {er:9.3f} {sc:5.1%} {md:5.3f} | {sat:8.1%} {mc:9.3f}")

# ---- (2) clean explicit gradient-descent inversion of the SAME readout -------
def clean_invert(tgt, steps=600, lr=0.3):
    goal=jnp.concatenate([mono(tgt[0]),mono(tgt[1])])
    def loss(bel): pr=readout(bel)[PROP:]; return jnp.sum((pr-goal)**2)
    g=jax.grad(loss); bel=jnp.zeros(2,DTYPE)
    def body(_,b): return b-lr*g(b)
    bel=jax.lax.fori_loop(0,steps,body,bel)
    return bel
cmds=jax.vmap(clean_invert)(targets)
tips=jax.vmap(lambda b: fk(jnp.tanh(b)))(cmds)
err=jnp.linalg.norm(tips-targets,axis=1)
print(f"  {'clean grad-descent':>22} | {float(jnp.mean(err)):9.3f} {float(jnp.mean(err<0.05)):5.1%} "
      f"{float(jnp.median(err)):5.3f} | {float(jnp.mean(jnp.abs(jnp.tanh(cmds))>0.95)):8.1%} {float(jnp.mean(jnp.abs(cmds))):9.3f}")

# ---- (3) does the rest of the graph interfere? Minimal motor+cb+sensory ------
# Hold EVERYTHING except motor + cerebellum + sensory; pin tip channels.
all_nodes=set(range(params.graph.n_nodes)); core3={m,cb,s}
extra_hold=tuple(sorted(all_nodes-core3))
def act_infer_minimal(nsteps):
    @eqx.filter_jit
    def one(tgt):
        out=pc_act_infer(state.graph,params.graph,motor_idx=m,outcome_idx=s,
                         preference=goal_vec(tgt),preference_mask=mask,hold_nodes=extra_hold,n_steps=nsteps)
        return out.command
    cmds=jax.vmap(one)(targets)
    tips=jax.vmap(lambda b: fk(jnp.tanh(b)))(cmds)
    err=jnp.linalg.norm(tips-targets,axis=1)
    return float(jnp.mean(err)),float(jnp.mean(err<0.05)),float(jnp.median(err))
er,sc,md=act_infer_minimal(500)
print(f"  {'minimal motor+cb+s n=500':>22} | {er:9.3f} {sc:5.1%} {md:5.3f} |   (only m,cb,s free)")

# ---- sanity: is each target even reachable by SOME command? (brute force) -----
gg=jnp.linspace(-1,1,81)
G=jnp.stack(jnp.meshgrid(gg,gg,indexing='ij'),-1).reshape(-1,2)   # candidate commands in [-1,1]
tipsG=jax.vmap(lambda c: fk(c))(G)
def best_reach(tgt):
    d=jnp.linalg.norm(tipsG-tgt,axis=1); return jnp.min(d)
bd=jax.vmap(best_reach)(targets)
print(f"\n  brute-force best reachable err (command grid): mean={float(jnp.mean(bd)):.3f} succ={float(jnp.mean(bd<0.05)):.1%}")
print("  (if this is small, every target IS reachable; any large reach_err above is the inversion's fault)")
