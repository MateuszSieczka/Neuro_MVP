"""Why is the inferred motor command ~0? Instrument the goal->motor error flow.

LS-ceiling readout fixed. Clamp a tip goal (masked). Step the relaxation by hand
and print, per step: |mu_motor|, |eps_sensory[tip]|, raw motor gradient g_motor,
motor curvature L_motor, and the preconditioned step. Find which link is broken.
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx

from core.backend import DTYPE, make_key
from core.pc_brain import init_pc_brain
from core.pc_graph import (pc_graph_clamp, pc_graph_relax, pc_graph_predictions,
                           pc_graph_errors, _phi, _phi_prime, _incoming, _outgoing,
                           _build_hold, _temporal_predictions)
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
e_cs=edge(params,cb,s); e_mc=edge(params,m,cb)

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

P=params.graph
gs=state.graph
act=P.act
# choose a reachable target
tgt=jnp.asarray([0.30,0.10],DTYPE)
goal=jnp.zeros(SENS,DTYPE).at[PROP:PROP+N].set(mono(tgt[0])).at[PROP+N:].set(mono(tgt[1]))
mask=jnp.zeros(SENS,bool).at[PROP:PROP+TIP].set(True)

# clamp: tip channels of sensory = goal (held); perceptual parents held at current mu
mu=list(gs.mu)
mu[s]=jnp.where(mask,goal,gs.mu[s])
clamp=(s,)+hold   # NOTE sensory whole-held here only to *inspect*; below we use masks properly

incoming=_incoming(P.edges,P.n_nodes); outgoing=_outgoing(P.edges,P.n_nodes)
pi=gs.pi
temporal=_temporal_predictions(gs.mu_prev,gs.w_dyn,P)
temporal=tuple(temporal[j]+gs.bias[j] for j in range(P.n_nodes))

print("motor children (src=motor):", [(a,b) for e,(a,b) in enumerate(P.edges) if a==m])
print("cerebellum children:", [(a,b) for e,(a,b) in enumerate(P.edges) if a==cb])
print("sensory parents (dst=sensory):", [(a,b) for e,(a,b) in enumerate(P.edges) if b==s])
print("feedforward_nodes:", P.feedforward_nodes, "| motor idx", m, "cb idx", cb, "s idx", s)
print("Pi_motor:", float(jnp.mean(pi[m])), "Pi_sensory:", float(jnp.mean(pi[s])), "Pi_cb:", float(jnp.mean(pi[cb])))
print(f"|W_cs|={float(jnp.linalg.norm(gs.weights[e_cs])):.3f} |W_mc|={float(jnp.linalg.norm(gs.weights[e_mc])):.3f}")
print(f"leak={float(P.leak)} eta_mu={float(P.eta_mu)}")

# manual relax with masked sensory, motor free, perceptual held
hold_mask=_build_hold(P,tuple(hold),{s:mask})
def phi(x): return _phi(act,x)
def phip(x): return _phi_prime(act,x)

mu=list(gs.mu)
mu[s]=jnp.where(mask,goal,gs.mu[s])
ff_set=frozenset(P.feedforward_nodes)
for step in range(0,1501):
    # feedforward cerebellum = its prediction
    pred_cb=jnp.zeros(P.node_sizes[cb],DTYPE)
    for e in incoming[cb]:
        pred_cb=pred_cb+gs.weights[e]@phi(mu[P.edges[e][0]])
    mu[cb]=jnp.where(hold_mask[cb], mu[cb], pred_cb+temporal[cb])
    # predictions
    preds=[jnp.zeros(n,DTYPE) for n in P.node_sizes]
    for e,(src,dst) in enumerate(P.edges):
        preds[dst]=preds[dst]+gs.weights[e]@phi(mu[src])
    pred_full=[preds[j]+temporal[j] for j in range(P.n_nodes)]
    eps=[mu[j]-pred_full[j] for j in range(P.n_nodes)]
    # xi for non-ff
    xi=[None]*P.n_nodes; curv=[None]*P.n_nodes
    for j in range(P.n_nodes):
        if j in ff_set: continue
        xi[j]=pi[j]*eps[j]; curv[j]=pi[j]
    # ff relay (cerebellum)
    acc=jnp.zeros(P.node_sizes[cb],DTYPE); cacc=jnp.zeros(P.node_sizes[cb],DTYPE)
    for e in outgoing[cb]:
        dst=P.edges[e][1]
        acc=acc+gs.weights[e].T@xi[dst]; cacc=cacc+(gs.weights[e]**2).T@curv[dst]
    xi[cb]=phip(mu[cb])*acc; curv[cb]=phip(mu[cb])**2*cacc
    # motor gradient
    g=xi[m]; L=pi[m]
    accm=jnp.zeros(P.node_sizes[m],DTYPE); cum=jnp.zeros(P.node_sizes[m],DTYPE)
    for e in outgoing[m]:
        dst=P.edges[e][1]
        accm=accm+gs.weights[e].T@xi[dst]; cum=cum+(gs.weights[e]**2).T@curv[dst]
    g=g-phip(mu[m])*accm; L=L+phip(mu[m])**2*cum
    g=g+P.leak*mu[m]; L=L+P.leak
    upd=mu[m]-P.eta_mu*g/(L+jnp.finfo(DTYPE).tiny)
    if step in (0,1,2,5,20,80,300,1500):
        eps_tip=jnp.linalg.norm(eps[s][PROP:PROP+TIP])
        xicb=float(jnp.linalg.norm(xi[cb]))
        print(f"  step {step:5d} | mu_motor={np.asarray(mu[m])} |eps_tip|={float(eps_tip):.3f} "
              f"|xi_s|={float(jnp.linalg.norm(xi[s])):.2f} |xi_cb|={xicb:.4f} "
              f"g={np.asarray(g)} L={np.asarray(L)} step={np.asarray(P.eta_mu*g/(L+1e-30))}")
    # update all free non-ff nodes (simplified: only motor + free sensory + others)
    # full update to keep dynamics faithful:
    new=list(mu)
    for j in range(P.n_nodes):
        if j in ff_set: continue
        gj=xi[j]; Lj=pi[j]
        a2=jnp.zeros(P.node_sizes[j],DTYPE); c2=jnp.zeros(P.node_sizes[j],DTYPE)
        for e in outgoing[j]:
            dst=P.edges[e][1]; a2=a2+gs.weights[e].T@xi[dst]; c2=c2+(gs.weights[e]**2).T@curv[dst]
        gj=gj-phip(mu[j])*a2; Lj=Lj+phip(mu[j])**2*c2
        gj=gj+P.leak*mu[j]; Lj=Lj+P.leak
        u=mu[j]-P.eta_mu*gj/(Lj+jnp.finfo(DTYPE).tiny)
        new[j]=jnp.where(hold_mask[j],mu[j],u)
    mu=new

c=jnp.tanh(mu[m]); print("final tip:",np.asarray(fk(c)), "target:",np.asarray(tgt),
                         "err:",float(jnp.linalg.norm(fk(c)-tgt)))
