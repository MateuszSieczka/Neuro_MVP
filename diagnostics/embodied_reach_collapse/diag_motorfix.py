"""Confirm: the DIAGONAL-Newton preconditioner on the flat-prior MOTOR node is
what freezes the inversion. Swap only the motor-node metric and re-measure reach.

The cerebellum is feedforward + sensory readout is linear, so the tip-code
forward map is analytic:
    z   = W_mc @ tanh(mu) + b_mc
    g   = tanh(z)                       # granule activity
    tip = W_cs[tip,:] @ g + b_s[tip]
    J   = W_cs[tip,:] @ diag(g') @ W_mc @ diag(tanh'(mu))     # (TIP, 2)

Inversion variants (all gradient flow identical; only the motor METRIC differs):
  diag-newton   : step = eta * JtPe / diag(JtPJ)      <- the substrate's rule (frozen)
  plain         : step = eta_small * JtPe             <- raw gradient, fixed lr
  normalized    : step = eta * JtPe / ||JtPe||        <- scale-free direction
  gauss-newton  : step = (JtPJ + lambda I)^-1 JtPe    <- full 2x2 natural inversion

If diag-newton ~0% but gauss-newton/normalized reach well on the SAME model,
the preconditioner (not the model) is the reach bug.
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx
from core.backend import DTYPE, make_key
from core.pc_brain import init_pc_brain
from core.pc_graph import pc_graph_clamp, pc_graph_relax, _phi
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
hold=tuple(params.perceptual_nodes); e_cs=edge(params,cb,s); e_mc=edge(params,m,cb)

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
Wcs=jnp.asarray(Wls[:-1].T,DTYPE); bs=jnp.asarray(Wls[-1],DTYPE)
Wmc=state.graph.weights[e_mc]; bmc=state.graph.bias[cb]
Pi_s=jnp.mean(state.graph.pi[s])      # scalar precision used by the substrate (==1.0 here)

Wcs_tip=Wcs[PROP:PROP+TIP,:]; bs_tip=bs[PROP:PROP+TIP]
def forward_tip(mu):
    z=Wmc@jnp.tanh(mu)+bmc; g=jnp.tanh(z); return Wcs_tip@g+bs_tip
def jac_tip(mu):
    z=Wmc@jnp.tanh(mu)+bmc; gp=1-jnp.tanh(z)**2; mp=1-jnp.tanh(mu)**2
    return (Wcs_tip*gp[None,:])@(Wmc*mp[None,:])     # (TIP,2)

rr=jax.random.uniform(make_key(11),(64,),DTYPE,0.15,0.42)
th=jax.random.uniform(make_key(12),(64,),DTYPE,-np.pi,np.pi)
targets=jnp.stack([rr*jnp.cos(th),rr*jnp.sin(th)],1)

def invert(variant, steps=400):
    def one(tgt):
        goal=jnp.concatenate([mono(tgt[0]),mono(tgt[1])])
        def body(_,mu):
            e=goal-forward_tip(mu); J=jac_tip(mu)
            JtPe=J.T@(Pi_s*e)
            if variant=='diag-newton':
                L=jnp.sum((J**2)*Pi_s,axis=0)          # diag(J^T Pi J)
                step=0.1*JtPe/(L+jnp.finfo(DTYPE).tiny)
            elif variant=='plain':
                step=3e-4*JtPe
            elif variant=='normalized':
                step=0.05*JtPe/(jnp.linalg.norm(JtPe)+1e-8)
            elif variant=='gauss-newton':
                H=J.T@(Pi_s*J)+1e-2*jnp.eye(2,dtype=DTYPE)
                step=jnp.linalg.solve(H,JtPe)
            return mu+step
        mu=jax.lax.fori_loop(0,steps,body,jnp.zeros(2,DTYPE))
        return mu
    cmds=jax.vmap(one)(targets)
    tips=jax.vmap(lambda b: fk(jnp.tanh(b)))(cmds)
    err=jnp.linalg.norm(tips-targets,axis=1)
    return float(jnp.mean(err)),float(jnp.mean(err<0.05)),float(jnp.median(err)),float(jnp.mean(jnp.abs(jnp.tanh(cmds))))

print(f"LS readout |W_cs|={float(jnp.linalg.norm(Wcs)):.0f}  Pi_s={float(Pi_s):.2f}")
print(f"  {'motor metric':>14} | {'reach_err':>9} {'succ':>5} {'med':>5} {'mean|cmd|':>9}")
for v in ['diag-newton','plain','normalized','gauss-newton']:
    er,sc,md,mc=invert(v)
    print(f"  {v:>14} | {er:9.3f} {sc:5.1%} {md:5.3f} {mc:9.3f}")
print("\n  diag-newton == substrate's current rule.  If it is ~0% and gauss-newton/")
print("  normalized reach well on the IDENTICAL model, the motor preconditioner is the bug.")
