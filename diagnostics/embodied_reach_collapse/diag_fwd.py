"""Why won't motor->cerebellum->sensory fit?

Hypothesis: the cerebellum is a free latent squeezed between TWO clamps during
learn_forward (motor clamped, sensory clamped). It settles OFF the
motor-reachable manifold (pulled by the sensory clamp), and cb->sensory is
trained to decode that off-manifold state. At forward-pass time the cerebellum
is ON-manifold (only motor drives it), so the decode is wrong.

Test: measure eps_cb (cerebellum's error vs its motor-driven prediction) at the
dual-clamp LEARNING equilibrium vs the single-clamp FORWARD-PASS equilibrium,
plus raw tip-block fit, as a function of n_relax. Also compare against a plain
supervised 2-layer tanh net trained by Adam on the same map (capacity probe).
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx

from core.backend import DTYPE, split_key, make_key
from core.pc_brain import init_pc_brain, pc_brain_learn_forward
from core.pc_graph import pc_graph_clamp, pc_graph_relax, pc_graph_predictions

from sensory.population_code import gaussian_population_encode, monotonic_population_encode

N_CELLS=12; L1=L2=0.25; HALF=0.5; JR=2.0
PROP=2*2*N_CELLS; TIP=2*N_CELLS; SENS=PROP+TIP

def fk(c):
    q=JR*c; return jnp.stack([L1*jnp.cos(q[0])+L2*jnp.cos(q[0]+q[1]),
                              L1*jnp.sin(q[0])+L2*jnp.sin(q[0]+q[1])])
def enc(c):
    q=JR*c
    ang=jnp.concatenate([gaussian_population_encode(q[j],N_CELLS,x_min=-JR,x_max=JR) for j in range(2)])
    vel=jnp.concatenate([gaussian_population_encode(0.0,N_CELLS,x_min=-JR,x_max=JR) for _ in range(2)])
    t=fk(c)
    tx=monotonic_population_encode(t[0],N_CELLS,x_min=-HALF,x_max=HALF)
    ty=monotonic_population_encode(t[1],N_CELLS,x_min=-HALF,x_max=HALF)
    return jnp.concatenate([ang,vel,tx,ty]).astype(DTYPE)

def run(n_relax, n_babble=20000, eta_w=0.05):
    params,state=init_pc_brain(make_key(0),sensory_size=SENS,motor_size=2,eta_w=eta_w,n_relax=n_relax)
    cb=params.cerebellum_idx; m=params.motor_idx; s=params.sensory_idx
    hold=tuple(params.perceptual_nodes)
    alpha=np.exp(-1/20.0); gain=1.5*np.sqrt(1-alpha*alpha)
    @eqx.filter_jit
    def chunk(state,belief,keys):
        def step(c,k):
            st,bel=c
            bel=alpha*bel+gain*jax.random.normal(k,(2,),DTYPE)
            st=pc_brain_learn_forward(st,params,bel,enc(jnp.tanh(bel)),n_relax=None)
            return (st,bel),None
        (state,belief),_=jax.lax.scan(step,(state,belief),keys)
        return state,belief
    belief=jnp.zeros(2,DTYPE)
    state,belief=chunk(state,belief,jax.random.split(make_key(1),n_babble))

    # eval
    cmds=jax.random.uniform(make_key(99),(64,2),DTYPE,-0.7,0.7)
    fwd_tip=[]; eps_cb_fwd=[]; eps_cb_learn=[]; tipmse=[]
    for c in cmds:
        bel=jnp.arctanh(jnp.clip(c,-0.999,0.999))
        reaff=enc(c)
        # forward pass: clamp motor only
        cl=pc_graph_clamp(state.graph,{m:bel})
        rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=max(80,n_relax*3))
        pred=pc_graph_predictions(rl,params.graph)[s]
        # eps_cb at forward pass = mu_cb - W_mc phi(motor)
        from core.pc_module import _phi
        eps_cb_fwd.append(float(jnp.linalg.norm(rl.mu[cb]-pred_cb(state,params,rl,m,cb))))
        # tip block raw mse
        tipmse.append(float(jnp.mean((pred[PROP:]-reaff[PROP:])**2)))
        # decode
        fwd_tip.append(float(jnp.linalg.norm(decode_tip(pred)-fk(c))))
        # learning equilibrium: clamp motor AND sensory, relax cb
        cl2=pc_graph_clamp(state.graph,{m:bel,s:reaff})
        rl2=pc_graph_relax(cl2,params.graph,clamp=(m,s)+hold,n_steps=max(80,n_relax*3))
        eps_cb_learn.append(float(jnp.linalg.norm(rl2.mu[cb]-pred_cb(state,params,rl2,m,cb))))
    wnorms={e:(float(jnp.max(jnp.abs(state.graph.weights[e])))) for e in range(params.graph.n_edges)}
    return dict(fwd_tip=np.mean(fwd_tip), tipmse=np.mean(tipmse),
                eps_cb_fwd=np.mean(eps_cb_fwd), eps_cb_learn=np.mean(eps_cb_learn))

def pred_cb(state,params,relaxed,m,cb):
    from core.pc_module import _phi
    # find motor->cerebellum edge
    for e,(src,dst) in enumerate(params.graph.edges):
        if src==m and dst==cb:
            return relaxed.weights[e]@_phi(params.graph.act,relaxed.mu[m])+relaxed.bias[cb]
    raise RuntimeError

_grid=jnp.linspace(-HALF,HALF,201)
_codes=jax.vmap(lambda v:monotonic_population_encode(v,N_CELLS,x_min=-HALF,x_max=HALF))(_grid)
def decode_axis(code):
    return _grid[jnp.argmin(jnp.sum((_codes-code[None])**2,axis=1))]
def decode_tip(sens):
    return jnp.stack([decode_axis(sens[PROP:PROP+N_CELLS]),decode_axis(sens[PROP+N_CELLS:])])

print("n_relax sweep (does more settling fix the forward model?)")
for nr in [30,80,200]:
    r=run(nr)
    print(f"  n_relax={nr:3d}: fwd_tip_err={r['fwd_tip']:.3f}m  tip_block_mse={r['tipmse']:.4f}  "
          f"|eps_cb| fwd-pass={r['eps_cb_fwd']:.3f}  learn-eq={r['eps_cb_learn']:.3f}")
print("\nIf eps_cb_learn >> eps_cb_fwd: cerebellum sits OFF the motor manifold while")
print("learning (pulled by the sensory clamp) -> cb->sensory decodes a state the")
print("forward pass never reproduces -> structural off-manifold training mismatch.")
