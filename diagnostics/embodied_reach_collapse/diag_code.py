"""Representable vs learnable, and monotonic vs gaussian tip code.

(A) Capacity probe: can plain Adam fit 2->64->72 tanh MLP to the FK->code map?
    -> if yes, the map is representable; the bottleneck is the PC learning scheme
       (online, local, one relax/step per sample), not capacity.
(B) Does the PC forward model fit a GAUSSIAN tip code better than the MONOTONIC
    one? The monotonic code (near step-functions) was introduced to fix the
    *inversion* gradient; it may be much harder to *regress*. If gaussian fits
    far better, the monotonic code is the forward-model culprit -> a real tension
    (invertible code is unfittable; fittable code isn't invertible).
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx

from core.backend import DTYPE, split_key, make_key
from core.pc_brain import init_pc_brain, pc_brain_learn_forward
from core.pc_graph import pc_graph_clamp, pc_graph_relax, pc_graph_predictions
from sensory.population_code import gaussian_population_encode, monotonic_population_encode

N=12; L1=L2=0.25; HALF=0.5; JR=2.0
PROP=2*2*N; TIP=2*N; SENS=PROP+TIP

def fk(c):
    q=JR*c; return jnp.stack([L1*jnp.cos(q[0])+L2*jnp.cos(q[0]+q[1]),
                              L1*jnp.sin(q[0])+L2*jnp.sin(q[0]+q[1])])
def enc(c, tip_code):
    q=JR*c
    ang=jnp.concatenate([gaussian_population_encode(q[j],N,x_min=-JR,x_max=JR) for j in range(2)])
    vel=jnp.concatenate([gaussian_population_encode(0.0,N,x_min=-JR,x_max=JR) for _ in range(2)])
    t=fk(c)
    tx=tip_code(t[0]); ty=tip_code(t[1])
    return jnp.concatenate([ang,vel,tx,ty]).astype(DTYPE)

mono=lambda v: monotonic_population_encode(v,N,x_min=-HALF,x_max=HALF)
gauss=lambda v: gaussian_population_encode(v,N,x_min=-HALF,x_max=HALF)

# ---- (A) Adam capacity probe on 2->64->72 tanh MLP -------------------------
def capacity_probe(tip_code, steps=4000):
    k=make_key(0)
    def init(k):
        k1,k2,k3,k4=jax.random.split(k,4)
        return dict(W1=jax.random.normal(k1,(64,2))/np.sqrt(2),
                    b1=jnp.zeros(64),
                    W2=jax.random.normal(k3,(SENS,64))/np.sqrt(64),
                    b2=jnp.zeros(SENS))
    p=init(k)
    def fwd(p,c):
        h=jnp.tanh(p['W1']@jnp.tanh(c)+p['b1'])
        return p['W2']@h+p['b2']
    def loss(p,cs,ys):
        pr=jax.vmap(lambda c:fwd(p,c))(cs)
        return jnp.mean((pr-ys)**2)
    # hand-rolled Adam
    mom={k:jnp.zeros_like(v) for k,v in p.items()}
    vel={k:jnp.zeros_like(v) for k,v in p.items()}
    b1,b2,eps,lr=0.9,0.999,1e-8,3e-3
    @jax.jit
    def upd(p,mom,vel,t,cs,ys):
        l,g=jax.value_and_grad(loss)(p,cs,ys)
        for k in p:
            mom[k]=b1*mom[k]+(1-b1)*g[k]
            vel[k]=b2*vel[k]+(1-b2)*g[k]**2
            mh=mom[k]/(1-b1**t); vh=vel[k]/(1-b2**t)
            p[k]=p[k]-lr*mh/(jnp.sqrt(vh)+eps)
        return p,mom,vel,l
    for i in range(steps):
        kc=make_key(i)
        cs=jax.random.uniform(kc,(256,2),DTYPE,-1,1)
        ys=jax.vmap(lambda c:enc(c,tip_code))(cs)
        p,mom,vel,l=upd(p,mom,vel,i+1,cs,ys)
    # eval decoded tip
    grid=jnp.linspace(-HALF,HALF,201)
    codes=jax.vmap(tip_code)(grid)
    def dec(s):
        a=grid[jnp.argmin(jnp.sum((codes-s[PROP:PROP+N][None])**2,1))]
        b=grid[jnp.argmin(jnp.sum((codes-s[PROP+N:][None])**2,1))]
        return jnp.stack([a,b])
    cs=jax.random.uniform(make_key(7),(64,2),DTYPE,-0.7,0.7)
    errs=[float(jnp.linalg.norm(dec(fwd(p,c))-fk(c))) for c in cs]
    return float(l), np.mean(errs)

# ---- (B) PC forward model fit, monotonic vs gaussian tip code --------------
def pc_fit(tip_code, n_babble=20000):
    params,state=init_pc_brain(make_key(0),sensory_size=SENS,motor_size=2,eta_w=0.05,n_relax=30)
    m=params.motor_idx; s=params.sensory_idx; hold=tuple(params.perceptual_nodes)
    a=np.exp(-1/20.0); g=1.5*np.sqrt(1-a*a)
    @eqx.filter_jit
    def chunk(state,bel,keys):
        def step(c,k):
            st,b=c
            b=a*b+g*jax.random.normal(k,(2,),DTYPE)
            st=pc_brain_learn_forward(st,params,b,enc(jnp.tanh(b),tip_code),n_relax=None)
            return (st,b),None
        (state,bel),_=jax.lax.scan(step,(state,bel),keys); return state,bel
    state,_=chunk(state,jnp.zeros(2,DTYPE),jax.random.split(make_key(1),n_babble))
    grid=jnp.linspace(-HALF,HALF,201); codes=jax.vmap(tip_code)(grid)
    def dec(s_):
        aa=grid[jnp.argmin(jnp.sum((codes-s_[PROP:PROP+N][None])**2,1))]
        bb=grid[jnp.argmin(jnp.sum((codes-s_[PROP+N:][None])**2,1))]
        return jnp.stack([aa,bb])
    cs=jax.random.uniform(make_key(7),(64,2),DTYPE,-0.7,0.7)
    tip_err=[]; mse=[]
    for c in cs:
        bel=jnp.arctanh(jnp.clip(c,-0.999,0.999))
        cl=pc_graph_clamp(state.graph,{m:bel})
        rl=pc_graph_relax(cl,params.graph,clamp=(m,)+hold,n_steps=120)
        pred=pc_graph_predictions(rl,params.graph)[s]
        tip_err.append(float(jnp.linalg.norm(dec(pred)-fk(c))))
        mse.append(float(jnp.mean((pred[PROP:]-enc(c,tip_code)[PROP:])**2)))
    return np.mean(tip_err), np.mean(mse)

print("(A) Adam capacity probe (is the FK->code map representable by 2->64->72 tanh?)")
for name,tc in [("monotonic",mono),("gaussian",gauss)]:
    l,e=capacity_probe(tc)
    print(f"    {name:9s}: final train MSE={l:.4f}  decoded tip err={e:.3f}m")

print("\n(B) PC online forward-model fit (the real scheme), 20k babble:")
for name,tc in [("monotonic",mono),("gaussian",gauss)]:
    te,mse=pc_fit(tc)
    print(f"    {name:9s}: decoded tip err={te:.3f}m  tip-block MSE={mse:.4f}")
