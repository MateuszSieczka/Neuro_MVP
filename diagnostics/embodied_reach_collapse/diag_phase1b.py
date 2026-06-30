"""Correctly-specified Phase 1: granule = FEEDFORWARD fixed random expansion
(non-inferential), only the Purkinje readout cerebellum->sensory is plastic.

Granule activity g(motor) = tanh(W_mc . tanh(motor) + b_mc), W_mc/b_mc FIXED random.
  - learning: delta rule on a FIXED feature basis (cerebellum clamped to g(motor),
              NOT pulled by the sensory clamp) -> trains the same features inference
              uses. dW_cs = eta (reaff - pred) . phi(g(motor)).
  - inversion: cerebellum is a deterministic function of motor (no extra free DOF),
              so reaching is a clean 2-D inversion: min_motor ||goal - readout(g(motor))||^2
              on the tip channels.

This is the architecture the diagnosis points to. If success is high, Phase 1
(as corrected) works; the core change is: granule node is feedforward/high-precision,
not a free latent, and W_mc is a frozen Marr-Albus expansion.
"""
import jax, jax.numpy as jnp
import numpy as np
from core.backend import DTYPE, make_key, split_key
from sensory.population_code import gaussian_population_encode, monotonic_population_encode

N=12; L1=L2=0.25; HALF=0.5; JR=2.0; PROP=2*2*N; TIP=2*N; SENS=PROP+TIP; CB=64
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
def dec(s):
    a=grid[jnp.argmin(jnp.sum((codes-s[PROP:PROP+N][None])**2,1))]
    b=grid[jnp.argmin(jnp.sum((codes-s[PROP+N:][None])**2,1))]
    return jnp.stack([a,b])

g_exp=1.3                                   # ELM/random-feature scaling (pre-act std ~1), not tuned to reach
kw,kb=split_key(make_key(42))
Wmc=jax.random.normal(kw,(CB,2),DTYPE)*g_exp
bmc=jax.random.normal(kb,(CB,),DTYPE)*g_exp
def granule(c):                             # feedforward fixed expansion (mossy->granule)
    return jnp.tanh(Wmc@jnp.tanh(c)+bmc)

# ---- learn readout by ONLINE delta rule on OU babble (cerebellum clamped to g) ----
a=np.exp(-1/20.0); gg=1.5*np.sqrt(1-a*a)
def train_online(eta):
    Wcs=jnp.zeros((SENS,CB),DTYPE); bs=jnp.zeros(SENS,DTYPE)
    @jax.jit
    def step(carry,k):
        Wcs,bs,bel=carry
        bel=a*bel+gg*jax.random.normal(k,(2,),DTYPE)
        c=jnp.tanh(bel); phi=granule(bel)       # granule driven by the motor belief (mossy)
        pred=Wcs@phi+bs; e=enc(c)-pred
        Wcs=Wcs+eta*jnp.outer(e,phi); bs=bs+eta*e
        return (Wcs,bs,bel),None
    (Wcs,bs,_),_=jax.lax.scan(step,(Wcs,bs,jnp.zeros(2,DTYPE)),jax.random.split(make_key(1),20000))
    return Wcs,bs
Wcs,bs=train_online(0.01)
def readout(bel): return Wcs@granule(bel)+bs

# forward-model accuracy
cs=jax.random.uniform(make_key(7),(64,2),DTYPE,-0.7,0.7)
fwd=np.mean([float(jnp.linalg.norm(dec(readout(jnp.arctanh(jnp.clip(c,-0.999,0.999))))-fk(c))) for c in cs])

# ---- reach: clean 2-D inversion over motor belief (cerebellum deterministic) ----
def invert(readfn, target, steps=400, lr=0.3):
    tx,ty=mono(target[0]),mono(target[1]); goal=jnp.concatenate([tx,ty])
    def loss(bel):
        pr=readfn(bel)[PROP:]; return jnp.sum((pr-goal)**2)
    bel=jnp.zeros(2,DTYPE); g=jax.grad(loss)
    for _ in range(steps): bel=bel-lr*g(bel)
    return jnp.tanh(bel)
rr=jax.random.uniform(make_key(7),(32,),DTYPE,0.15,0.45)
th=jax.random.uniform(make_key(8),(32,),DTYPE,-np.pi,np.pi)
tg=jnp.stack([rr*jnp.cos(th),rr*jnp.sin(th)],1)
errs=np.array([float(jnp.linalg.norm(fk(invert(readout,t))-t)) for t in tg])

# least-squares ceiling for the readout (best possible on these fixed features)
train=jax.random.uniform(make_key(2),(3000,2),DTYPE,-1,1)
Phi=jnp.concatenate([jax.vmap(lambda c:granule(jnp.arctanh(jnp.clip(c,-0.999,0.999))))(train),
                     jnp.ones((3000,1))],1)
Y=jax.vmap(enc)(train)
Wls=jnp.linalg.solve(Phi.T@Phi+1e-4*jnp.eye(CB+1),Phi.T@Y)
def readout_ls(bel):
    phi=jnp.concatenate([granule(bel),jnp.ones((1,),DTYPE)]); return (phi@Wls)
fwd_ls=np.mean([float(jnp.linalg.norm(dec(readout_ls(jnp.arctanh(jnp.clip(c,-0.999,0.999))))-fk(c))) for c in cs])
errs_ls=np.array([float(jnp.linalg.norm(fk(invert(readout_ls,t))-t)) for t in tg])

print("Phase 1 (corrected): feedforward fixed-random granule + plastic Purkinje readout")
print(f"  forward-model decoded tip err: online delta-rule(eta=.01) = {fwd:.3f} m | least-squares ceiling = {fwd_ls:.3f} m")
print(f"  reach 2-D inversion, online readout: mean FK err = {errs.mean():.3f} m  "
      f"success(<0.05m) = {np.mean(errs<0.05):5.1%}  median = {np.median(errs):.3f} m")
print(f"  reach 2-D inversion, LS  readout:    mean FK err = {errs_ls.mean():.3f} m  "
      f"success(<0.05m) = {np.mean(errs_ls<0.05):5.1%}  median = {np.median(errs_ls):.3f} m")
