"""Is a FIXED random motor->cerebellum expansion (true Marr-Albus) + a learned
linear cerebellum->sensory readout enough to represent the forward model?

This is NOT tuning: it is the biological architecture the codebase comment
already claims ('Marr-Albus granule expansion'). Marr-Albus granule cells are a
fixed, high-D, random nonlinear expansion of mossy-fibre input; only the
granule->Purkinje (cerebellum->sensory) synapse is plastic. There is then NO
deep plastic edge to collapse.

Sweep the expansion scale; report the best linear-readout decoded tip error.
"""
import jax, jax.numpy as jnp
import numpy as np
from core.backend import DTYPE, make_key
from sensory.population_code import gaussian_population_encode, monotonic_population_encode

N=12; L1=L2=0.25; HALF=0.5; JR=2.0; PROP=2*2*N; TIP=2*N; SENS=PROP+TIP; CB=64
mono=lambda v: monotonic_population_encode(v,N,x_min=-HALF,x_max=HALF)
def fk(c):
    q=JR*c; return jnp.stack([L1*jnp.cos(q[0])+L2*jnp.cos(q[0]+q[1]),
                              L1*jnp.sin(q[0])+L2*jnp.sin(q[0]+q[1])])
def tipcode(c):
    t=fk(c); return jnp.concatenate([mono(t[0]),mono(t[1])])
grid=jnp.linspace(-HALF,HALF,201); codes=jax.vmap(mono)(grid)
def dec(y):
    a=grid[jnp.argmin(jnp.sum((codes-y[:N][None])**2,1))]
    b=grid[jnp.argmin(jnp.sum((codes-y[N:][None])**2,1))]
    return jnp.stack([a,b])

# fixed random expansion: cerebellum = tanh(W_mc @ tanh(motor) + b_mc)
k=make_key(0)
k1,k2=jax.random.split(k)
train=jax.random.uniform(make_key(1),(2000,2),DTYPE,-1,1)
test=jax.random.uniform(make_key(2),(256,2),DTYPE,-0.7,0.7)

print("expansion scale -> best linear-readout decoded tip err (test, interior):")
for scale in [0.5,1.0,2.0,4.0,8.0]:
    W=jax.random.normal(k1,(CB,2),DTYPE)*scale
    b=jax.random.normal(k2,(CB,),DTYPE)*scale
    feat=lambda c: jnp.tanh(W@jnp.tanh(c)+b)
    Phi=jax.vmap(feat)(train); Phi=jnp.concatenate([Phi,jnp.ones((Phi.shape[0],1))],1)
    Y=jax.vmap(tipcode)(train)
    Wr=jnp.linalg.solve(Phi.T@Phi+1e-4*jnp.eye(Phi.shape[1]),Phi.T@Y)
    Pte=jax.vmap(feat)(test); Pte=jnp.concatenate([Pte,jnp.ones((Pte.shape[0],1))],1)
    Yh=Pte@Wr
    err=np.mean([float(jnp.linalg.norm(dec(Yh[i])-fk(test[i]))) for i in range(len(test))])
    print(f"   scale={scale:4.1f}: tip err={err:.3f} m")
print("\nIf any scale reaches <0.05 m, a fixed random expansion + plastic readout")
print("represents the forward model with NO deep edge to collapse -> the plastic")
print("motor->cerebellum edge is the wrong design; freezing it is the principled fix.")
