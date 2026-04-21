"""Diagnostic: isolate which part of action_brain_step hangs."""
import sys, time, os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

print("importing...", flush=True)
t0 = time.time()
import jax
import jax.numpy as jnp
print(f"  jax import: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
from core.backend import BackendContext
from core.brain_graph import init_action_brain_params, init_action_brain_state
from sensory.retina import RetinaConfig
from sensory.sensory_stack import init_sensory_stack_params, sensory_stack_step
from embodiment.visual_grid import VisualGridBody
print(f"  module imports: {time.time()-t0:.1f}s", flush=True)

ctx = BackendContext(dt=1.0)
cfg = RetinaConfig(fovea_size=8, n_pyramid=2, periphery_tile=4)

t0 = time.time()
body = VisualGridBody.create(
    jax.random.PRNGKey(0), size=3, start=(0, 0), goal=(2, 2),
    tex_size=32, max_steps=40, retina_cfg=cfg,
)
print(f"  body create: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
ss_params = init_sensory_stack_params(
    ctx, retina_cfg=cfg, n_l4=16, n_l23_state=8, n_l23_error=4, n_l5=4,
)
print(f"  ss_params: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
params = init_action_brain_params(
    ctx, sensory_size=0, n_body_actions=4,
    sensory_stack_params=ss_params, substeps=2,
    n_tc=16, n_ct=8, n_trn=8,
    cortex_n_l4=16, cortex_n_l23_state=16, cortex_n_l23_error=16, cortex_n_l5=8,
    critic_hidden=16, wm_hidden=16, wm_n_error=16,
    mossy_size=8, cerebellum_n_purkinje=8,
)
state = init_action_brain_state(jax.random.PRNGKey(100), params)
print(f"  brain init: {time.time()-t0:.1f}s", flush=True)

body, sample = body.reset(jax.random.PRNGKey(7))
image = sample.info["image"]
fix = sample.info["fixation_xy"]

# --- Test 1: single sensory_stack_step (JIT'd) ---
print("TEST 1: single sensory_stack_step...", flush=True)
t0 = time.time()
o = sensory_stack_step(
    state.sensory_stack, params.sensory_stack, ctx,
    image, fix, ach=0.5, da=0.5, ne=0.5, apply_ipool_stdp=True,
)
print(f"  sensory_stack_step: {time.time()-t0:.1f}s  l4={o.l4_rate.shape}", flush=True)

# --- Test 2: scan over sensory_stack_step ---
print("TEST 2: scan(sensory_stack_step, length=2)...", flush=True)
t0 = time.time()
def _ss_body(ss_st, _):
    o2 = sensory_stack_step(
        ss_st, params.sensory_stack, ctx, image, fix,
        ach=0.5, da=0.5, ne=0.5, apply_ipool_stdp=True,
    )
    return o2.state, (o2.l4_rate, o2.pe_rate)
new_ss, (l4h, peh) = jax.lax.scan(_ss_body, state.sensory_stack, None, length=2)
print(f"  sensory scan: {time.time()-t0:.1f}s", flush=True)

# --- Test 3: single _perceive_substep (JIT'd) ---
print("TEST 3: single _perceive_substep...", flush=True)
from core.brain_graph import _perceive_substep
sensory = jnp.zeros((params.sensory_size,), jnp.float32)
t0 = time.time()
new_st, readouts = _perceive_substep(
    state, params, ctx, sensory,
    td_error=jnp.float32(0.0), novelty=jnp.float32(0.0),
    key=jax.random.PRNGKey(42),
)
print(f"  _perceive_substep: {time.time()-t0:.1f}s", flush=True)

# --- Test 4: scan over _perceive_substep ---
print("TEST 4: scan(_perceive_substep, length=2)...", flush=True)
t0 = time.time()
k_scan = jax.random.split(jax.random.PRNGKey(99), 2)
def scan_body(st, step_key):
    new_s, ro = _perceive_substep(
        st, params, ctx, sensory,
        td_error=jnp.float32(0.0), novelty=jnp.float32(0.0),
        key=step_key,
    )
    return new_s, ro
state2, readouts2 = jax.lax.scan(scan_body, state, k_scan)
print(f"  perceive scan: {time.time()-t0:.1f}s", flush=True)

print("ALL DONE", flush=True)
