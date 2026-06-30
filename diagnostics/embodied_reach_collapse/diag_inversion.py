"""Local diagnostic: is the reach plateau the forward MODEL or the INVERSION?

No MuJoCo needed — replace the MJX arm with a synthetic 2-link FK (analytic),
encode reafference with the *exact* sensory layout (proprio gaussian + tip
monotonic), and drive the *real* core graph through the *real* babble/reach
calls (pc_brain_learn_forward / pc_brain_act).

Discriminator:
  * forward-model accuracy: decode predicted tip-code vs true FK tip.
  * inversion accuracy:      FK(inferred command) vs target.
  * off-manifold check:      cerebellum error magnitude at the inversion
                             equilibrium, and "model image of inferred motor"
                             (forward pass of the read-out command) vs target.
If the forward model is accurate but inversion still misses → the inversion is
structurally underdetermined (the wide free cerebellum absorbs the goal).
"""
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx

from core.backend import DTYPE, split_key, make_key
from core.pc_brain import init_pc_brain, pc_brain_act, pc_brain_learn_forward
from core.pc_graph import pc_graph_clamp, pc_graph_relax, pc_graph_predictions, pc_graph_errors
from sensory.population_code import gaussian_population_encode, monotonic_population_encode

N_CELLS = 12
L1 = L2 = 0.25
HALF = 0.5
JOINT_RANGE = 2.0
PROPRIO = 2 * 2 * N_CELLS          # 48
TIP = 2 * N_CELLS                  # 24
SENSORY = PROPRIO + TIP            # 72

def fk(command):
    """command in [-1,1]^2 -> tip xy.  q = JOINT_RANGE*command."""
    q = JOINT_RANGE * command
    q0, q1 = q[0], q[1]
    x = L1 * jnp.cos(q0) + L2 * jnp.cos(q0 + q1)
    y = L1 * jnp.sin(q0) + L2 * jnp.sin(q0 + q1)
    return jnp.stack([x, y])

def encode_sensory(command):
    """Settled reafference: proprio(angle gaussian, vel=0 gaussian) + tip monotonic."""
    q = JOINT_RANGE * command
    ang = jnp.concatenate([
        gaussian_population_encode(q[j], N_CELLS, x_min=-JOINT_RANGE, x_max=JOINT_RANGE)
        for j in range(2)])
    vel = jnp.concatenate([
        gaussian_population_encode(0.0, N_CELLS, x_min=-JOINT_RANGE, x_max=JOINT_RANGE)
        for _ in range(2)])
    tip = fk(command)
    tx = monotonic_population_encode(tip[0], N_CELLS, x_min=-HALF, x_max=HALF)
    ty = monotonic_population_encode(tip[1], N_CELLS, x_min=-HALF, x_max=HALF)
    return jnp.concatenate([ang, vel, tx, ty]).astype(DTYPE)

def tip_pref(tip):
    pref = jnp.zeros(SENSORY, DTYPE)
    tx = monotonic_population_encode(tip[0], N_CELLS, x_min=-HALF, x_max=HALF)
    ty = monotonic_population_encode(tip[1], N_CELLS, x_min=-HALF, x_max=HALF)
    pref = pref.at[PROPRIO:PROPRIO+N_CELLS].set(tx)
    pref = pref.at[PROPRIO+N_CELLS:SENSORY].set(ty)
    return pref

TIP_MASK = jnp.zeros(SENSORY, bool).at[PROPRIO:SENSORY].set(True)

# Decode a tip-code (the 24-d tip block) to xy by nearest code on a fine grid.
_grid = jnp.linspace(-HALF, HALF, 201)
_codes = jax.vmap(lambda v: monotonic_population_encode(v, N_CELLS, x_min=-HALF, x_max=HALF))(_grid)
def decode_axis(code):
    d = jnp.sum((_codes - code[None, :]) ** 2, axis=1)
    return _grid[jnp.argmin(d)]
def decode_tip(sens):
    tx = sens[PROPRIO:PROPRIO+N_CELLS]
    ty = sens[PROPRIO+N_CELLS:SENSORY]
    return jnp.stack([decode_axis(tx), decode_axis(ty)])

# ------------------------------------------------------------------ build
key = make_key(0)
k_brain, key = split_key(key)
params, state = init_pc_brain(k_brain, sensory_size=SENSORY, motor_size=2,
                              eta_w=0.05, n_relax=30)
cb_idx = params.cerebellum_idx
m_idx = params.motor_idx
s_idx = params.sensory_idx

# ------------------------------------------------------------------ babble
N_BABBLE = 20000
TAU, SIGMA = 20.0, 1.5
alpha = np.exp(-1.0/TAU)
gain = SIGMA*np.sqrt(1-alpha*alpha)
belief = jnp.zeros(2, DTYPE)

@eqx.filter_jit
def babble_chunk(state, belief, keys):
    def step(carry, k):
        state, belief = carry
        belief = alpha*belief + gain*jax.random.normal(k, (2,), DTYPE)
        command = jnp.tanh(belief)
        reaff = encode_sensory(command)
        state = pc_brain_learn_forward(state, params, belief, reaff, n_relax=None)
        return (state, belief), None
    (state, belief), _ = jax.lax.scan(step, (state, belief), keys)
    return state, belief

def babble_n(state, belief, key, n):
    key, kc = split_key(key)
    keys = jax.random.split(kc, n)
    state, belief = babble_chunk(state, belief, keys)
    return state, belief, key

print(f"graph: sensory={SENSORY} motor=2 cerebellum={params.graph.node_sizes[cb_idx]}")
print("babbling", N_BABBLE, "cycles (CPU, ~minutes)...")
import time
checkpoints = [0, 5000, 10000, 20000]
done = 0
results = {}
for ckpt in checkpoints:
    if ckpt > done:
        t0 = time.time()
        state, belief, key = babble_n(state, belief, key, ckpt - done)
        done = ckpt
        print(f"  babbled to {ckpt} ({time.time()-t0:.1f}s)")
    # ---- forward-model accuracy on a test grid of commands
    test_cmds = jax.random.uniform(make_key(99), (64, 2), DTYPE, -1.0, 1.0)
    fwd_err = []
    for c in test_cmds:
        true_tip = fk(c)
        # model image: clamp motor=atanh-ish belief... use belief s.t. tanh=c
        bel = jnp.arctanh(jnp.clip(c, -0.999, 0.999))
        clamped = pc_graph_clamp(state.graph, {m_idx: bel})
        relaxed = pc_graph_relax(clamped, params.graph,
                                 clamp=(m_idx,) + tuple(params.perceptual_nodes),
                                 n_steps=60)
        pred = pc_graph_predictions(relaxed, params.graph)[s_idx]
        dec = decode_tip(pred)
        fwd_err.append(float(jnp.linalg.norm(dec - true_tip)))
    fwd_err = np.array(fwd_err)
    # ---- inversion accuracy on a grid of targets in the reachable annulus
    k_t = make_key(7)
    rr = jax.random.uniform(k_t, (32,), DTYPE, 0.15, 0.45)
    th = jax.random.uniform(make_key(8), (32,), DTYPE, -np.pi, np.pi)
    targets = jnp.stack([rr*jnp.cos(th), rr*jnp.sin(th)], axis=1)
    inv_err, manifold_err, cb_eps, cmd_sat, model_img_err = [], [], [], [], []
    for tg in targets:
        pref = tip_pref(tg)
        act = pc_brain_act(state, params, pref, preference_mask=TIP_MASK, n_relax=80)
        cmd = act.joint_command                     # tanh(motor)
        achieved = fk(cmd)
        inv_err.append(float(jnp.linalg.norm(achieved - tg)))
        cmd_sat.append(float(jnp.max(jnp.abs(cmd))))
        # off-manifold: relax with motor clamped to the inferred belief; read model tip
        bel = act.motor_belief
        clamped = pc_graph_clamp(state.graph, {m_idx: bel})
        relaxed = pc_graph_relax(clamped, params.graph,
                                 clamp=(m_idx,) + tuple(params.perceptual_nodes),
                                 n_steps=80)
        model_tip = decode_tip(pc_graph_predictions(relaxed, params.graph)[s_idx])
        model_img_err.append(float(jnp.linalg.norm(model_tip - tg)))
        # cerebellum error at the inversion equilibrium (re-run the inversion relax,
        # read eps_cb): approximate via the clamped-goal relax
        # (reuse act's internal eq is not exposed; recompute)
    inv_err = np.array(inv_err); cmd_sat = np.array(cmd_sat)
    model_img_err = np.array(model_img_err)
    succ = float(np.mean(inv_err < 0.05))
    results[ckpt] = dict(fwd=fwd_err.mean(), inv=inv_err.mean(), succ=succ,
                         sat=cmd_sat.mean(), modimg=model_img_err.mean())
    print(f"[babble={ckpt:6d}] fwd-model tip err={fwd_err.mean():.3f}m | "
          f"inversion FK err={inv_err.mean():.3f}m  success={succ:5.1%} | "
          f"model-image err={model_img_err.mean():.3f}m  mean|cmd|max={cmd_sat.mean():.2f}")

print("\nSUMMARY")
print("  fwd  = how well the trained forward model predicts the tip (decoded).")
print("  inv  = distance from target of FK(inferred command)  <-- the reach metric.")
print("  modimg = distance from target of the MODEL's own image of the inferred command.")
print("           modimg<<inv  => model accurate, body-vs-model mismatch.")
print("           modimg~=inv & both large, but fwd small => inversion underdetermined.")
