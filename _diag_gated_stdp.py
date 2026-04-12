"""
GATED STDP SIMULATION: What would happen if we remove the REINFORCE
override and use STDP eligibility + gate_eligibility() instead?
"""
import numpy as np
np.random.seed(42)

from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena.task_config import get as get_task

task = get_task('CartPole-v1')
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent = make_agent(task, env)

print("=" * 80)
print("GATED STDP SIMULATION")
print("=" * 80)

# ── Simulate act() WITHOUT the REINFORCE override ───────────────────
state = env.reset()
agent.reset()

# Run substep loop (same as act() but skip REINFORCE override)
agent.actor.reset_spike_counts()
agent.critic.reset_spike_counts(agent._n_substeps)
pop_rates = agent._pop_encoder.encode(state.astype(np.float32))
for _ in range(agent._n_substeps):
    enc = agent._poisson.encode(pop_rates)
    sensory = agent._build_sensory_inputs(enc, state.astype(np.float32))
    agent.network.step(sensory_inputs=sensory, neuromodulator=agent.neuromod, attention=None)

action = agent.actor.get_action()
print(f"Action: {action}")

# ── Show STDP eligibility BEFORE gating ────────────────────────────
e_d1 = agent.actor.e_d1.copy()
e_d2 = agent.actor.e_d2.copy()
npa = agent.actor.n_per_action

print(f"\n### STDP Eligibility (before gating) ###")
for a in range(2):
    s = a * npa
    block = e_d1[:, s:s+npa]
    print(f"  Action {a}: mean={np.mean(block):.6f}, |mean|={np.mean(np.abs(block)):.6f}, "
          f"neg%={np.mean(block < -1e-8):.4f}, pos%={np.mean(block > 1e-8):.4f}")

# ── Apply gating ──────────────────────────────────────────────────
agent.actor.gate_eligibility(action)
e_d1_gated = agent.actor.e_d1.copy()

print(f"\n### STDP Eligibility (after gating, selected={action}) ###")
for a in range(2):
    s = a * npa
    block = e_d1_gated[:, s:s+npa]
    label = "SELECTED" if a == action else "ZEROED  "
    print(f"  {label} Action {a}: mean={np.mean(block):.6f}, |mean|={np.mean(np.abs(block)):.6f}, "
          f"neg%={np.mean(block < -1e-8):.4f}, pos%={np.mean(block > 1e-8):.4f}")

# ── Simulate what weight update would look like ──────────────────
td = 1.0  # Typical positive TD
lr = agent._bg_config.actor_lr
dw_d1 = lr * td * e_d1_gated
print(f"\n### Predicted weight update (TD=+1.0) ###")
for a in range(2):
    s = a * npa
    block = dw_d1[:, s:s+npa]
    label = "SELECTED" if a == action else "ZEROED  "
    print(f"  {label} Action {a}: dw mean={np.mean(block):.8f}, |dw| mean={np.mean(np.abs(block)):.8f}")

# Negative TD
td_neg = -1.0
dw_d1_neg = lr * td_neg * e_d1_gated
print(f"\n### Predicted weight update (TD=-1.0) ###")
for a in range(2):
    s = a * npa
    block = dw_d1_neg[:, s:s+npa]
    label = "SELECTED" if a == action else "ZEROED  "
    print(f"  {label} Action {a}: dw mean={np.mean(block):.8f}, |dw| mean={np.mean(np.abs(block)):.8f}")

# ── How many episodes until weights differentiate? ──────────────
print(f"\n### Weight differentiation estimate ###")
sel_block = e_d1_gated[:, action*npa:(action+1)*npa]
dw_per_step = lr * 1.0 * np.mean(np.abs(sel_block))
initial_weight = np.mean(agent.actor.w_d1)
print(f"  |dw| per step per selected synapse: {dw_per_step:.6f}")
print(f"  Initial weight mean: {initial_weight:.4f}")
print(f"  Per episode (~20 steps): {dw_per_step * 20:.6f}")
print(f"  Episodes to change weights by 10% of init: {initial_weight * 0.1 / (dw_per_step * 20):.1f}")

# ── Show D1 vs D2 update asymmetry ──────────────────────────────
print(f"\n### D1 vs D2 update asymmetry ###")
e_d2_gated = agent.actor.e_d2.copy()  # Already gated (shared gating)
for pathway, elig in [("D1", e_d1_gated), ("D2", e_d2_gated)]:
    sel_block = elig[:, action*npa:(action+1)*npa]
    print(f"  {pathway} selected: mean={np.mean(sel_block):.6f}, |mean|={np.mean(np.abs(sel_block)):.6f}")
    # D1: dw = lr * td * e
    # D2: dw = d2_lr * (-td) * e (where d2_lr = lr * d2_ltd_protection if td>0)
    if pathway == "D1":
        dw = lr * 1.0 * sel_block
    else:
        dw = lr * 0.5 * (-1.0) * sel_block  # td>0, d2_ltd_protection=0.5
    print(f"    With td=+1: dw mean={np.mean(dw):.8f}")

# ── Run 50 steps WITHOUT override, track differentiation ─────────
print(f"\n### Simulated 50-step learning (no override) ###")
# Reset for clean simulation
env2 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent2 = make_agent(task, env2)

# Monkey-patch: disable the REINFORCE override
orig_set_pg = agent2._set_actor_policy_gradient
def noop_pg(rates): pass  # Do nothing
agent2._set_actor_policy_gradient = noop_pg

state = env2.reset()
agent2.reset()
actions_taken = []
for step in range(50):
    a = agent2.act(state)
    # Apply gating instead of PG override
    agent2.actor.gate_eligibility(a)
    actions_taken.append(a)
    
    ns, r, done, info = env2.step(a)
    agent2.observe(state, a, r, ns, done, info)
    state = ns
    if done:
        state = env2.reset()
        agent2.reset()

w_d1 = agent2.actor.w_d1
npa = agent2.actor.n_per_action
a0_mean = np.mean(w_d1[:, :npa])
a1_mean = np.mean(w_d1[:, npa:2*npa])
cos_sim = np.dot(w_d1[:, :npa].mean(1), w_d1[:, npa:2*npa].mean(1)) / (
    np.linalg.norm(w_d1[:, :npa].mean(1)) * np.linalg.norm(w_d1[:, npa:2*npa].mean(1)) + 1e-10)

print(f"  After 50 steps:")
print(f"    Action distribution: {np.bincount(actions_taken, minlength=2)}")
print(f"    w_d1 action 0 mean: {a0_mean:.4f}")
print(f"    w_d1 action 1 mean: {a1_mean:.4f}")
print(f"    cosine(w_d1[a0], w_d1[a1]): {cos_sim:.4f}")

# Compare with the override version
env3 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent3 = make_agent(task, env3)
np.random.seed(42)
state = env3.reset()
agent3.reset()
actions_base = []
for step in range(50):
    a = agent3.act(state)
    actions_base.append(a)
    ns, r, done, info = env3.step(a)
    agent3.observe(state, a, r, ns, done, info)
    state = ns
    if done:
        state = env3.reset()
        agent3.reset()

w_d1_base = agent3.actor.w_d1
a0_base = np.mean(w_d1_base[:, :npa])
a1_base = np.mean(w_d1_base[:, npa:2*npa])
cos_base = np.dot(w_d1_base[:, :npa].mean(1), w_d1_base[:, npa:2*npa].mean(1)) / (
    np.linalg.norm(w_d1_base[:, :npa].mean(1)) * np.linalg.norm(w_d1_base[:, npa:2*npa].mean(1)) + 1e-10)

print(f"\n  Baseline (WITH override) after 50 steps:")
print(f"    w_d1 action 0 mean: {a0_base:.4f}")
print(f"    w_d1 action 1 mean: {a1_base:.4f}")
print(f"    cosine(w_d1[a0], w_d1[a1]): {cos_base:.4f}")

env.close()
env2.close()
env3.close()
print("\n=== GATED STDP SIMULATION COMPLETE ===")
