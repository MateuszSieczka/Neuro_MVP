"""
NORMALIZATION & UPDATE MAGNITUDE DIAGNOSTIC
Checks whether the net_evidence normalization destroys the signal,
and traces the exact actor update arithmetic.
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
print("NORMALIZATION & UPDATE MAGNITUDE DIAGNOSTIC")
print("=" * 80)

# ── 1. Net evidence breakdown ──────────────────────────────────────
print("\n### Net Evidence Breakdown (10 independent act() calls) ###")
for trial in range(10):
    state = env.reset()
    agent.reset()
    agent.actor.reset_spike_counts()
    agent.critic.reset_spike_counts(agent._n_substeps)
    
    pop_rates = agent._pop_encoder.encode(state.astype(np.float32))
    for _ in range(agent._n_substeps):
        enc = agent._poisson.encode(pop_rates)
        sensory = agent._build_sensory_inputs(enc, state.astype(np.float32))
        agent.network.step(sensory_inputs=sensory, neuromodulator=agent.neuromod, attention=None)
    
    npa = agent.actor.n_per_action
    n_fwd = agent.actor._n_forward
    
    # Raw v_accum
    v_d1 = agent.actor._v_accum_d1
    v_d2 = agent.actor._v_accum_d2
    motor_d1_raw = v_d1[:agent.actor._total_motor].reshape(2, npa).sum(axis=1)
    motor_d2_raw = v_d2[:agent.actor._total_motor].reshape(2, npa).sum(axis=1)
    raw_diff = motor_d1_raw - motor_d2_raw
    
    _norm = npa * np.sqrt(float(n_fwd))
    net_evidence = raw_diff / _norm
    
    # Temperature-scaled noise
    ne = agent.neuromod.competition_sharpness
    temperature = 1.0 + 4.0 * (ne - 0.5) ** 2
    noise_sample = np.random.normal(0, temperature, 2)
    noisy = net_evidence + noise_sample
    action = int(np.argmax(noisy))
    
    print(f"  Trial {trial}: d1_raw={motor_d1_raw}, d2_raw={motor_d2_raw}")
    print(f"    raw_diff={raw_diff}, norm={_norm:.1f}, net_ev={net_evidence}")
    print(f"    noise={noise_sample}, noisy={noisy}, action={action}")
    print(f"    spike_d1={agent.actor._spike_count_d1.sum():.0f}, spike_d2={agent.actor._spike_count_d2.sum():.0f}")

# ── 2. Compare with/without normalization ──────────────────────────
print("\n### SNR Analysis ###")
signals = []
noises = []
for _ in range(100):
    state = env.reset()
    agent.reset()
    agent.actor.reset_spike_counts()
    pop_rates = agent._pop_encoder.encode(state.astype(np.float32))
    for _ in range(agent._n_substeps):
        enc = agent._poisson.encode(pop_rates)
        sensory = agent._build_sensory_inputs(enc, state.astype(np.float32))
        agent.network.step(sensory_inputs=sensory, neuromodulator=agent.neuromod, attention=None)
    
    npa = agent.actor.n_per_action
    n_fwd = agent.actor._n_forward
    v_d1 = agent.actor._v_accum_d1[:agent.actor._total_motor].reshape(2, npa).sum(axis=1)
    v_d2 = agent.actor._v_accum_d2[:agent.actor._total_motor].reshape(2, npa).sum(axis=1)
    raw_diff = v_d1 - v_d2
    _norm = npa * np.sqrt(float(n_fwd))
    net_ev = raw_diff / _norm
    signals.append(net_ev)

signals = np.array(signals)
mean_signal = np.mean(signals, axis=0)
std_signal = np.std(signals, axis=0)
print(f"  net_evidence: mean={mean_signal}, std={std_signal}")
print(f"  |mean|/std (SNR) = {np.abs(mean_signal)/std_signal}")
print(f"  Temperature noise std ≈ 1.0")
print(f"  Effective SNR (signal vs noise+signal_noise) = {np.abs(mean_signal)/np.sqrt(std_signal**2 + 1)}")

# ── 3. Exact update arithmetic trace ──────────────────────────────
print("\n### Exact Update Arithmetic (1 step) ###")
state = env.reset()
agent.reset()

action = agent.act(state)
ns, r, done, info = env.step(action)

# Get eligibility
e_d1 = agent.actor.e_d1.copy()
e_d2 = agent.actor.e_d2.copy()

print(f"  action={action}, reward={r}")
print(f"  e_d1 stats: mean={np.mean(e_d1):.8f}, |mean|={np.mean(np.abs(e_d1)):.8f}, nnz={np.count_nonzero(e_d1)}/{e_d1.size}")
print(f"  e_d2 stats: mean={np.mean(e_d2):.8f}, |mean|={np.mean(np.abs(e_d2)):.8f}")

# Predict what update should be
# TD will be approximately: r + gamma * V(s') - V(s)
prev_v = agent.critic.last_value
# save state
w_d1_before = agent.actor.w_d1.copy()
agent.observe(state, action, r, ns, done, info)
td = agent._last_td_error
w_d1_after = agent.actor.w_d1.copy()

# Predicted update
lr = agent._bg_config.actor_lr  # 0.01
gate = agent.neuromod.consolidation_gate
acfg = agent._agent_cfg
plasticity_scale = acfg.consolidation_floor + (1.0 - acfg.consolidation_floor) / (
    1.0 + np.exp(acfg.consolidation_steepness * (gate - acfg.consolidation_midpoint)))
td_for_update = td * plasticity_scale

predicted_dw_d1 = lr * td_for_update * e_d1
np.clip(predicted_dw_d1, -1, 1, out=predicted_dw_d1)
actual_dw_d1 = w_d1_after - w_d1_before

print(f"\n  TD={td:.6f}, plasticity_scale={plasticity_scale:.6f}, td_for_update={td_for_update:.6f}")
print(f"  lr={lr}, lr*td_update={lr*td_for_update:.8f}")
print(f"  predicted |dw_d1| mean: {np.mean(np.abs(predicted_dw_d1)):.10f}")
print(f"  actual |dw_d1| mean:    {np.mean(np.abs(actual_dw_d1)):.10f}")
print(f"  Floor effect: max(w_d1, 0.01) clips {np.sum(w_d1_after <= 0.011)} entries near floor")

# ── 4. STDP vs PG magnitude comparison ──────────────────────────────
print("\n### STDP vs REINFORCE Override Magnitude ###")
state = env.reset()
agent.reset()

agent.actor.reset_spike_counts()
agent.critic.reset_spike_counts(agent._n_substeps)
pop_rates = agent._pop_encoder.encode(state.astype(np.float32))
for _ in range(agent._n_substeps):
    enc = agent._poisson.encode(pop_rates)
    sensory = agent._build_sensory_inputs(enc, state.astype(np.float32))
    agent.network.step(sensory_inputs=sensory, neuromodulator=agent.neuromod, attention=None)

e_stdp = agent.actor.e_d1.copy()
print(f"  STDP eligibility: |mean|={np.mean(np.abs(e_stdp)):.8f}, Frobenius={np.linalg.norm(e_stdp):.6f}")

agent._set_actor_policy_gradient(pop_rates)
e_pg = agent.actor.e_d1.copy()
print(f"  PG eligibility:   |mean|={np.mean(np.abs(e_pg)):.8f}, Frobenius={np.linalg.norm(e_pg):.6f}")
print(f"  Ratio STDP/PG:    {np.linalg.norm(e_stdp)/np.linalg.norm(e_pg):.1f}x magnitude difference")

# Predicted update with each
td_typical = 1.0
dw_stdp = lr * td_typical * np.mean(np.abs(e_stdp))
dw_pg = lr * td_typical * np.mean(np.abs(e_pg))
print(f"\n  With td=1, lr={lr}:")
print(f"    STDP: |dw| per element ≈ {dw_stdp:.8f}")
print(f"    PG:   |dw| per element ≈ {dw_pg:.8f}")
print(f"    Per episode (~20 steps): STDP Δw ≈ {dw_stdp * 20:.6f}, PG Δw ≈ {dw_pg * 20:.6f}")
print(f"    Per 200 episodes: STDP Δw ≈ {dw_stdp * 20 * 200:.4f}, PG Δw ≈ {dw_pg * 20 * 200:.4f}")

# ── 5. Homeostatic scaling contribution ─────────────────────────────
print("\n### Homeostatic Scaling Contribution ###")
print(f"  homeo_interval={agent._bg_config.homeo_interval}")
print(f"  homeo_max_change={agent._bg_config.homeo_max_change}")
print(f"  homeo_target_rate={agent._bg_config.homeo_target_rate}")
print(f"  actor d1 homeo_rate_d1 mean: {np.mean(agent.actor._homeo_rate_d1):.6f}")
print(f"  actor d2 homeo_rate_d2 mean: {np.mean(agent.actor._homeo_rate_d2):.6f}")
print(f"  critic homeo_rate mean: {np.mean(agent.critic._homeo_rate):.6f}")
print(f"  If actual_rate ≈ 0 and target=0.01: error≈1.0, scale=1+0.02*1=1.02")
print(f"  Over 200 eps × 20 steps = 4000 steps, {4000//200} homeostatic events")
print(f"  Cumulative scaling: 1.02^{4000//200} = {1.02**(4000//200):.4f}")

env.close()
print("\n=== DIAGNOSTIC COMPLETE ===")
