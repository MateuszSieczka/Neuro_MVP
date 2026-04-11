"""
Deep diagnostic: prove whether BG neurons spike and why/why not.

Computes AdEx rheobase for critic and actor, compares to actual I_syn,
tracks membrane voltages, spike counts, eligibility traces, and weight changes.
"""
import numpy as np
np.random.seed(42)

from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena.task_config import get as get_task
from core.spike_encoder import PoissonEncoder, GaussianPopulationEncoder

# ── Setup ─────────────────────────────────────────────────────────────
task = get_task("CartPole-v1")
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds,
             reward_scale=task.reward_scale)
state = env.reset(seed=42)
agent = make_agent(task, env)

critic = agent.critic
actor = agent.actor
ncfg_c = critic._ncfg
ncfg_a = actor._ncfg

print("=" * 72)
print("  SECTION 1: AdEx Rheobase Analysis")
print("=" * 72)

# Critic g_L comes from NeuronConfig default (30 nS) since
# SNNDeepCritic creates NeuronConfig without specifying g_L
g_L_critic = ncfg_c.g_L
g_L_actor_up = ncfg_a.C_m / actor.config.tau_m_msn_up
g_L_actor_down = ncfg_a.C_m / actor.config.tau_m_msn_down
gap = abs(ncfg_a.v_thresh - ncfg_a.v_rest)
delta_t = ncfg_a.delta_t

i_rheo_critic = g_L_critic * (gap - delta_t)
i_rheo_actor_up = g_L_actor_up * (gap - delta_t)
i_rheo_actor_down = g_L_actor_down * (gap - delta_t)

print(f"\nCritic NeuronConfig: g_L={g_L_critic} nS, C_m={ncfg_c.C_m} pF, "
      f"tau_m={ncfg_c.tau_m} ms")
print(f"Actor  NeuronConfig: g_L={ncfg_a.g_L} nS, C_m={ncfg_a.C_m} pF, "
      f"tau_m={ncfg_a.tau_m} ms")
print(f"  Actor Up-state effective g_L = C_m/tau_up = {g_L_actor_up:.2f} nS")
print(f"  Actor Down-state effective g_L = C_m/tau_down = {g_L_actor_down:.2f} nS")
print(f"\nGap = |v_thresh - v_rest| = {gap:.1f} mV")
print(f"delta_t = {delta_t:.1f} mV")
print(f"\nI_rheobase (pA):  (= g_L × (gap − delta_t))")
print(f"  Critic:       {i_rheo_critic:.1f} pA")
print(f"  Actor Up:     {i_rheo_actor_up:.1f} pA")
print(f"  Actor Down:   {i_rheo_actor_down:.1f} pA")

print(f"\nActual input_gain:")
print(f"  Critic: {critic._input_gain:.6f}")
print(f"  Actor:  {actor._input_gain:.6f}")

# ── What actual I_syn looks like ──────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 2: Actual Synaptic Current vs Rheobase")
print("=" * 72)

poisson = PoissonEncoder()
pop_encoder = GaussianPopulationEncoder(n_dims=4, n_neurons_per_dim=15,
                                         value_min=-1.0, value_max=1.0)
state_f32 = state.astype(np.float32)
pop_rates = pop_encoder.encode(state_f32)
encoded = poisson.encode(pop_rates)
print(f"\nRaw state: {state_f32}")
print(f"Population encoded → Poisson ({len(encoded)}-dim):")
print(f"  Pop rates: mean={pop_rates.mean():.3f}, active(>0.1)={int(np.sum(pop_rates > 0.1))}")
print(f"  Spikes: {int(np.sum(encoded > 0.5))} active / {len(encoded)} total")

# Critic I_syn
raw_current_c = encoded @ critic.w_h
scaled_current_c = raw_current_c * critic._input_gain
print(f"\nCritic (fan_in={critic._state_size}, hidden={critic.config.hidden_size}):")
print(f"  Raw current (state @ w_h):     mean={raw_current_c.mean():.4f}, "
      f"max={raw_current_c.max():.4f}")
print(f"  Scaled (× gain={critic._input_gain:.6f}): mean={scaled_current_c.mean():.4f}, "
      f"max={scaled_current_c.max():.4f}")
print(f"  Ratio to rheobase: {scaled_current_c.max():.4f} / {i_rheo_critic:.1f} "
      f"= {scaled_current_c.max() / i_rheo_critic:.6f} ({scaled_current_c.max() / i_rheo_critic * 100:.3f}%)")

# Actor I_syn
raw_current_a = encoded @ actor.w_d1
scaled_current_a = raw_current_a * actor._input_gain
print(f"\nActor (fan_in={actor._state_size}, action_dim={actor.action_dim}):")
print(f"  Raw current (state @ w_d1):    mean={raw_current_a.mean():.4f}, "
      f"max={raw_current_a.max():.4f}")
print(f"  Scaled (× gain={actor._input_gain:.6f}): mean={scaled_current_a.mean():.4f}, "
      f"max={scaled_current_a.max():.4f}")
print(f"  Ratio to rheobase (Up): {scaled_current_a.max():.4f} / {i_rheo_actor_up:.1f} "
      f"= {scaled_current_a.max() / i_rheo_actor_up:.6f} ({scaled_current_a.max() / i_rheo_actor_up * 100:.3f}%)")

# ── What *should* the gain be (rheobase formula) ─────────────────────
print("\n" + "=" * 72)
print("  SECTION 3: What gain SHOULD be (rheobase formula)")
print("=" * 72)

for name, i_rheo, fan_in in [
    ("Critic", i_rheo_critic, critic._state_size),
    ("Actor",  i_rheo_actor_up, actor._state_size),
]:
    expected_active = max(1.0, fan_in * 0.05)
    correct_gain = i_rheo / expected_active
    print(f"\n{name}: I_rheo={i_rheo:.1f} pA, expected_active={expected_active:.1f}")
    print(f"  Correct gain = {correct_gain:.2f} pA/spike")
    print(f"  Current gain = {critic._input_gain if name == 'Critic' else actor._input_gain:.6f}")
    print(f"  Ratio (correct/current) = {correct_gain / (critic._input_gain if name == 'Critic' else actor._input_gain):.1f}×")

# ── Encoding comparison ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 4: Population Encoding Summary")
print("=" * 72)

print(f"\nGaussianPopulationEncoder(15/dim) → {len(pop_rates)}-dim (used by agent):")
print(f"  Rates: mean={pop_rates.mean():.3f}, active(>0.1)={int(np.sum(pop_rates > 0.1))}")
print(f"  After Poisson: {int(np.sum(encoded > 0.5))} spikes / {len(encoded)} total")
print(f"  Information density: {int(np.sum(encoded > 0.5)) / len(encoded) * 100:.1f}%")

# ── Section 5: Run episode, track spikes ─────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 5: Episode Run — Spike / Eligibility / Weight Tracking")
print("=" * 72)

w_d1_before = actor.w_d1.copy()
w_h_before = critic.w_h.copy()

state = env.reset(seed=42)
agent.reset()
total_reward = 0.0
total_critic_spikes = 0
total_actor_d1_spikes = 0
total_actor_d2_spikes = 0
max_e_h = 0.0
max_e_d1 = 0.0
v_max_critic = -999.0
v_max_actor = -999.0

for step in range(200):
    action = agent.act(state)
    ns, r, done, info = env.step(action)
    agent.observe(state, action, r, ns, done, info)

    total_critic_spikes += int(np.sum(critic.spikes_hidden))
    total_actor_d1_spikes += int(np.sum(actor.spikes_d1))
    total_actor_d2_spikes += int(np.sum(actor.spikes_d2))
    max_e_h = max(max_e_h, float(np.max(np.abs(critic.e_h))))
    max_e_d1 = max(max_e_d1, float(np.max(np.abs(actor.e_d1))))
    v_max_critic = max(v_max_critic, float(np.max(critic.v_hidden)))
    v_max_actor = max(v_max_actor, float(np.max(actor.v_d1)))

    total_reward += r
    state = ns
    if done:
        break

w_d1_after = actor.w_d1.copy()
w_h_after = critic.w_h.copy()

print(f"\nEpisode: {step+1} steps, reward={total_reward:.0f}")
print(f"\nSpike counts over episode:")
print(f"  Critic hidden: {total_critic_spikes} (of {critic.config.hidden_size} neurons × {step+1} steps)")
print(f"  Actor D1:      {total_actor_d1_spikes}")
print(f"  Actor D2:      {total_actor_d2_spikes}")
print(f"\nMax membrane voltage reached:")
print(f"  Critic: {v_max_critic:.2f} mV  (threshold={ncfg_c.v_thresh:.1f}, cutoff={ncfg_c.v_spike_cutoff:.1f})")
print(f"  Actor:  {v_max_actor:.2f} mV  (threshold={ncfg_a.v_thresh:.1f}, cutoff={ncfg_a.v_spike_cutoff:.1f})")
print(f"\nMax eligibility trace magnitude:")
print(f"  Critic e_h: {max_e_h:.8f}")
print(f"  Actor e_d1: {max_e_d1:.8f}")
print(f"\nWeight change (Frobenius norm):")
print(f"  Critic w_h: {np.linalg.norm(w_h_after - w_h_before):.8f}")
print(f"  Actor w_d1: {np.linalg.norm(w_d1_after - w_d1_before):.8f}")
print(f"\nV_trace={critic.v_trace:.6f}, last_value={critic.last_value:.6f}")
print(f"TD error (last step): {agent._last_td_error:.6f}")

# ── Noise analysis ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 6: Noise Unit Analysis")
print("=" * 72)
noise_std_mV = critic.config.membrane_noise_std
print(f"\nmembrane_noise_std = {noise_std_mV} (documented as mV)")
print(f"But noise is added to I_syn BEFORE AdEx division by C_m={ncfg_c.C_m} pF")
print(f"Effective voltage noise per step ≈ noise_std / C_m × dt = "
      f"{noise_std_mV / ncfg_c.C_m:.6f} mV  (near zero)")
print(f"For rheobase-scale noise, need std ≈ g_L × 2.0 = {ncfg_c.g_L * 2.0:.0f} pA")

env.close()
print("\n" + "=" * 72)
print("  DIAGNOSTIC COMPLETE")
print("=" * 72)
"""Deep diagnostic: trace per-sub-step internals to find why actor neurons die."""
