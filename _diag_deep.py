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

# Critic g_L now derived from C_m/tau_m (biophysical consistency fix)
g_L_critic = ncfg_c.g_L
# Actor g_L also derived from C_m/tau_m_msn_up in NeuronConfig
g_L_actor_ncfg = ncfg_a.g_L
g_L_actor_up = ncfg_a.C_m / actor.config.tau_m_msn_up
g_L_actor_down = ncfg_a.C_m / actor.config.tau_m_msn_down
gap = abs(ncfg_a.v_thresh - ncfg_a.v_rest)
delta_t = ncfg_a.delta_t

i_rheo_critic = g_L_critic * (gap - delta_t)
i_rheo_actor_up = g_L_actor_up * (gap - delta_t)
i_rheo_actor_down = g_L_actor_down * (gap - delta_t)

print(f"\nCritic NeuronConfig: g_L={g_L_critic:.2f} nS (=C_m/tau_m={ncfg_c.C_m}/{ncfg_c.tau_m}), "
      f"C_m={ncfg_c.C_m} pF, tau_m={ncfg_c.tau_m} ms")
print(f"Actor  NeuronConfig: g_L={g_L_actor_ncfg:.2f} nS (=C_m/tau_m={ncfg_a.C_m}/{ncfg_a.tau_m}), "
      f"C_m={ncfg_a.C_m} pF, tau_m={ncfg_a.tau_m} ms")
print(f"  Actor Up-state effective g_L = C_m/tau_up = {g_L_actor_up:.2f} nS  (matches ncfg)")
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

# Monkey-patch forward() to accumulate spike counts across ALL substeps,
# not just the last-substep snapshot.
import types

_orig_critic_fwd = critic.forward.__func__
_orig_actor_fwd = actor.forward.__func__
_cum_critic = [0]
_cum_d1 = [0]
_cum_d2 = [0]
_v_max_c = [-999.0]
_v_max_a = [-999.0]

def _patched_critic_fwd(self, state_spikes):
    result = _orig_critic_fwd(self, state_spikes)
    _cum_critic[0] += int(np.sum(self.spikes_hidden))
    _v_max_c[0] = max(_v_max_c[0], float(np.max(self.v_hidden)))
    return result

def _patched_actor_fwd(self, state_spikes):
    result = _orig_actor_fwd(self, state_spikes)
    _cum_d1[0] += int(np.sum(self.spikes_d1))
    _cum_d2[0] += int(np.sum(self.spikes_d2))
    _v_max_a[0] = max(_v_max_a[0], float(np.max(self.v_d1)))
    return result

critic.forward = types.MethodType(_patched_critic_fwd, critic)
actor.forward = types.MethodType(_patched_actor_fwd, actor)

w_d1_before = actor.w_d1.copy()
w_h_before = critic.w_h.copy()

state = env.reset(seed=42)
agent.reset()
total_reward = 0.0
max_e_h = 0.0
max_e_d1 = 0.0

n_sub = agent._n_substeps
print(f"\nn_substeps = {n_sub} (tau_max / dt, unclamped)")

for step in range(200):
    action = agent.act(state)
    ns, r, done, info = env.step(action)
    agent.observe(state, action, r, ns, done, info)

    max_e_h = max(max_e_h, float(np.max(np.abs(critic.e_h))))
    max_e_d1 = max(max_e_d1, float(np.max(np.abs(actor.e_d1))))

    total_reward += r
    state = ns
    if done:
        break

w_d1_after = actor.w_d1.copy()
w_h_after = critic.w_h.copy()

total_substeps = (step + 1) * 2 * n_sub  # act + observe
print(f"\nEpisode: {step+1} steps, reward={total_reward:.0f}")
print(f"Total substep calls: {total_substeps} ({step+1} steps × 2 × {n_sub} substeps)")
print(f"\nCumulative spike counts (across ALL substeps):")
print(f"  Critic hidden: {_cum_critic[0]}  "
      f"(rate={_cum_critic[0]/(total_substeps*critic.config.hidden_size)*100:.2f}% per neuron per substep)")
print(f"  Actor D1:      {_cum_d1[0]}  "
      f"(rate={_cum_d1[0]/(total_substeps*actor.action_dim)*100:.2f}%)")
print(f"  Actor D2:      {_cum_d2[0]}  "
      f"(rate={_cum_d2[0]/(total_substeps*actor.action_dim)*100:.2f}%)")
print(f"\nMax membrane voltage reached (across all substeps):")
print(f"  Critic: {_v_max_c[0]:.2f} mV  (threshold={ncfg_c.v_thresh:.1f}, cutoff={ncfg_c.v_spike_cutoff:.1f})")
print(f"  Actor:  {_v_max_a[0]:.2f} mV  (threshold={ncfg_a.v_thresh:.1f}, cutoff={ncfg_a.v_spike_cutoff:.1f})")
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
critic_noise_pA = ncfg_c.g_L * noise_std_mV
actor_noise_up_pA = g_L_actor_up * noise_std_mV
print(f"\nmembrane_noise_std = {noise_std_mV} (config value)")
print(f"Code multiplies by g_L before injecting as current (noise_std_pA = g_L × noise_std):")
print(f"  Critic noise std: {ncfg_c.g_L:.2f} nS × {noise_std_mV} = {critic_noise_pA:.1f} pA")
print(f"  Actor noise std (Up): {g_L_actor_up:.2f} nS × {noise_std_mV} = {actor_noise_up_pA:.1f} pA")
print(f"  Critic noise / rheobase: {critic_noise_pA / i_rheo_critic * 100:.1f}%")
print(f"  Actor noise / rheobase (Up): {actor_noise_up_pA / i_rheo_actor_up * 100:.1f}%")

env.close()
print("\n" + "=" * 72)
print("  DIAGNOSTIC COMPLETE")
print("=" * 72)
"""Deep diagnostic: trace per-sub-step internals to find why actor neurons die."""
