"""
DEEP PIPELINE DIAGNOSTIC: Traces every stage of the learning pipeline
for a single episode to understand why learning fails.
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
print("DEEP PIPELINE DIAGNOSTIC")
print("=" * 80)

# ── 1. Architecture summary ──────────────────────────────────────────────
print(f"\n### Architecture ###")
print(f"  state_size={agent.state_size}, n_actions={agent.n_actions}")
print(f"  use_world_model={agent._use_wm}, use_working_memory={agent._use_working_memory}")
print(f"  pop_encoder: {agent._pop_encoder.n_dims}D -> {agent._pop_encoder.output_size}D")
print(f"  n_substeps_act={agent._n_substeps}, n_substeps_critic={agent._n_substeps_critic}")
print(f"  critic: input={agent.critic._state_size}, hidden={agent.critic.config.hidden_size}")
print(f"  actor: input={agent.actor._state_size}, action_dim={agent.actor.action_dim}")
print(f"  actor: motor_dim={agent.actor.motor_dim}, n_per_action={agent.actor.n_per_action}")
print(f"  actor: total_motor={agent.actor._total_motor}")
cfg = agent._bg_config
print(f"\n### BG Config ###")
print(f"  gamma={cfg.gamma}, critic_lr={cfg.critic_lr}, actor_lr={cfg.actor_lr}")
print(f"  tau_e_actor={cfg.tau_e_actor}, tau_e_critic={cfg.tau_e_critic}")
print(f"  tau_m_msn_up={cfg.tau_m_msn_up}, tau_m_msn_down={cfg.tau_m_msn_down}")
print(f"  tau_m_critic={cfg.tau_m_critic}")
print(f"  membrane_noise_std={cfg.membrane_noise_std}")
print(f"  hidden_size={cfg.hidden_size}, neurons_per_action={cfg.neurons_per_action}")
print(f"  d1_bias={cfg.d1_bias}, d2_bias={cfg.d2_bias}")
print(f"  homeo_target={cfg.homeo_target_rate}, homeo_interval={cfg.homeo_interval}")
print(f"  ltd_ratio={cfg.ltd_ratio}, d2_ltd_protection={cfg.d2_ltd_protection}")
print(f"  w_clip={cfg.w_clip}, w_clip_critic={cfg.w_clip_critic}")
print(f"  readout_decay={cfg.readout_decay}")

# ── 2. Input gain analysis ────────────────────────────────────────────
print(f"\n### Input Gains ###")
print(f"  critic._input_gain = {agent.critic._input_gain:.4f}")
print(f"  actor._input_gain = {agent.actor._input_gain:.4f}")
ncfg_c = agent.critic._ncfg
ncfg_a = agent.actor._ncfg
print(f"  critic ncfg: g_L={ncfg_c.g_L:.4f}, C_m={ncfg_c.C_m}, delta_t={ncfg_c.delta_t}")
print(f"  critic ncfg: a={ncfg_c.a:.4f}, b={ncfg_c.b:.4f}")
print(f"  actor  ncfg: g_L={ncfg_a.g_L:.4f}, C_m={ncfg_a.C_m}, delta_t={ncfg_a.delta_t}")
print(f"  actor  ncfg: a={ncfg_a.a:.4f}, b={ncfg_a.b:.4f}")
gap_c = abs(ncfg_c.v_thresh - ncfg_c.v_rest)
i_rheo_c = ncfg_c.g_L * (gap_c - ncfg_c.delta_t)
gap_a = abs(ncfg_a.v_thresh - ncfg_a.v_rest)
i_rheo_a = ncfg_a.g_L * (gap_a - ncfg_a.delta_t)
print(f"  critic I_rheo = {i_rheo_c:.2f} pA")
print(f"  actor  I_rheo = {i_rheo_a:.2f} pA")

# ── 3. Single step trace ──────────────────────────────────────────────
print(f"\n### Single Step Trace ###")
state = env.reset()
agent.reset()
print(f"  raw state: {state}")

# Encode
pop_rates = agent._pop_encoder.encode(state.astype(np.float32))
print(f"  pop_rates: shape={pop_rates.shape}, min={pop_rates.min():.4f}, max={pop_rates.max():.4f}, mean={pop_rates.mean():.4f}")
print(f"  active neurons (rate>0.5): {np.sum(pop_rates > 0.5)}/{len(pop_rates)}")

# Poisson encode
encoded = agent._poisson.encode(pop_rates)
print(f"  encoded (Poisson): {np.sum(encoded)} spikes out of {len(encoded)}")

# Critic forward - manual check of current magnitude
print(f"\n### Critic Synaptic Current ###")
w_h = agent.critic.w_h
current_test = (encoded @ w_h) * agent.critic._input_gain
print(f"  w_h: shape={w_h.shape}, mean={np.mean(w_h):.4f}, std={np.std(w_h):.4f}")
print(f"  raw current (inp @ w_h): mean={np.mean(encoded @ w_h):.4f}, max={np.max(encoded @ w_h):.4f}")
print(f"  scaled current: mean={np.mean(current_test):.2f} pA, max={np.max(current_test):.2f} pA")
print(f"  current / I_rheo: mean={np.mean(np.abs(current_test))/i_rheo_c:.3f}, max={np.max(np.abs(current_test))/i_rheo_c:.3f}")

# Actor forward - same check
print(f"\n### Actor Synaptic Current ###")
current_d1 = (encoded @ agent.actor.w_d1) * agent.actor._input_gain
current_d2 = (encoded @ agent.actor.w_d2) * agent.actor._input_gain
print(f"  w_d1: mean={np.mean(agent.actor.w_d1):.4f}, std={np.std(agent.actor.w_d1):.4f}")
print(f"  scaled current_d1: mean={np.mean(current_d1):.2f} pA, max={np.max(current_d1):.2f} pA")
print(f"  scaled current_d2: mean={np.mean(current_d2):.2f} pA, max={np.max(current_d2):.2f} pA")
print(f"  current_d1 / I_rheo: mean={np.mean(np.abs(current_d1))/i_rheo_a:.3f}")

# ── 4. Run act() and inspect everything ──────────────────────────────
print(f"\n### act() Detailed Trace ###")
action = agent.act(state)
print(f"  action={action}")
print(f"  critic V(s) after act: {agent.critic.last_value:.6f}")
print(f"  critic spike_count sum: {np.sum(agent.critic._spike_count):.0f}")
print(f"  critic v_accum mean: {np.mean(agent.critic._v_accum):.4f}")
print(f"  critic v_hidden: min={np.min(agent.critic.v_hidden):.2f}, max={np.max(agent.critic.v_hidden):.2f}, mean={np.mean(agent.critic.v_hidden):.2f}")
print(f"  critic spikes this step: {np.sum(agent.critic.spikes_hidden)}")
print(f"  critic w_adapt: mean={np.mean(agent.critic.w_adapt_hidden):.4f}")

print(f"\n  actor v_d1: min={np.min(agent.actor.v_d1):.2f}, max={np.max(agent.actor.v_d1):.2f}")
print(f"  actor v_d2: min={np.min(agent.actor.v_d2):.2f}, max={np.max(agent.actor.v_d2):.2f}")
print(f"  actor spikes_d1 total: {np.sum(agent.actor._spike_count_d1):.0f}")
print(f"  actor spikes_d2 total: {np.sum(agent.actor._spike_count_d2):.0f}")
print(f"  actor v_accum_d1 mean: {np.mean(agent.actor._v_accum_d1):.4f}")
print(f"  actor v_accum_d2 mean: {np.mean(agent.actor._v_accum_d2):.4f}")
print(f"  actor w_adapt_d1: mean={np.mean(agent.actor.w_adapt_d1):.4f}")
print(f"  actor _last_probs: {agent.actor._last_probs}")

# Check eligibility traces
print(f"\n### Eligibility After act() ###")
print(f"  actor e_d1: min={np.min(agent.actor.e_d1):.6f}, max={np.max(agent.actor.e_d1):.6f}, |mean|={np.mean(np.abs(agent.actor.e_d1)):.6f}")
print(f"  actor e_d2: min={np.min(agent.actor.e_d2):.6f}, max={np.max(agent.actor.e_d2):.6f}, |mean|={np.mean(np.abs(agent.actor.e_d2)):.6f}")
print(f"  actor e_d1 neg frac: {np.mean(agent.actor.e_d1 < -1e-8):.4f}")
print(f"  actor e_d2 neg frac: {np.mean(agent.actor.e_d2 < -1e-8):.4f}")
print(f"  critic e_h: min={np.min(agent.critic.e_h):.6f}, max={np.max(agent.critic.e_h):.6f}, |mean|={np.mean(np.abs(agent.critic.e_h)):.6f}")
print(f"  critic e_v: min={np.min(agent.critic.e_v):.6f}, max={np.max(agent.critic.e_v):.6f}")
print(f"  critic e_bv: {agent.critic.e_bv:.6f}")

# ── 5. observe() trace ───────────────────────────────────────────────
print(f"\n### observe() Trace ###")
ns, r, done, info = env.step(action)
print(f"  reward={r}, done={done}")

w_d1_before = agent.actor.w_d1.copy()
w_d2_before = agent.actor.w_d2.copy()
w_h_before = agent.critic.w_h.copy()
w_v_before = agent.critic.w_v.copy()
bv_before = agent.critic.b_v

agent.observe(state, action, r, ns, done, info)

td = agent._last_td_error
print(f"  TD error: {td:.6f}")
print(f"  prev_v (V(s)): should have been ~{agent.critic.last_value:.6f}")

dw_d1 = agent.actor.w_d1 - w_d1_before
dw_d2 = agent.actor.w_d2 - w_d2_before
dw_h = agent.critic.w_h - w_h_before
dw_v = agent.critic.w_v - w_v_before
db_v = agent.critic.b_v - bv_before

print(f"\n### Weight Updates ###")
print(f"  |dw_d1|: mean={np.mean(np.abs(dw_d1)):.8f}, max={np.max(np.abs(dw_d1)):.8f}")
print(f"  |dw_d2|: mean={np.mean(np.abs(dw_d2)):.8f}, max={np.max(np.abs(dw_d2)):.8f}")
print(f"  |dw_h|:  mean={np.mean(np.abs(dw_h)):.8f}, max={np.max(np.abs(dw_h)):.8f}")
print(f"  |dw_v|:  mean={np.mean(np.abs(dw_v)):.8f}, max={np.max(np.abs(dw_v)):.8f}")
print(f"  |db_v|:  {abs(db_v):.8f}")
print(f"  dw_d1 range: [{np.min(dw_d1):.8f}, {np.max(dw_d1):.8f}]")
print(f"  dw_d2 range: [{np.min(dw_d2):.8f}, {np.max(dw_d2):.8f}]")

# ── 6. Run 5 episodes, track per-step V(s), TD, actions ──────────────
print(f"\n\n### Multi-Episode Tracking (5 episodes) ###")
state = ns if not done else env.reset()
if done:
    agent.reset()

for ep in range(5):
    state = env.reset()
    agent.reset()
    vs = []
    tds = []
    actions = []
    rewards = []
    for step in range(500):
        a = agent.act(state)
        ns, r, done, info = env.step(a)
        v_s = agent.critic.last_value
        agent.observe(state, a, r, ns, done, info)
        vs.append(v_s)
        tds.append(agent._last_td_error)
        actions.append(a)
        rewards.append(r)
        state = ns
        if done: break
    
    act_balance = np.mean(np.array(actions) == 0)
    print(f"  Ep {ep}: steps={len(vs)}, V mean={np.mean(vs):.4f} std={np.std(vs):.4f}, "
          f"TD mean={np.mean(tds):.4f} std={np.std(tds):.4f}, act0_frac={act_balance:.2f}")

# ── 7. V(s) discrimination test ──────────────────────────────────────
print(f"\n### V(s) Discrimination Test ###")
test_states = {
    'balanced': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    'tilt_right': np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float32),
    'tilt_left': np.array([0.0, 0.0, -0.1, 0.0], dtype=np.float32),
    'falling': np.array([0.0, 0.0, 0.5, 2.0], dtype=np.float32),
}

for name, s in test_states.items():
    # Normalize like GymEnv would
    obs_low = task.obs_bounds[0]
    obs_high = task.obs_bounds[1]
    s_norm = 2.0 * (s - obs_low) / (obs_high - obs_low + 1e-8) - 1.0
    
    v_samples = []
    for _ in range(10):
        agent.critic.reset_spike_counts(agent._n_substeps)
        pop = agent._pop_encoder.encode(s_norm)
        for _ in range(agent._n_substeps):
            enc = agent._poisson.encode(pop)
            agent.critic.forward(enc)
        v_samples.append(agent.critic.last_value)
    
    print(f"  {name:15s}: V(s)={np.mean(v_samples):.4f} ± {np.std(v_samples):.4f}")

# ── 8. InhibitoryPool diagnostics ────────────────────────────────────
print(f"\n### InhibitoryPool Status ###")
for pool_name, pool in [("critic", agent.critic.inh_pool),
                         ("actor_d1", agent.actor.inh_pool_d1),
                         ("actor_d2", agent.actor.inh_pool_d2)]:
    print(f"  {pool_name}: n_inh={pool.config.n_interneurons}, "
          f"v_inh mean={np.mean(pool.v_inh):.2f}, "
          f"spikes={np.sum(pool.spikes_inh)}, "
          f"gaba_a mean={np.mean(pool.i_gaba_a):.6f}, "
          f"gaba_b mean={np.mean(pool.i_gaba_b):.6f}, "
          f"input_gain={pool._input_gain:.2f}")

# ── 9. Neuromodulator state ──────────────────────────────────────────
nm = agent.neuromod
print(f"\n### Neuromodulator State ###")
print(f"  DA={nm.dopamine:.4f}, ACh={nm.acetylcholine:.4f}, NE={nm.noradrenaline:.4f}, 5HT={nm.serotonin:.4f}")
print(f"  tonic_da={nm.tonic_da:.6f}")
print(f"  learning_rate_mod={nm.learning_rate_modulation:.4f}")
print(f"  consolidation_gate={nm.consolidation_gate:.4f}")

env.close()
print("\n=== DIAGNOSTIC COMPLETE ===")
