"""
LEARNING DYNAMICS DIAGNOSTIC: Trace weight evolution, TD error patterns,
eligibility structure, and V(s) discrimination over 200 episodes.
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
print("LEARNING DYNAMICS DIAGNOSTIC")
print("=" * 80)

# KEY METRIC: Track V(s) for canonical states after each batch
obs_low = task.obs_bounds[0]
obs_high = task.obs_bounds[1]

def make_canonical_state(raw):
    return 2.0 * (raw - obs_low) / (obs_high - obs_low + 1e-8) - 1.0

canonical_states = {
    'balanced': make_canonical_state(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
    'falling': make_canonical_state(np.array([0.0, 0.0, 0.5, 2.0], dtype=np.float32)),
}

def eval_v(state_norm, n_trials=5):
    """Evaluate V(s) for a given normalized state."""
    vals = []
    for _ in range(n_trials):
        agent.critic.reset_spike_counts(agent._n_substeps)
        pop = agent._pop_encoder.encode(state_norm)
        for _ in range(agent._n_substeps):
            enc = agent._poisson.encode(pop)
            agent.critic.forward(enc)
        vals.append(agent.critic.last_value)
    return np.mean(vals), np.std(vals)

# ── PHASE 1: Track per-step details for first 3 episodes ──────────────
print("\n### Phase 1: Detailed first 3 episodes ###")
for ep in range(3):
    state = env.reset()
    agent.reset()
    step_data = []
    for step in range(500):
        a = agent.act(state)
        ns, r, done, info = env.step(a)
        
        prev_v = agent.critic.last_value
        e_d1_norm = np.linalg.norm(agent.actor.e_d1)
        e_d2_norm = np.linalg.norm(agent.actor.e_d2)
        e_h_neg = np.mean(agent.critic.e_h < -1e-8)
        
        w_d1_pre = agent.actor.w_d1.copy()
        w_d2_pre = agent.actor.w_d2.copy()
        
        agent.observe(state, a, r, ns, done, info)
        
        td = agent._last_td_error
        dw_d1 = np.mean(np.abs(agent.actor.w_d1 - w_d1_pre))
        dw_d2 = np.mean(np.abs(agent.actor.w_d2 - w_d2_pre))
        
        if step < 5 or step == len(step_data):
            print(f"  Ep{ep} Step{step}: a={a}, r={r}, V={prev_v:.4f}, TD={td:.4f}, "
                  f"|e_d1|={e_d1_norm:.6f}, |e_d2|={e_d2_norm:.6f}, "
                  f"|dw_d1|={dw_d1:.8f}, |dw_d2|={dw_d2:.8f}, "
                  f"e_h_neg%={e_h_neg:.3f}")
        
        state = ns
        if done: break

# ── PHASE 2: Track learning over 200 episodes ─────────────────────────
print("\n### Phase 2: Learning curves (200 eps) ###")
scores = []
ep_data = []

for ep in range(200):
    state = env.reset()
    agent.reset()
    total_r = 0
    ep_tds = []
    ep_vs = []
    actions = []
    
    for step in range(500):
        a = agent.act(state)
        ns, r, done, info = env.step(a)
        v_s = agent.critic.last_value
        agent.observe(state, a, r, ns, done, info)
        
        ep_tds.append(agent._last_td_error)
        ep_vs.append(v_s)
        actions.append(a)
        total_r += r
        state = ns
        if done: break
    
    scores.append(total_r)
    
    if (ep+1) % 25 == 0:
        v_bal, v_bal_std = eval_v(canonical_states['balanced'])
        v_fall, v_fall_std = eval_v(canonical_states['falling'])
        
        last25 = scores[-25:]
        w_d1 = agent.actor.w_d1
        w_d2 = agent.actor.w_d2
        w_h = agent.critic.w_h
        w_v = agent.critic.w_v
        
        # Per-action weight analysis
        for act_idx in range(2):
            s = act_idx * agent.actor.n_per_action
            e = s + agent.actor.n_per_action
            d1_cols = w_d1[:, s:e]
            d2_cols = w_d2[:, s:e]
        
        act_balance = np.mean(np.array(actions) == 0)
        
        print(f"  Ep {ep+1:3d}: mean25={np.mean(last25):.1f}, max={np.max(last25):.0f}, "
              f"V(bal)={v_bal:.3f}±{v_bal_std:.3f}, V(fall)={v_fall:.3f}±{v_fall_std:.3f}, "
              f"disc={v_bal-v_fall:.3f}")
        print(f"          b_v={agent.critic.b_v:.3f}, |w_v|={np.mean(np.abs(w_v)):.4f}, "
              f"|w_h|={np.mean(np.abs(w_h)):.4f}, "
              f"|w_d1|={np.mean(w_d1):.4f}, |w_d2|={np.mean(w_d2):.4f}, "
              f"act0%={act_balance:.2f}")
        print(f"          TD: mean={np.mean(ep_tds):.4f}, std={np.std(ep_tds):.4f}, "
              f"V(s): mean={np.mean(ep_vs):.4f}, std={np.std(ep_vs):.4f}")

# ── PHASE 3: Eligibility analysis on a fresh step ─────────────────────
print("\n### Phase 3: Post-training eligibility structure ###")
state = env.reset()
agent.reset()
a = agent.act(state)

e_d1 = agent.actor.e_d1
e_d2 = agent.actor.e_d2
e_h = agent.critic.e_h

print(f"  e_d1: shape={e_d1.shape}")
print(f"    neg%={np.mean(e_d1 < -1e-8):.4f}, pos%={np.mean(e_d1 > 1e-8):.4f}, zero%={np.mean(np.abs(e_d1) < 1e-8):.4f}")
print(f"    |mean|={np.mean(np.abs(e_d1)):.8f}, max_abs={np.max(np.abs(e_d1)):.6f}")

# Selected vs non-selected action
sel = agent.actor._last_action
for act_idx in range(2):
    s = act_idx * agent.actor.n_per_action
    e_block = e_d1[:, s:s+agent.actor.n_per_action]
    label = "SELECTED" if act_idx == sel else "non-sel "
    print(f"    {label} act{act_idx}: mean={np.mean(e_block):.8f}, |mean|={np.mean(np.abs(e_block)):.8f}")

# ── PHASE 4: Correlation analysis - does V(s) track episode length? ───
print("\n### Phase 4: V(s) vs episode quality correlation ###")
# Compute correlation between V(s) at step 0 and total reward
states_at_0 = []
v_at_0 = []
total_rewards = []

for ep in range(50):
    state = env.reset()
    agent.reset()
    
    agent.critic.reset_spike_counts(agent._n_substeps)
    pop = agent._pop_encoder.encode(state.astype(np.float32))
    for _ in range(agent._n_substeps):
        enc = agent._poisson.encode(pop)
        agent.critic.forward(enc)
    v0 = agent.critic.last_value
    
    # Need to re-reset for actual playing
    agent.reset()
    total_r = 0
    for step in range(500):
        a = agent.act(state)
        ns, r, done, info = env.step(a)
        agent.observe(state, a, r, ns, done, info)
        total_r += r
        state = ns
        if done: break
    
    v_at_0.append(v0)
    total_rewards.append(total_r)

corr = np.corrcoef(v_at_0, total_rewards)[0, 1]
print(f"  V(s0) vs total_reward correlation: {corr:.4f}")
print(f"  V(s0): mean={np.mean(v_at_0):.4f}, std={np.std(v_at_0):.4f}")
print(f"  rewards: mean={np.mean(total_rewards):.1f}, std={np.std(total_rewards):.1f}")

# ── PHASE 5: Weight direction analysis ─────────────────────────────────
print("\n### Phase 5: Are weights differentiating between actions? ###")
w_d1 = agent.actor.w_d1
w_d2 = agent.actor.w_d2
n_per = agent.actor.n_per_action

# Cosine sim between action 0 and action 1 weight columns
d1_a0 = w_d1[:, :n_per].mean(axis=1)
d1_a1 = w_d1[:, n_per:2*n_per].mean(axis=1)
cos_d1 = np.dot(d1_a0, d1_a1) / (np.linalg.norm(d1_a0) * np.linalg.norm(d1_a1) + 1e-10)

d2_a0 = w_d2[:, :n_per].mean(axis=1)
d2_a1 = w_d2[:, n_per:2*n_per].mean(axis=1)
cos_d2 = np.dot(d2_a0, d2_a1) / (np.linalg.norm(d2_a0) * np.linalg.norm(d2_a1) + 1e-10)

print(f"  cosine(w_d1[act0], w_d1[act1]) = {cos_d1:.4f}")
print(f"  cosine(w_d2[act0], w_d2[act1]) = {cos_d2:.4f}")
print(f"  d1_a0 norm={np.linalg.norm(d1_a0):.4f}, d1_a1 norm={np.linalg.norm(d1_a1):.4f}")
print(f"  d2_a0 norm={np.linalg.norm(d2_a0):.4f}, d2_a1 norm={np.linalg.norm(d2_a1):.4f}")

# ── PHASE 6: Check the REINFORCE override impact ─────────────────────
print("\n### Phase 6: REINFORCE override analysis ###")
state = env.reset()
agent.reset()

# Run act but capture eligibility at different stages
# First, let's see what the voltage-based STDP would give
agent.critic.reset_spike_counts(agent._n_substeps)
agent.actor.reset_spike_counts()

pop_rates = agent._pop_encoder.encode(state.astype(np.float32))
for _ in range(agent._n_substeps):
    enc = agent._poisson.encode(pop_rates)
    sensory = agent._build_sensory_inputs(enc, state.astype(np.float32))
    agent.network.step(sensory_inputs=sensory, neuromodulator=agent.neuromod, attention=None)

# This is the STDP eligibility BEFORE policy gradient override
e_d1_stdp = agent.actor.e_d1.copy()
e_d2_stdp = agent.actor.e_d2.copy()
print(f"  STDP eligibility (before override):")
print(f"    e_d1: neg%={np.mean(e_d1_stdp < -1e-8):.4f}, pos%={np.mean(e_d1_stdp > 1e-8):.4f}, |mean|={np.mean(np.abs(e_d1_stdp)):.8f}")

# Now apply the override
agent._set_actor_policy_gradient(pop_rates)
e_d1_pg = agent.actor.e_d1.copy()
print(f"  PG eligibility (after override):")
print(f"    e_d1: neg%={np.mean(e_d1_pg < -1e-8):.4f}, pos%={np.mean(e_d1_pg > 1e-8):.4f}, |mean|={np.mean(np.abs(e_d1_pg)):.8f}")
print(f"    ratio |PG|/|STDP| = {np.mean(np.abs(e_d1_pg))/(np.mean(np.abs(e_d1_stdp))+1e-15):.4f}")

env.close()
print("\n=== LEARNING DYNAMICS DIAGNOSTIC COMPLETE ===")
