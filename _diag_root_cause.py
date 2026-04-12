"""
ROOT CAUSE DIAGNOSTIC: Traces the full learning pipeline to find why
the network doesn't learn on CartPole.

Key questions:
1. Are eligibility traces ever negative (LTD working)?
2. Is the critic V(s) actually tracking reward?
3. Are TD errors meaningful and correctly computed?
4. Are weight updates actually non-zero?
5. Is the actor action selection differentiating between states?
6. Does the voltage-based eligibility produce useful gradients?
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
print("ROOT CAUSE DIAGNOSTIC")
print("=" * 80)

# ===== TEST 1: Eligibility trace sign analysis =====
print("\n### TEST 1: Eligibility trace sign after single step ###")
state = env.reset()
agent.reset()

# Do one act
a = agent.act(state)
ns, r, done, info = env.step(a)

# Check eligibility traces BEFORE observe (accumulated during act)
print(f"After act(), before observe():")
print(f"  e_d1: min={np.min(agent.actor.e_d1):.6f}, max={np.max(agent.actor.e_d1):.6f}")
print(f"  e_d2: min={np.min(agent.actor.e_d2):.6f}, max={np.max(agent.actor.e_d2):.6f}")
print(f"  e_d1 negative fraction: {np.mean(agent.actor.e_d1 < 0):.4f}")
print(f"  e_d2 negative fraction: {np.mean(agent.actor.e_d2 < 0):.4f}")
print(f"  e_d1 nonzero fraction: {np.mean(np.abs(agent.actor.e_d1) > 1e-8):.4f}")
print(f"  e_d2 nonzero fraction: {np.mean(np.abs(agent.actor.e_d2) > 1e-8):.4f}")

# The formula for eligibility in the actor is:
# e_d1 = e_d1 * trace_decay + outer(pre_binary, v_d1_elig)
# where v_d1_elig = v_d1_norm + spikes_d1  (always >= 0!)
# This means e_d1 is ALWAYS >= 0. LTD is IMPOSSIBLE.
print(f"\n  CRITICAL: v_d1_elig = v_d1_norm + spikes_d1")
print(f"  v_d1_norm is clipped to [0,1], spikes_d1 is 0 or 1")
print(f"  pre_binary is 0 or 1")
print(f"  => outer(pre_binary, v_d1_elig) is ALWAYS >= 0")
print(f"  => e_d1 = decay * e_d1 + (>=0) is ALWAYS >= 0 if initialized at 0")
print(f"  => LTD IS STRUCTURALLY IMPOSSIBLE WITH VOLTAGE-BASED ELIGIBILITY")

env.close()

# ===== TEST 2: Actor update analysis =====
print("\n\n### TEST 2: Actor update() sign analysis ###")

env2 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent2 = make_agent(task, env2)

state2 = env2.reset()
agent2.reset()

# Run a few steps
for _ in range(10):
    a2 = agent2.act(state2)
    ns2, r2, done2, info2 = env2.step(a2)
    
    # Save weights before update
    w_d1_before = agent2.actor.w_d1.copy()
    w_d2_before = agent2.actor.w_d2.copy()
    w_h_before = agent2.critic.w_h.copy()
    w_v_before = agent2.critic.w_v.copy()
    
    agent2.observe(state2, a2, r2, ns2, done2, info2)
    
    td = agent2._last_td_error
    dw_d1 = agent2.actor.w_d1 - w_d1_before
    dw_d2 = agent2.actor.w_d2 - w_d2_before
    dw_h = agent2.critic.w_h - w_h_before
    dw_v = agent2.critic.w_v - w_v_before
    
    print(f"  TD={td:+.4f}: "
          f"|dw_d1|={np.mean(np.abs(dw_d1)):.6f}, "
          f"|dw_d2|={np.mean(np.abs(dw_d2)):.6f}, "
          f"|dw_h|={np.mean(np.abs(dw_h)):.6f}, "
          f"|dw_v|={np.mean(np.abs(dw_v)):.6f}")
    print(f"         dw_d1 range=[{np.min(dw_d1):.6f}, {np.max(dw_d1):.6f}], "
          f"dw_d2 range=[{np.min(dw_d2):.6f}, {np.max(dw_d2):.6f}]")
    
    state2 = ns2
    if done2: break

env2.close()

# ===== TEST 3: Actor update rule traced mathematically =====
print("\n\n### TEST 3: Manual actor update rule trace ###")
print("Actor update rule (from code):")
print("  D1: dw = lr * max(td,0) * e_d1 - lr * ltd_ratio * max(-td,0) * e_d1")
print("  D2: dw = lr * max(-td,0) * e_d2 - lr * ltd_ratio * max(td,0) * e_d2") 
print("")
print("  Since e_d1 >= 0 and e_d2 >= 0 ALWAYS:")
print("  When td > 0 (reward better than expected):")
print("    D1: dw = +lr * td * e_d1   (ALWAYS >= 0, D1 weights ONLY increase)")
print("    D2: dw = -lr * ltd_ratio * td * e_d2  (ALWAYS <= 0, D2 weights decrease)")
print("  When td < 0 (reward worse than expected):")
print("    D1: dw = -lr * ltd_ratio * |td| * e_d1  (ALWAYS <= 0, D1 weights decrease)")
print("    D2: dw = +lr * |td| * e_d2  (ALWAYS >= 0, D2 weights increase)")
print("")
print("  PROBLEM: With no LTD in eligibility, the update is:")
print("    D1 always grows when td>0, D2 always grows when td<0")
print("    There's no way to SELECTIVELY weaken specific pathways")
print("    This is equivalent to a rate-based rule, NOT proper STDP")
print("    The 'Go/NoGo' learning degenerates to simple Hebbian accumulation")

# ===== TEST 4: Critic V(s) trajectory =====
print("\n\n### TEST 4: Full episode V(s) trajectory ###")
env3 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent3 = make_agent(task, env3)

episode_data = []
for ep in range(30):
    state3 = env3.reset()
    agent3.reset()
    ep_v = []
    ep_td = []
    ep_r = []
    total_r = 0
    for step in range(500):
        a3 = agent3.act(state3)
        ns3, r3, done3, info3 = env3.step(a3)
        
        v_before = agent3.critic.last_value
        agent3.observe(state3, a3, r3, ns3, done3, info3)
        
        ep_v.append(v_before)
        ep_td.append(agent3._last_td_error)
        ep_r.append(r3)
        total_r += r3
        state3 = ns3
        if done3: break
    
    episode_data.append({
        'reward': total_r,
        'v_mean': np.mean(ep_v),
        'v_std': np.std(ep_v),
        'td_mean': np.mean(ep_td),
        'td_std': np.std(ep_td),
        'steps': step + 1,
    })
    
    if (ep+1) % 10 == 0:
        ed = episode_data[-1]
        print(f"  Ep {ep+1}: R={ed['reward']:.0f}, steps={ed['steps']}, "
              f"V(s)={ed['v_mean']:.2f}±{ed['v_std']:.2f}, "
              f"TD={ed['td_mean']:.3f}±{ed['td_std']:.3f}")

env3.close()

# ===== TEST 5: Weight norm evolution =====
print("\n\n### TEST 5: Weight floor and Dale's law effect ###")
env4 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent4 = make_agent(task, env4)

print(f"  Initial w_d1: min={np.min(agent4.actor.w_d1):.4f}, max={np.max(agent4.actor.w_d1):.4f}")
print(f"  Initial w_d2: min={np.min(agent4.actor.w_d2):.4f}, max={np.max(agent4.actor.w_d2):.4f}")
print(f"  Initial w_floor applied at: 0.01")
print(f"  This means weights CANNOT go below 0.01")
print(f"  Combined with no negative eligibility: ")
print(f"    When td>0: D1 grows, D2 shrinks but floor=0.01 prevents death")
print(f"    When td<0: D1 shrinks but floor=0.01 prevents death, D2 grows")
print(f"  Net effect over time: BOTH D1 and D2 oscillate near the floor or grow")
print(f"  There's no mechanism for differential learning across actions")

# Run some episodes
for ep in range(50):
    state4 = env4.reset()
    agent4.reset()
    for step in range(500):
        a4 = agent4.act(state4)
        ns4, r4, done4, info4 = env4.step(a4)
        agent4.observe(state4, a4, r4, ns4, done4, info4)
        state4 = ns4
        if done4: break
    
    if (ep+1) % 25 == 0:
        w_d1 = agent4.actor.w_d1
        w_d2 = agent4.actor.w_d2
        print(f"  Ep {ep+1}: w_d1=[{np.min(w_d1):.4f}, {np.max(w_d1):.4f}] mean={np.mean(w_d1):.4f}, "
              f"w_d2=[{np.min(w_d2):.4f}, {np.max(w_d2):.4f}] mean={np.mean(w_d2):.4f}")
        # Show per-action column norms
        for act_idx in range(agent4.actor.motor_dim):
            start = act_idx * agent4.actor.n_per_action
            end = start + agent4.actor.n_per_action
            d1_norm = np.linalg.norm(w_d1[:, start:end])
            d2_norm = np.linalg.norm(w_d2[:, start:end])
            print(f"    Action {act_idx}: ||w_d1||={d1_norm:.3f}, ||w_d2||={d2_norm:.3f}, diff={d1_norm-d2_norm:.3f}")

env4.close()

# ===== TEST 6: Critic eligibility — does it have LTD? =====
print("\n\n### TEST 6: Critic eligibility trace sign ###")
env5 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent5 = make_agent(task, env5)
state5 = env5.reset()
agent5.reset()

for step in range(30):
    a5 = agent5.act(state5)
    ns5, r5, done5, info5 = env5.step(a5)
    
    print(f"  Step {step}: e_h range=[{np.min(agent5.critic.e_h):.4f}, {np.max(agent5.critic.e_h):.4f}], "
          f"neg_frac={np.mean(agent5.critic.e_h < 0):.4f}, "
          f"e_v range=[{np.min(agent5.critic.e_v):.6f}, {np.max(agent5.critic.e_v):.6f}], "
          f"neg_frac={np.mean(agent5.critic.e_v < 0):.4f}")
    
    agent5.observe(state5, a5, r5, ns5, done5, info5)
    state5 = ns5
    if done5: break

env5.close()

# ===== TEST 7: Action selection discrimination =====
print("\n\n### TEST 7: Action selection - can the network discriminate states? ###")
env6 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent6 = make_agent(task, env6)

# Different states that should produce different actions
# In CartPole: if pole leans right (positive angle), push right (action 1)
# if pole leans left (negative angle), push left (action 0)
test_states = {
    'lean_right': np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float32),
    'lean_left': np.array([0.0, 0.0, -0.1, 0.0], dtype=np.float32),
    'center': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}

# Normalize these like env would
obs_low, obs_high = task.obs_bounds
for name, s in test_states.items():
    test_states[name] = 2.0 * (s - obs_low) / (obs_high - obs_low) - 1.0

# Test before any learning
print("  Before learning:")
for name, s in test_states.items():
    actions = []
    for trial in range(20):
        agent6.reset()
        a = agent6.act(s)
        actions.append(a)
    print(f"    {name}: action counts = {np.bincount(actions, minlength=2)}")

# Train for 50 episodes
state6 = None
for ep in range(50):
    state6 = env6.reset()
    agent6.reset()
    for _ in range(500):
        a6 = agent6.act(state6)
        ns6, r6, done6, info6 = env6.step(a6)
        agent6.observe(state6, a6, r6, ns6, done6, info6)
        state6 = ns6
        if done6: break

print("  After 50 episodes:")
for name, s in test_states.items():
    actions = []
    for trial in range(20):
        agent6.reset()
        a = agent6.act(s)
        actions.append(a)
    print(f"    {name}: action counts = {np.bincount(actions, minlength=2)}")

env6.close()

# ===== TEST 8: V_accum membrane potential readout =====
print("\n\n### TEST 8: V_accum vs spike count for action discrimination ###")
env7 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent7 = make_agent(task, env7)
state7 = env7.reset()
agent7.reset()

for step in range(5):
    a7 = agent7.act(state7)
    
    # After act, check the v_accum values
    actor = agent7.actor
    n_sub = agent7._n_substeps
    v_d1 = actor._v_accum_d1[:actor._total_motor].reshape(actor.motor_dim, actor.n_per_action).sum(axis=1)
    v_d2 = actor._v_accum_d2[:actor._total_motor].reshape(actor.motor_dim, actor.n_per_action).sum(axis=1)
    sc_d1 = actor._spike_count_d1[:actor._total_motor].reshape(actor.motor_dim, actor.n_per_action).sum(axis=1)
    sc_d2 = actor._spike_count_d2[:actor._total_motor].reshape(actor.motor_dim, actor.n_per_action).sum(axis=1)
    
    _norm = actor.n_per_action * np.sqrt(float(max(actor._n_forward, 1)))
    net_ev = (v_d1 - v_d2) / _norm
    
    print(f"  Step {step}: action={a7}")
    print(f"    v_accum_d1={v_d1}, v_accum_d2={v_d2}")
    print(f"    spike_count_d1={sc_d1}, spike_count_d2={sc_d2}")
    print(f"    net_evidence (v-based)={net_ev}")
    print(f"    n_forward={actor._n_forward}, norm={_norm:.2f}")
    
    ns7, r7, done7, info7 = env7.step(a7)
    agent7.observe(state7, a7, r7, ns7, done7, info7)
    state7 = ns7
    if done7: break

env7.close()

print("\n" + "=" * 80)
print("SUMMARY OF ROOT CAUSES")
print("=" * 80)
print("""
1. ACTOR ELIGIBILITY IS ALWAYS >= 0 (CRITICAL BUG):
   The voltage-based eligibility: e = decay*e + outer(pre, v_norm + spike)
   Since v_norm in [0,1] and spike in {0,1}, the update is always >= 0.
   This means e_d1 and e_d2 are always non-negative.
   => No LTD is possible => pure rate-based Hebbian accumulation
   => No proper credit assignment for Go vs NoGo pathways

2. WEIGHT FLOOR PREVENTS DIFFERENTIAL LEARNING:
   np.maximum(w_d1, 0.01) prevents weights from reaching zero.
   Combined with always-positive eligibility, weights oscillate
   without developing meaningful action-state associations.

3. CRITIC ELIGIBILITY HAS PROPER LTD (via STDP):
   The critic uses causal STDP windows with signed eligibility.
   BUT the readout eligibility (e_v) tracks activation (EMA of spikes),
   which is always >= 0. So the V(s) readout also lacks proper LTD.

4. V(S) CONVERGES TO BIAS (NOT LEARNING):
   Membrane-potential readout may not properly track return value.
   The critic learns very slowly due to sparse spiking.
""")
