"""
DIAGNOSTIC 2: Deeper analysis of critic V(s) readout and weight dynamics.

Questions:
1. Does V(s) track cumulative reward at all?
2. Is the readout weight update magnitude sufficient?
3. Is w_v growing without bound or stabilizing?
4. What happens to eligibility norms over 500 episodes?
5. Is homeostatic scaling helping or hurting?
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
print("DIAGNOSTIC 2: Critic V(s) readout and weight dynamics")
print("=" * 80)

# Track everything across episodes
all_scores = []
critic_data = []

for ep in range(200):
    state = env.reset()
    agent.reset()
    total_r = 0
    
    ep_v_first5 = []
    ep_v_last5 = []
    
    for step in range(500):
        a = agent.act(state)
        ns, r, done, info = env.step(a)
        
        v = agent.critic.last_value
        if step < 5:
            ep_v_first5.append(v)
        
        # Save weights before observe
        w_v_pre = agent.critic.w_v.copy()
        w_h_pre = agent.critic.w_h.copy()
        
        agent.observe(state, a, r, ns, done, info)
        
        dw_v = agent.critic.w_v - w_v_pre
        dw_h = agent.critic.w_h - w_h_pre
        
        if step >= max(0, total_r + r - 6):  # near-end steps
            ep_v_last5.append(v)
        
        total_r += r
        state = ns
        if done: break
    
    all_scores.append(total_r)
    
    # Record critic state
    critic_data.append({
        'w_v_abs': float(np.mean(np.abs(agent.critic.w_v))),
        'w_v_max': float(np.max(np.abs(agent.critic.w_v))),
        'w_h_abs': float(np.mean(np.abs(agent.critic.w_h))),
        'b_v': agent.critic.b_v,
        'v_first5': np.mean(ep_v_first5) if ep_v_first5 else 0,
        'v_last5': np.mean(ep_v_last5) if ep_v_last5 else 0,
        'e_h_norm': float(np.linalg.norm(agent.critic.e_h)),
        'e_v_norm': float(np.linalg.norm(agent.critic.e_v)),
        'activation_mean': float(np.mean(agent.critic.activation)),
        'homeo_rate_mean': float(np.mean(agent.critic._homeo_rate)),
    })
    
    if (ep+1) % 25 == 0:
        cd = critic_data[-1]
        last25 = all_scores[-25:]
        print(f"\nEp {ep+1}: score mean25={np.mean(last25):.1f}, max25={np.max(last25)}")
        print(f"  w_v: abs_mean={cd['w_v_abs']:.4f}, max={cd['w_v_max']:.4f}, b_v={cd['b_v']:.4f}")
        print(f"  w_h: abs_mean={cd['w_h_abs']:.4f}")
        print(f"  V(s): first5={cd['v_first5']:.2f}, last5={cd['v_last5']:.2f}")
        print(f"  e_h norm={cd['e_h_norm']:.2f}, e_v norm={cd['e_v_norm']:.4f}")
        print(f"  activation={cd['activation_mean']:.6f}, homeo_rate={cd['homeo_rate_mean']:.6f}")

env.close()

# Analyze trends
print("\n\n### V(s) TREND ANALYSIS ###")
v_firsts = [cd['v_first5'] for cd in critic_data]
w_v_abs_trend = [cd['w_v_abs'] for cd in critic_data]
print(f"V(s) at episode start:")
print(f"  Ep 1-25:   mean={np.mean(v_firsts[:25]):.2f}")
print(f"  Ep 26-50:  mean={np.mean(v_firsts[25:50]):.2f}")
print(f"  Ep 51-100: mean={np.mean(v_firsts[50:100]):.2f}")
print(f"  Ep 101-200: mean={np.mean(v_firsts[100:]):.2f}")

print(f"\n|w_v| trend:")
print(f"  Ep 1-25:   mean={np.mean(w_v_abs_trend[:25]):.4f}")
print(f"  Ep 26-50:  mean={np.mean(w_v_abs_trend[25:50]):.4f}")
print(f"  Ep 51-100: mean={np.mean(w_v_abs_trend[50:100]):.4f}")
print(f"  Ep 101-200: mean={np.mean(w_v_abs_trend[100:]):.4f}")

# ===== Test: Is the update LR too small? =====
print("\n\n### LEARNING RATE EFFECTIVE ANALYSIS ###")
bg_cfg = agent._bg_config
print(f"critic_lr = {bg_cfg.critic_lr}")
print(f"actor_lr = {bg_cfg.actor_lr}")

# For a typical step:
# dw_v = lr * td * e_v
# e_v tracks activation (spike rate EMA), typical ~0.01
# td typical ~1.0
# dw_v ~ 0.007 * 1.0 * 0.01 = 0.00007 per element
print(f"Typical dw_v per element = {bg_cfg.critic_lr} * 1.0 * 0.01 = {bg_cfg.critic_lr * 1.0 * 0.01:.6f}")
print(f"With 128 neurons and 500 steps/ep, cumulative change = {bg_cfg.critic_lr * 1.0 * 0.01 * 500:.4f}")
print(f"This is very small relative to initial w_v scale = 1/sqrt(128) = {1/np.sqrt(128):.4f}")

# For critic hidden weights:
# dw_h = lr * td * e_h
# e_h has both + and - entries, typical magnitude ~0.01
# dw_h ~ 0.007 * 1.0 * 0.01 = 0.00007
print(f"\nTypical dw_h per element = {bg_cfg.critic_lr} * 1.0 * 0.01 = {bg_cfg.critic_lr * 1.0 * 0.01:.6f}")
print(f"w_h column norm target = {bg_cfg.w_clip_critic}")

# For actor:
# e_d1 tracks outer(pre, v_norm+spike), typical ~0.5 per active
# dw_d1 ~ 0.005 * 1.0 * 0.5 = 0.0025 per active element
print(f"\nTypical dw_d1 per element = {bg_cfg.actor_lr} * 1.0 * 0.5 = {bg_cfg.actor_lr * 1.0 * 0.5:.6f}")
print(f"w_d1 column norm target = {bg_cfg.w_clip}")

# ===== V(s) readout mechanism =====
print("\n\n### V(S) READOUT MECHANISM ANALYSIS ###")
print("V(s) = dot(w_v, v_mean) + b_v")
print("where v_mean = v_accum / n_substeps")
print("v_accum = sum over substeps of normalized voltage [0,1]")
print(f"n_substeps = {agent._n_substeps}")
print(f"So v_mean is in [0, 1] (mean normalized voltage per neuron)")
print(f"With 128 neurons, V(s) = sum(w_v_i * v_mean_i) + b_v")
print(f"  = ~128 * mean(w_v) * mean(v_mean) + b_v")
print(f"Expected range: 128 * 0.08 * 0.5 = {128 * 0.08 * 0.5:.1f}")
print(f"Actual V(s) range from data: ~0 to ~30")
print()
print("The issue: V(s) depends on MEMBRANE POTENTIAL not spikes")
print("Membrane potential is driven by synaptic current, which is always positive")
print("(because all input weights are excitatory, Dale's law)")
print("So V(s) is essentially a weighted sum of 'how depolarized each neuron is'")
print("This IS informative about the input state, BUT:")
print("  - It always reads positive (depolarization above rest)")
print("  - The mapping from state→V(s) is noisy and indirect")
print("  - Readout weights w_v need to learn negative values for some neurons")
print("    to create differential V(s) across states")

# ===== Check: Does the critic spike at all? =====
print("\n\n### CRITIC SPIKING ANALYSIS ###")
env2 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent2 = make_agent(task, env2)
state2 = env2.reset()
agent2.reset()

total_spikes = 0
total_substeps = 0
for step in range(50):
    a2 = agent2.act(state2)
    # Count spikes during the act
    total_spikes += np.sum(agent2.critic._spike_count)
    total_substeps += agent2._n_substeps
    
    ns2, r2, done2, info2 = env2.step(a2)
    agent2.observe(state2, a2, r2, ns2, done2, info2)
    state2 = ns2
    if done2:
        state2 = env2.reset()
        agent2.reset()

print(f"Over 50 env steps ({total_substeps} substeps, 128 critic neurons):")
print(f"  Total spikes: {total_spikes:.0f}")
print(f"  Mean per substep per neuron: {total_spikes / (total_substeps * 128):.4f}")
print(f"  This is the spike rate that drives eligibility e_v")
print(f"  With activation EMA decay={agent2.critic._rate_decay:.4f}:")
print(f"    activation converges to: spike_rate / (1 - decay) = {total_spikes / (total_substeps * 128) / (1 - agent2.critic._rate_decay):.4f}")

env2.close()

# ===== Check: readout_decay effect =====
print("\n\n### READOUT DECAY ANALYSIS ###")
print(f"readout_decay = {bg_cfg.readout_decay}")
print(f"Per step: w_v *= (1 - {bg_cfg.readout_decay}) = {1 - bg_cfg.readout_decay}")
print(f"Per episode (20 steps): w_v *= {(1 - bg_cfg.readout_decay)**20:.8f}")
print(f"Per 1000 steps: w_v *= {(1 - bg_cfg.readout_decay)**1000:.6f}")
print(f"This is negligible: {1 - (1-bg_cfg.readout_decay)**1000:.6f} fractional loss")
