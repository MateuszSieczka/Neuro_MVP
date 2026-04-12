"""
DIAGNOSTIC 3: Verify gate_eligibility behavior and its impact.
Also test: is the action gating destroying learning signal?
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
print("DIAGNOSTIC 3: Eligibility gating + update flow analysis")
print("=" * 80)

state = env.reset()
agent.reset()

for step in range(10):
    # Before act: check eligibility
    e_d1_pre_act = agent.actor.e_d1.copy()
    e_d2_pre_act = agent.actor.e_d2.copy()
    
    a = agent.act(state)
    
    # After act, before gating (gating happens inside act)
    e_d1_after_act = agent.actor.e_d1.copy()
    e_d2_after_act = agent.actor.e_d2.copy()
    
    # Check which action columns are zeroed
    npa = agent.actor.n_per_action
    e_selected_start = a * npa
    e_selected_end = e_selected_start + npa
    e_selected_d1 = e_d1_after_act[:, e_selected_start:e_selected_end]
    e_other_d1 = np.delete(e_d1_after_act[:, :agent.actor._total_motor], 
                           list(range(e_selected_start, e_selected_end)), axis=1)
    
    print(f"\nStep {step}: action={a}")
    print(f"  e_d1 selected action col norm: {np.linalg.norm(e_selected_d1):.4f}")
    print(f"  e_d1 other action col norm:    {np.linalg.norm(e_other_d1):.4f}")
    print(f"  e_d1 total motor norm: {np.linalg.norm(e_d1_after_act[:, :agent.actor._total_motor]):.4f}")
    
    # Now observe - this does the update
    w_d1_before = agent.actor.w_d1.copy()
    w_d2_before = agent.actor.w_d2.copy()
    
    ns, r, done, info = env.step(a)
    agent.observe(state, a, r, ns, done, info)
    
    td = agent._last_td_error
    dw_d1 = agent.actor.w_d1 - w_d1_before
    dw_d2 = agent.actor.w_d2 - w_d2_before
    
    # How much change per action column?
    for act_idx in range(agent.actor.motor_dim):
        start = act_idx * npa
        end = start + npa
        d1_change = np.linalg.norm(dw_d1[:, start:end])
        d2_change = np.linalg.norm(dw_d2[:, start:end])
        marker = " <-- selected" if act_idx == a else ""
        print(f"  Action {act_idx}: |dw_d1|={d1_change:.6f}, |dw_d2|={d2_change:.6f}{marker}")
    
    state = ns
    if done: break

env.close()

# ===== Test: What does the DA modulation look like? =====
print("\n\n### DA MODULATION EFFECT ON CURRENTS ###")
print("In forward():")
print("  d1_mod = d1_bias + da * (1 - d1_bias) = 0.6 + da * 0.4")  
print("  d2_mod = d2_bias * (1 - da) + (1 - d2_bias) = 0.4*(1-da) + 0.6")
print()
for da in [0.0, 0.25, 0.5, 0.75, 1.0]:
    d1_mod = 0.6 + da * 0.4
    d2_mod = 0.4 * (1 - da) + 0.6
    print(f"  DA={da:.2f}: D1_mod={d1_mod:.2f}, D2_mod={d2_mod:.2f}, ratio D1/D2={d1_mod/d2_mod:.2f}")

print("\nNote: DA is typically around 0.6 (from diagnostics)")
print("At DA=0.6: D1_mod=0.84, D2_mod=0.76, ratio=1.11")
print("This is a VERY small difference. D1 gets only 11% more current than D2")
print("Combined with the fact that both have the same initial weights,")
print("and eligibility is always positive, the Go/NoGo signal is very weak.")

# ===== Test: eligibility magnitude vs trace decay =====
print("\n\n### ELIGIBILITY TRACE ACCUMULATION vs DECAY ###")
env2 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent2 = make_agent(task, env2)
state2 = env2.reset()
agent2.reset()

trace_decay = agent2.actor._trace_decay
print(f"Actor trace_decay per substep: {trace_decay:.6f}")
print(f"Over {agent2._n_substeps} substeps: {trace_decay**agent2._n_substeps:.6f}")
print(f"=> Eligibility from first substep retains {trace_decay**agent2._n_substeps*100:.1f}% by last substep")
print()
print(f"Critic trace_decay per substep: {agent2.critic._trace_decay:.6f}")
print(f"Over {agent2._n_substeps} substeps: {agent2.critic._trace_decay**agent2._n_substeps:.6f}")

# During act, n_substeps=25, each substep adds to eligibility
# The cumulative eligibility = sum_{t=0}^{24} decay^(24-t) * e_instant_t  
# If e_instant is constant c per step:
# sum = c * (1 - decay^25) / (1 - decay)
print(f"\nIf per-substep eligibility contribution is constant c:")
print(f"  Actor cumulative = c * (1 - {trace_decay}^25) / (1 - {trace_decay}) = c * {(1 - trace_decay**25) / (1 - trace_decay):.2f}")
print(f"  Critic cumulative = c * (1 - {agent2.critic._trace_decay}^25) / (1 - {agent2.critic._trace_decay}) = c * {(1 - agent2.critic._trace_decay**25) / (1 - agent2.critic._trace_decay):.2f}")

env2.close()

# ===== Verify the save/restore of eligibility in observe =====
print("\n\n### SAVE/RESTORE ELIGIBILITY IN OBSERVE ###")
env3 = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent3 = make_agent(task, env3)
state3 = env3.reset()
agent3.reset()

a3 = agent3.act(state3)
e_d1_after_act = agent3.actor.e_d1.copy()
e_d2_after_act = agent3.actor.e_d2.copy()
e_h_after_act = agent3.critic.e_h.copy()
e_v_after_act = agent3.critic.e_v.copy()

ns3, r3, done3, info3 = env3.step(a3)

# Now observe will save, integrate critic for V(s'), restore, then update
# We need to check that after observe, the eligibility used for update
# was indeed the one from act (not from the V(s') integration)
agent3.observe(state3, a3, r3, ns3, done3, info3)

# The update() in observe uses the RESTORED eligibility
# But after update(), new traces are NOT set - observe doesn't run forward again
# So we need to check: is the update applied to the correct traces?
print("After observe, actor eligibility should have been used from act:")
print(f"  e_d1 norm after act: {np.linalg.norm(e_d1_after_act):.4f}")
print(f"  e_d1 norm after obs: {np.linalg.norm(agent3.actor.e_d1):.4f}")
print("  (Should differ only by the gate_eligibility zeroing + update dw)")

env3.close()

# ===== Net Evidence normalization analysis =====
print("\n\n### NET EVIDENCE NORMALIZATION ###")
print("net_evidence = (v_accum_d1 - v_accum_d2) / (n_per_action * sqrt(n_forward))")
print(f"n_per_action = 8, n_forward = 25")
print(f"norm = 8 * sqrt(25) = 8 * 5 = 40")
print()
print("From TEST 8 data:")
print("  v_accum_d1 per action ~ 130-160 (8 neurons, 25 substeps, ~0.7 avg normalized voltage)")
print("  v_accum_d2 per action ~ 120-150")
print("  net evidence ~ ±0.3 to ±0.7")
print("  temperature = 1 + 4*(NE-0.5)^2, NE~0.43 => T ≈ 1 + 4*0.0049 = 1.02")
print("  noise std = temperature = 1.02")
print("  So noise ~ N(0, 1.02) is COMPARABLE to signal ~0.5")
print("  => Action selection is largely random!")
print()
print("This is by design (exploration), but it means that:")
print("  1. Early episodes are essentially random")
print("  2. Learning signal (TD error) is based on random behavior")
print("  3. TD-modulated STDP must overcome this noise to learn")
print("  4. With always-positive eligibility, it CANNOT do proper credit assignment")
