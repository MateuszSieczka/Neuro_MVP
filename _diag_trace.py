"""Trace PunishmentAvoidance learning dynamics — deep diagnostic."""
import numpy as np
from arena.environments import PunishmentAvoidanceEnv
from arena.snn_agent import SNNAgent
from core.config import BasalGangliaConfig

SEED = 0
N_EP = 200
BLOCK = 50

np.random.seed(SEED * 11 + 3)
env = PunishmentAvoidanceEnv()
bg_cfg = BasalGangliaConfig()
agent = SNNAgent(
    state_size=env.state_size,
    n_actions=env.n_actions,
    bg_config=bg_cfg,
    use_world_model=False,
    use_working_memory=False,
)
actor = agent.bg.actor
npa = actor.n_per_action
n_in = actor.w_d1.shape[0]  # 30 (population encoded)
gc = actor._d2_gain_comp
print(f"Actor: n_inputs={n_in}, n_per_action={npa}, d2_gain_comp={gc:.4f}")
print(f"trace_decay={actor._trace_decay:.6f}, _e_compl={1-actor._trace_decay:.6f}")
print(f"w_d1 shape={actor.w_d1.shape}, init mean={actor.w_d1.mean():.6f}")

ep_rewards = []
for ep in range(N_EP):
    state = env.reset(seed=ep)
    agent.reset()
    ctx = env._context
    
    # Snapshot weights BEFORE this episode
    w_d1_pre = actor.w_d1.copy()
    w_d2_pre = actor.w_d2.copy()
    
    action = agent.act(state)
    
    # Snapshot eligibility AFTER act (before update)
    e_d1_snap = actor.e_d1.copy()
    e_d2_snap = actor.e_d2.copy()
    
    next_state, reward, done, info = env.step(action)
    agent.observe(state, action, reward, next_state, done, info)
    
    # Weight change this episode
    dw_d1 = actor.w_d1 - w_d1_pre
    dw_d2 = actor.w_d2 - w_d2_pre
    
    td = agent._last_td_error
    ep_rewards.append(reward)
    
    # Print first 20 episodes and every BLOCK after
    if ep < 20 or (ep + 1) % BLOCK == 0:
        # Eligibility for chosen action neurons
        e_d1_act = e_d1_snap[:, action*npa:(action+1)*npa]
        e_d2_act = e_d2_snap[:, action*npa:(action+1)*npa]
        
        # Eligibility for non-chosen action
        other = 1 - action
        e_d1_oth = e_d1_snap[:, other*npa:(other+1)*npa]
        e_d2_oth = e_d2_snap[:, other*npa:(other+1)*npa]
        
        # Weight changes for chosen action
        dw_d1_act = np.abs(dw_d1[:, action*npa:(action+1)*npa]).mean()
        dw_d2_act = np.abs(dw_d2[:, action*npa:(action+1)*npa]).mean()
        # Weight changes for other action
        dw_d1_oth = np.abs(dw_d1[:, other*npa:(other+1)*npa]).mean()
        dw_d2_oth = np.abs(dw_d2[:, other*npa:(other+1)*npa]).mean()
        
        print(f"ep{ep:3d} ctx={ctx} a={action} r={reward:+.0f} td={td:+.3f} "
              f"| e_d1[act]={e_d1_act.mean():.5f} max={e_d1_act.max():.5f} "
              f"| e_d2[act]={e_d2_act.mean():.5f} max={e_d2_act.max():.5f} "
              f"| dw_d1[act]={dw_d1_act:.7f} dw_d2[act]={dw_d2_act:.7f} "
              f"| dw_d1[oth]={dw_d1_oth:.7f} dw_d2[oth]={dw_d2_oth:.7f}")

late = float(np.mean(ep_rewards[-100:]))
print(f"\nFinal late-100 mean: {late:.2f}")
