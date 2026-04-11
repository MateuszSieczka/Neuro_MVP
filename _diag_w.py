import numpy as np; np.random.seed(42)
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena.task_config import get as get_task

task = get_task("CartPole-v1")
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
state = env.reset(seed=42)
agent = make_agent(task, env)
actor = agent.actor

w_d1_init = actor.w_d1.copy()
w_d2_init = actor.w_d2.copy()

scores = []
for ep in range(100):
    state = env.reset()
    agent.reset()
    ep_r = 0.0
    for step in range(500):
        action = agent.act(state)
        ns, r, done, info = env.step(action)
        agent.observe(state, action, r, ns, done, info)
        ep_r += r
        state = ns
        if done:
            break
    scores.append(ep_r)
    if ep % 20 == 19:
        d1_diff = np.linalg.norm(actor.w_d1[:,0] - actor.w_d1[:,1])
        d1_chg = np.linalg.norm(actor.w_d1 - w_d1_init)
        d2_chg = np.linalg.norm(actor.w_d2 - w_d2_init)
        avg_score = np.mean(scores[-20:])
        print(f"Ep {ep+1:3d} | avg_R={avg_score:.1f} | d1_col_diff={d1_diff:.3f} | d1_chg={d1_chg:.3f} | d2_chg={d2_chg:.3f}")
env.close()
