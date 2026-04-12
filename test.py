import numpy as np; np.random.seed(42)
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena.task_config import get as get_task
task = get_task('CartPole-v1')
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
agent = make_agent(task, env)
scores = []
for ep in range(1000):
    state = env.reset()
    agent.reset()
    total_r = 0
    for _ in range(500):
        a = agent.act(state)
        ns, r, done, info = env.step(a)
        agent.observe(state, a, r, ns, done, info)
        total_r += r
        state = ns
        if done: break
    scores.append(total_r)
    if (ep+1) % 25 == 0:
        last100 = scores[-25:]
        c = agent.critic
        a_obj = agent.actor
        print(f'Ep {(ep+1):4d}: m100={np.mean(last100):.1f}, max={np.max(last100):.0f}, V_s={agent.vta.last_v_s:.2f}, V_sp={agent.vta.last_v_s_prime:.2f}, wv_norm={np.linalg.norm(agent.vta.w_value):.3f}, d1={np.mean(np.abs(a_obj.w_d1)):.4f}, d2={np.mean(np.abs(a_obj.w_d2)):.4f}, ent={a_obj.action_entropy:.3f}, rpe={agent.vta.last_rpe:.3f}, gamma={agent.vta.last_gamma_eff:.4f}')
env.close()
print(f'\nFinal: mean={np.mean(scores):.1f}, last100={np.mean(scores[-25:]):.1f}, max={np.max(scores):.0f}')
for i in range(0, len(scores), 200):
    chunk = scores[i:i+200]
    print(f'  Ep {i+1}-{i+len(chunk)}: mean={np.mean(chunk):.1f}, max={np.max(chunk):.0f}')