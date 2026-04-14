import numpy as np, time, warnings
warnings.filterwarnings('ignore')
from arena.environments import TMazeEnv
from arena.snn_agent import SNNAgent

# Monkey-patch to disable sleep
orig_observe = SNNAgent.observe.__wrapped__ if hasattr(SNNAgent.observe, '__wrapped__') else None

np.random.seed(599)
env = TMazeEnv()
agent = SNNAgent(state_size=env.state_size, n_actions=env.n_actions, use_working_memory=True)
# Disable sleep by emptying replay buffer condition
agent._use_wm_sleep = False  # won't work, need to patch

# Actually just patch _needs_sleep to always return False, and set done branch
import types
def no_sleep(self): return False
agent._needs_sleep = types.MethodType(no_sleep, agent)
# Also need to prevent the done branch — let's just set _use_wm to True but not run sleep
# Patch: clear replay buffer after each store to prevent sleep
orig_use_wm = agent._use_wm

t0 = time.time()
rewards = []
for ep in range(2000):
    state = env.reset(seed=ep)
    agent.reset()
    total = 0.0
    for step in range(100):
        action = agent.act(state)
        ns, r, done, info = env.step(action)
        agent.observe(state, action, r, ns, done, info)
        total += r
        state = ns
        if done: break
    # Clear replay buffer to prevent sleep on next episode (sleep checks len > 0)
    agent.replay_buffer.clear()
    rewards.append(total)
    if (ep+1) % 50 == 0:
        m = np.mean(rewards[-50:])
        print(f'Ep {ep+1}: mean50={m:.2f} elapsed={time.time()-t0:.1f}s')

print(f'Last 100 mean: {np.mean(rewards[-100:]):.2f}')
print(f'Total time: {time.time()-t0:.1f}s')