"""Check PunishmentAvoidance with different episode counts."""
import numpy as np
from arena.environments import PunishmentAvoidanceEnv
from arena.snn_agent import SNNAgent
from core.config import BasalGangliaConfig

for n_ep in [500, 750, 1000]:
    scores = []
    for seed in range(5):
        np.random.seed(seed * 11 + 3)
        env = PunishmentAvoidanceEnv()
        agent = SNNAgent(
            state_size=env.state_size, n_actions=env.n_actions,
            bg_config=BasalGangliaConfig(),
            use_world_model=False, use_working_memory=False,
        )
        rewards = []
        for ep in range(n_ep):
            state = env.reset(seed=ep)
            agent.reset()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done, info)
            rewards.append(reward)
        late = float(np.mean(rewards[-100:]))
        scores.append(late)
    print(f"Episodes={n_ep}: {[f'{s:.2f}' for s in scores]} mean={np.mean(scores):.2f}")
