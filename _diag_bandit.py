"""Diagnose ShiftingBandit reversal failure."""
import numpy as np
from arena.environments import ShiftingBanditEnv
from arena.snn_agent import SNNAgent
from core.config import BasalGangliaConfig


def diagnose_shifting_bandit():
    env = ShiftingBanditEnv(shift_interval=200)
    agent = SNNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        bg_config=BasalGangliaConfig(),
        use_world_model=False,
        use_working_memory=False,
    )

    action_history = []
    reward_history = []
    phase_history = []

    for ep in range(600):
        state = env.reset(seed=ep)
        agent.reset()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.observe(state, action, reward, next_state, done, info)

        action_history.append(action)
        reward_history.append(reward)
        phase_history.append(info["phase"])

        if ep in [0, 50, 100, 150, 199, 200, 250, 300, 350, 399]:
            w_d1 = agent.actor.w_d1
            w_d2 = agent.actor.w_d2
            d1_norms = []
            d2_norms = []
            for a in range(3):
                start = a * agent.actor.n_per_action
                end = start + agent.actor.n_per_action
                d1_norms.append(float(np.linalg.norm(w_d1[:, start:end])))
                d2_norms.append(float(np.linalg.norm(w_d2[:, start:end])))

            ev = agent.actor._last_net_evidence

            phase = info["phase"]
            print(f"ep={ep:3d} ph={phase} a={action} r={reward:.0f} "
                  f"td={agent._last_td_error:.3f} "
                  f"D1gap01={d1_norms[0]-d1_norms[1]:.3f} "
                  f"ev=[{ev[0]:.4f},{ev[1]:.4f},{ev[2]:.4f}] "
                  f"NE={agent.neuromod.noradrenaline:.3f} "
                  f"DA={agent.neuromod.dopamine:.3f}")

    # Print action distribution per window
    print("\nAction distribution per 50-episode window:")
    for start in range(0, 600, 50):
        end = min(start + 50, 600)
        actions = action_history[start:end]
        rewards = reward_history[start:end]
        phases = phase_history[start:end]
        counts = [actions.count(a) for a in range(3)]
        mean_r = np.mean(rewards)
        print(f"  ep {start:3d}-{end:3d}: ph={phases[0]} "
              f"actions=[{counts[0]:2d},{counts[1]:2d},{counts[2]:2d}] "
              f"mean_r={mean_r:.3f}")


if __name__ == "__main__":
    diagnose_shifting_bandit()
