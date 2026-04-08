import numpy as np
np.random.seed(23)
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena import task_config

task = task_config.get('MountainCar-v0')
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
env.reset(seed=23)
agent = make_agent(task, env)

# Run 3 episodes, then probe
for ep in range(3):
    s = env.reset(); agent.reset()
    for st in range(200):
        a = agent.act(s)
        ns, r, d, info = env.step(a)
        agent.observe(s, a, r, ns, d, info)
        s = ns
        if d: break

# PROBE: check Active Inference values for different states
import sys
s = env.reset(); agent.reset()
# Run 10 steps to get into a state
for st in range(10):
    a = agent.act(s)
    ns, r, d, info = env.step(a)
    agent.observe(s, a, r, ns, d, info)
    s = ns

print('=== Active Inference Probe ===')
print(f'Current state (normalized): {s}')
print(f'Raw pos approx: {s[0]*0.9 + (-0.3):.4f}')

# Run mental rehearsal manually
results = agent.world_model.mental_rehearsal(s, [0, 1, 2])
for a in [0, 1, 2]:
    info = results[a]
    print(f'  Action {a}: novelty={info["novelty"]:.6f}, predicted_next={info["predicted_state"]}')

# Check decoder error for a known transition
pred_err = agent.world_model.prediction_error
print(f'Last decoder error: mean_abs={np.mean(np.abs(pred_err)):.6f}, values={pred_err}')
print(f'Last encoder PE: mean_abs={np.mean(np.abs(agent.world_model._encoder.prediction_error)):.6f}')
print(f'Curiosity signal: {agent.world_model.curiosity_signal():.6f}')
print(f'Encoder input size: {agent.world_model.input_size}')
print(f'Pop encoder: {agent.world_model._pop_encoder is not None}')
if agent.world_model._pop_encoder:
    test_encode = agent.world_model._pop_encoder.encode(s)
    print(f'Pop encoded shape: {test_encode.shape}, range: [{test_encode.min():.3f}, {test_encode.max():.3f}]')
    print(f'Pop encoded nonzero: {np.count_nonzero(test_encode > 0.01)} / {len(test_encode)}')

# Check BG actor state
print(f'Actor w_mu shape: {agent.bg.actor.w_mu.shape}')
print(f'Actor logits for current state:')
aug = agent._augment_state(s)
logits = np.dot(aug, agent.bg.actor.w_mu)
print(f'  Logits: {logits[:3]}')
probs = agent.bg.actor._softmax(logits[:3] / max(1.0 * agent.bg.actor.noise_scale, 1e-4))
print(f'  Probs (temp={1.0 * agent.bg.actor.noise_scale:.3f}): {probs}')
print(f'  noise_scale: {agent.bg.actor.noise_scale:.4f}')
