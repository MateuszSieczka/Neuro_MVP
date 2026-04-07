"""Quick diagnostic: what does the world model encoder actually produce?"""
import numpy as np
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena import task_config

task = task_config.get("MountainCar-v0")
np.random.seed(23)
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds, reward_scale=task.reward_scale)
state = env.reset(seed=23)
agent = make_agent(task, env)

enc = agent.world_model._encoder
print(f"Encoder: {enc.num_inputs} inputs, {enc.num_neurons} neurons, k={enc.kwta_config.k_winners}")
print(f"v_thresh={enc.config.v_thresh}, v_rest={enc.config.v_rest}, gap={enc.config.v_thresh - enc.config.v_rest}")
print(f"w shape={enc.w.shape}, w stats: mean={enc.w.mean():.4f}, std={enc.w.std():.4f}")

# Run a few steps and check encoder state
for step in range(10):
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    agent.observe(state, action, reward, next_state, done, info)
    
    v = enc.v
    spikes = enc.has_spiked
    rate = np.clip((v - enc.config.v_rest) / (enc.config.v_thresh - enc.config.v_rest), 0.0, 1.0)
    
    print(f"Step {step}: v=[{v.min():.1f}, {v.max():.1f}], "
          f"rate=[{rate.min():.3f}, {rate.max():.3f}], "
          f"spikes={spikes.sum()}, "
          f"phase_pending={enc._phase_reset_pending}, "
          f"window_size={enc._current_window_size}")

# Check threshold after some steps
if hasattr(enc, 'v_thresh_adaptive'):
    print(f"\nAdaptive thresh: [{enc.v_thresh_adaptive.min():.2f}, {enc.v_thresh_adaptive.max():.2f}]")
else:
    print("\nNo adaptive threshold")

# Check decoder
spikes_float = enc.has_spiked.astype(np.float32)
graded = np.clip((enc.v - enc.config.v_rest) / (enc.config.v_thresh - enc.config.v_rest), 0.0, 1.0)
pred_spike = spikes_float @ agent.world_model.w_decode
pred_graded = graded @ agent.world_model.w_decode
print(f"\nDecoder from spikes: {pred_spike}")
print(f"Decoder from graded: {pred_graded}")
print(f"Graded rate stats: mean={graded.mean():.4f}, max={graded.max():.4f}, nonzero={np.count_nonzero(graded)}")

env.close()
