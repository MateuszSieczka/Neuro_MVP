"""Quick diagnostic: actor current levels and manual 30 substep trace."""
import numpy as np
np.random.seed(42)
from arena.gym_env import GymEnv
from arena.agent_factory import make_agent
from arena.task_config import get as get_task
from core.spike_encoder import PoissonEncoder, GaussianPopulationEncoder

task = get_task("CartPole-v1")
env = GymEnv(task.env_id, normalize=True, fixed_bounds=task.obs_bounds,
             reward_scale=task.reward_scale)
state = env.reset(seed=42)
agent = make_agent(task, env)
actor = agent.actor
ncfg = actor._ncfg

print(f"Actor g_L = {ncfg.g_L:.4f}")
print(f"Actor C_m = {ncfg.C_m:.4f}")
print(f"Actor input_gain = {actor._input_gain:.4f}")
print(f"Actor state_size = {actor._state_size}")
print(f"n_substeps = {agent._n_substeps}")

pop_enc = agent._pop_encoder
poisson = agent._poisson
s = state.astype(np.float32)
rates = pop_enc.encode(s)
enc = poisson.encode(rates)
active = np.sum(enc > 0.5)
print(f"Pop rates: mean={rates.mean():.3f} active(>0.1)={int(np.sum(rates > 0.1))}")
print(f"Spikes: {int(active)}/{len(enc)}")
raw = enc @ actor.w_d1
scaled = raw * actor._input_gain
gap = abs(ncfg.v_thresh - ncfg.v_rest)
i_rheo = ncfg.g_L * (gap - ncfg.delta_t)
print(f"Raw current per action: {raw}")
print(f"Scaled (x gain): {scaled}")
print(f"Rheobase = {i_rheo:.1f} pA")
print(f"Ratio to rheobase: {scaled / i_rheo}")

# Manual 30-substep simulation
print(f"\n--- Manual 30-sub simulation (simple Euler) ---")
v = actor.v_d1.copy()
w_adapt = np.zeros_like(v)
print(f"Initial v_d1 = {v}")
g_L_eff = ncfg.C_m / 25.0  # up-state

for sub in range(30):
    enc2 = poisson.encode(rates)
    raw2 = enc2 @ actor.w_d1
    I = raw2 * actor._input_gain * 0.8  # approx d1_mod
    exp_term = np.exp(np.clip((v - ncfg.v_thresh) / ncfg.delta_t, -20, 10))
    leak = -g_L_eff * (v - ncfg.v_rest)
    exp_cur = g_L_eff * ncfg.delta_t * exp_term
    F = (1 / ncfg.C_m) * (leak + exp_cur + I - w_adapt)
    if sub % 5 == 0 or sub >= 25:
        print(f"  sub={sub:2d} V={np.array2string(v, precision=2)} "
              f"I_syn={np.array2string(I, precision=1)} F={np.array2string(F, precision=3)}")
    v_new = v + F * 1.0
    spiked = v_new >= ncfg.v_spike_cutoff
    if np.any(spiked):
        print(f"  SPIKE at sub={sub}! V_new={np.array2string(v_new, precision=2)}")
        v_new[spiked] = ncfg.v_reset
        w_adapt[spiked] += ncfg.b
    v = v_new

print(f"\nFinal v_d1 = {np.array2string(v, precision=2)}")
env.close()
