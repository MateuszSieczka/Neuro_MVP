"""
Deep diagnostic: trace exact numbers through the SNN learning loop.

This script runs SingleButton for a few episodes and prints every
critical intermediate value to identify bottlenecks.
"""
import numpy as np
from arena.environments import SingleButtonEnv, TwoButtonEnv, PunishmentAvoidanceEnv
from arena.snn_agent import SNNAgent
from core.config import BasalGangliaConfig
from core.receptor import hill_response


def make_agent(env, use_wm=False, use_wmem=False):
    return SNNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        bg_config=BasalGangliaConfig(),
        use_world_model=use_wm,
        use_working_memory=use_wmem,
    )


def diagnose_single_button(n_episodes=300):
    """Trace internal values for SingleButton learning."""
    env = SingleButtonEnv()
    agent = make_agent(env)

    # Print architectural info
    bg = agent._bg_config
    print("=" * 70)
    print("ARCHITECTURAL ANALYSIS")
    print("=" * 70)
    print(f"State size: {agent.state_size}")
    print(f"Encoded size: {agent._encoded_size}")
    print(f"N actions: {agent.n_actions}")
    print(f"N substeps (actor): {agent._n_substeps}")
    print(f"N substeps (critic): {agent._n_substeps_critic}")
    print(f"Critic hidden: {bg.hidden_size}")
    print(f"Neurons per action: {bg.neurons_per_action}")
    print(f"Actor dim: {agent.actor.action_dim}")
    print(f"Critic input gain: {agent.critic._input_gain:.4f}")
    print(f"Actor input gain: {agent.actor._input_gain:.4f}")
    print(f"Actor D2 gain comp: {agent.actor._d2_gain_comp:.4f}")

    # D1/D2 modulation at baseline DA
    da = bg.baseline_da
    d1_resp = hill_response(da, bg.d1_ec50, bg.d1_hill_n)
    d1_mod = 1.0 + bg.d1_receptor_density * d1_resp
    d2_resp = hill_response(da, bg.d2_ec50, bg.d2_hill_n)
    d2_mod = 1.0 - bg.d2_receptor_density * d2_resp
    d2_tonic = 1.0 + bg.d2_tonic_boost_max * (1.0 - da)
    print(f"\nDA modulation at baseline DA={da}:")
    print(f"  D1: hill_resp={d1_resp:.4f}, d1_mod={d1_mod:.4f}")
    print(f"  D2: hill_resp={d2_resp:.4f}, d2_mod={d2_mod:.4f}, tonic={d2_tonic:.4f}")
    print(f"  D2 net modulation: {d2_mod * d2_tonic:.4f}")
    print(f"  D2 gain compensation: {agent.actor._d2_gain_comp:.4f}")
    print(f"  D2 effective after comp: {d2_mod * d2_tonic * agent.actor._d2_gain_comp:.4f}")

    # Weight norms
    print(f"\nInitial weight norms:")
    print(f"  Critic w_h: {np.linalg.norm(agent.critic.w_h):.4f}")
    print(f"  Actor w_d1: {np.linalg.norm(agent.actor.w_d1):.4f}")
    print(f"  Actor w_d2: {np.linalg.norm(agent.actor.w_d2):.4f}")
    print(f"  VTA w_value: {np.linalg.norm(agent.vta.w_value):.4f}")

    # Rheobase calculation
    ncfg_c = agent.critic._ncfg
    ncfg_a = agent.actor._ncfg
    gap_c = abs(ncfg_c.v_thresh - ncfg_c.v_rest)
    gap_a = abs(ncfg_a.v_thresh - ncfg_a.v_rest)
    i_rheo_c = ncfg_c.g_L * (gap_c - ncfg_c.delta_t)
    i_rheo_a = ncfg_a.g_L * (gap_a - ncfg_a.delta_t)
    print(f"\nBiophysics:")
    print(f"  Critic: g_L={ncfg_c.g_L:.2f}, tau_m={bg.tau_m_critic}, gap={gap_c:.2f}, I_rheo={i_rheo_c:.2f} pA")
    print(f"  Actor:  g_L={ncfg_a.g_L:.2f}, tau_m={bg.tau_m_msn_up}, gap={gap_a:.2f}, I_rheo={i_rheo_a:.2f} pA")
    print(f"  Noise std (mV): {bg.membrane_noise_std}")
    print(f"  Critic noise (pA): {ncfg_c.g_L * bg.membrane_noise_std:.2f}")
    print(f"  Actor noise (pA): {ncfg_a.g_L * bg.membrane_noise_std:.2f}")

    print("\n" + "=" * 70)
    print("EPISODE-BY-EPISODE DIAGNOSTICS")
    print("=" * 70)

    action_counts = [0, 0]
    rewards_history = []

    for ep in range(n_episodes):
        state = env.reset(seed=ep)
        agent.reset()

        # act()
        action = agent.act(state)
        action_counts[action] += 1

        next_state, reward, done, info = env.step(action)

        # Capture before observe
        critic_act = agent.critic.activation.copy()
        d1_spikes = agent.actor.spikes_d1.copy()
        d2_spikes = agent.actor.spikes_d2.copy()
        v_d1 = agent.actor.v_d1.copy()
        v_d2 = agent.actor.v_d2.copy()
        net_ev = agent.actor._last_net_evidence.copy() if agent.actor._last_net_evidence is not None else None
        vta_vs = agent.vta.last_v_s
        e_d1_norm = np.linalg.norm(agent.actor.e_d1)
        e_d2_norm = np.linalg.norm(agent.actor.e_d2)
        e_h_norm = np.linalg.norm(agent.critic.e_h)
        e_val = agent.vta.e_value.copy()

        # Save weights before observe for delta computation
        w_d1_before = agent.actor.w_d1.copy()
        w_d2_before = agent.actor.w_d2.copy()
        w_h_before = agent.critic.w_h.copy()
        w_val_before = agent.vta.w_value.copy()

        agent.observe(state, action, reward, next_state, done, info)

        # Weight deltas
        dw_d1 = np.linalg.norm(agent.actor.w_d1 - w_d1_before)
        dw_d2 = np.linalg.norm(agent.actor.w_d2 - w_d2_before)
        dw_h = np.linalg.norm(agent.critic.w_h - w_h_before)
        dw_val = np.linalg.norm(agent.vta.w_value - w_val_before)

        rewards_history.append(reward)

        if ep < 5 or ep % 50 == 0 or ep == n_episodes - 1:
            print(f"\n--- Episode {ep} ---")
            print(f"  Action: {action}, Reward: {reward:.1f}")
            print(f"  Critic: act_mean={np.mean(critic_act):.5f}, "
                  f"act_max={np.max(critic_act):.5f}, "
                  f"spikes={np.mean(agent.critic.spikes_hidden):.4f}")
            print(f"  D1: spikes={np.mean(d1_spikes):.4f}, "
                  f"v_mean={np.mean(v_d1):.2f}, "
                  f"v_range=[{np.min(v_d1):.2f}, {np.max(v_d1):.2f}]")
            print(f"  D2: spikes={np.mean(d2_spikes):.4f}, "
                  f"v_mean={np.mean(v_d2):.2f}, "
                  f"v_range=[{np.min(v_d2):.2f}, {np.max(v_d2):.2f}]")
            print(f"  Net evidence: {net_ev}")
            print(f"  Eligibility: ||e_d1||={e_d1_norm:.5f}, "
                  f"||e_d2||={e_d2_norm:.5f}, "
                  f"||e_h||={e_h_norm:.5f}, "
                  f"||e_val||={np.linalg.norm(e_val):.5f}")
            print(f"  VTA: V(s)={vta_vs:.5f}, "
                  f"RPE={agent.vta.last_rpe:.5f}, "
                  f"γ_eff={agent.vta.last_gamma_eff:.5f}, "
                  f"auto_rms={agent.vta._auto_rms:.5f}")
            print(f"  Δw: ||Δw_d1||={dw_d1:.6f}, ||Δw_d2||={dw_d2:.6f}, "
                  f"||Δw_h||={dw_h:.6f}, ||Δw_val||={dw_val:.6f}")
            print(f"  Weight norms: w_d1={np.linalg.norm(agent.actor.w_d1):.4f}, "
                  f"w_d2={np.linalg.norm(agent.actor.w_d2):.4f}, "
                  f"w_h={np.linalg.norm(agent.critic.w_h):.4f}, "
                  f"w_val={np.linalg.norm(agent.vta.w_value):.4f}")
            print(f"  Homeo rates: critic={np.mean(agent.critic._homeo_rate):.5f}, "
                  f"d1={np.mean(agent.actor._homeo_rate_d1):.5f}, "
                  f"d2={np.mean(agent.actor._homeo_rate_d2):.5f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    window = 50
    for start in range(0, n_episodes, window):
        end = min(start + window, n_episodes)
        mean_r = np.mean(rewards_history[start:end])
        print(f"  Episodes {start:3d}-{end:3d}: mean_reward={mean_r:.3f}")

    print(f"\nAction distribution: {action_counts}")
    print(f"Action 1 (press) fraction: {action_counts[1] / n_episodes:.3f}")


if __name__ == "__main__":
    diagnose_single_button(300)
