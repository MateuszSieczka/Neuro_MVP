"""Test wydajności próbkowania (Sample Efficiency) na środowisku CartPole."""
import numpy as np
from arena.gym_env import GymEnv
from arena.snn_agent import SNNAgent
from arena.core import Trainer
from core.basal_ganglia import ContinuousBGConfig
from core.config import SNNWorldModelConfig, NeuromodulatorConfig

BOUNDS = (np.array([-2.4, -3.0, -0.21, -3.0]), np.array([2.4, 3.0, 0.21, 3.0]))
SEEDS = [1, 17, 42, 99, 145, 256, 500]

def run_spark_benchmark():
    # 1. Konfiguracja Jąder Podstawnych — domyślne wartości z ContinuousBGConfig
    bg_cfg = ContinuousBGConfig(
        gamma=0.95,
        exploration_noise=0.2,
        hidden_size=128,
    )

    # 2. Konfiguracja Modelu Świata i Neuromodulacji
    wm_cfg = SNNWorldModelConfig(
        hidden_size=64,
        k_winners=4,
        rehearsal_steps=5       # "Wyobraźnia" napędzająca k-WTA
    )
    
    nm_cfg = NeuromodulatorConfig() # Używamy domyślnych, zbalansowanych wartości

    n_ep = 200
    scores = []

    print(f"--- Benchmark zbieżności dla CartPole (Cel: <= 80-100 epizodów) ---")

    for seed in SEEDS:
        np.random.seed(seed)
        env = GymEnv("CartPole-v1", normalize=True, fixed_bounds=BOUNDS)
        # Seed the gymnasium env for reproducibility
        env.reset(seed=seed)

        agent = SNNAgent(
            state_size=env.state_size,
            n_actions=env.n_actions,
            bg_config=bg_cfg,
            use_world_model=False,
            trace_decay=0.0,        # Środowisko dostarcza prędkości, nie chcemy rozmycia
        )
        
        trainer = Trainer(env, agent)
        result = trainer.train(n_episodes=n_ep, max_steps=500)
        
        # Oceniamy na podstawie ostatnich 20 epizodów z uciętego przebiegu
        final_score = result.mean_reward(last_n=20)
        scores.append(final_score)
        
        # Opcjonalnie: możemy też sprawdzić, w którym epizodzie agent pierwszy raz wbił max
        first_solved = next((i for i, log in enumerate(result.episode_logs) if log.total_reward >= 490), f">{n_ep}")
        
        print(f"Seed {seed:3d} | Średni wynik (ost. 20 ep): {final_score:5.0f}/500 | Pierwszy sukces w ep: {first_solved}")
        env.close()

    solved = sum(1 for s in scores if s >= 450)
    print(f"\nPodsumowanie: Średnia nagród={np.mean(scores):.0f} | Zaliczone ziarna={solved}/{len(SEEDS)}")

if __name__ == "__main__":
    run_spark_benchmark()