"""Test wydajności próbkowania (Sample Efficiency) na środowisku CartPole."""
import numpy as np
from arena.gym_env import GymEnv
from arena.snn_agent import SNNAgent
from arena.core import Trainer
from core.basal_ganglia import ContinuousBGConfig
from core.config import SNNWorldModelConfig, NeuromodulatorConfig

BOUNDS = (np.array([-2.4, -3.0, -0.21, -3.0]), np.array([2.4, 3.0, 0.21, 3.0]))
SEEDS = [1, 17, 42, 99, 256]

def run_spark_benchmark():
    # 1. Agresywna konfiguracja Jąder Podstawnych
    bg_cfg = ContinuousBGConfig(
        gamma=0.98,
        critic_lr=0.015,        # Zmniejszone do stabilnego poziomu
        actor_lr=0.005,         # Znacznie stabilniejszy aktor (nie zniszczy wag po sukcesie)
        exploration_noise=0.15, # Wystarczający do nauki, na tyle mały by eksploatować cel
        hidden_size=64
    )

    # 2. Konfiguracja Modelu Świata i Neuromodulacji
    wm_cfg = SNNWorldModelConfig(
        hidden_size=64,
        k_winners=4,
        rehearsal_steps=5       # "Wyobraźnia" napędzająca k-WTA
    )
    
    nm_cfg = NeuromodulatorConfig() # Używamy domyślnych, zbalansowanych wartości

    n_ep = 120 # Skracamy uczenie ze 800 do 120 epizodów!
    scores = []

    print(f"--- Benchmark zbieżności dla CartPole (Cel: <= 80-100 epizodów) ---")

    for seed in SEEDS:
        np.random.seed(seed)
        env = GymEnv("CartPole-v1", normalize=True, fixed_bounds=BOUNDS)
        
        agent = SNNAgent(
            state_size=env.state_size, 
            n_actions=env.n_actions,
            bg_config=bg_cfg,
            use_world_model=False,  # Wyłączamy pożeracz CPU (nie jest podpięty pod aktora)
            trace_decay=0.0,        # Środowisko dostarcza prędkości, nie chcemy rozmycia
        )
        
        trainer = Trainer(env, agent)
        result = trainer.train(n_episodes=n_ep, max_steps=500)
        
        # Oceniamy na podstawie ostatnich 20 epizodów z uciętego przebiegu
        final_score = result.mean_reward(last_n=20)
        scores.append(final_score)
        
        # Opcjonalnie: możemy też sprawdzić, w którym epizodzie agent pierwszy raz wbił max
        first_solved = next((i for i, log in enumerate(result.episode_logs) if log.total_reward >= 490), ">120")
        
        print(f"Seed {seed:3d} | Średni wynik (ost. 20 ep): {final_score:5.0f}/500 | Pierwszy sukces w ep: {first_solved}")
        env.close()

    solved = sum(1 for s in scores if s >= 450)
    print(f"\nPodsumowanie: Średnia nagród={np.mean(scores):.0f} | Zaliczone ziarna={solved}/{len(SEEDS)}")

if __name__ == "__main__":
    run_spark_benchmark()