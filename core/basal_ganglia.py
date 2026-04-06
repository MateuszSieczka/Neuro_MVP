import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, kw_only=True)
class ContinuousBGConfig:
    gamma: float = 0.99
    critic_lr: float = 0.005
    actor_lr: float = 0.001
    tau_e: float = 200.0
    tau_hidden: float = 20.0  # Stała czasowa membrany warstwy ukrytej Krytyka (ms)
    dt: float = 1.0
    exploration_noise: float = 0.2  # Odchylenie standardowe szumu
    hidden_size: int = 128  # Rozmiar warstwy ukrytej Krytyka
    tau_ne_compression: float = 4.0  # Max trace-tau compression factor at NE=1.0


class SNNDeepCritic:
    """
    Głęboki Krytyk (Nieliniowy). Posiada jedną warstwę ukrytą LIF,
    aby poprawnie estymować funkcję wartości dla złożonych, nieliniowych
    przestrzeni (rozwiązanie problemu XOR/szachownicy).
    """

    def __init__(self, state_size: int, config: ContinuousBGConfig):
        self.config = config

        # Wagi Stan -> Warstwa Ukryta
        self.w_h = np.random.uniform(-0.1, 0.1, (state_size, config.hidden_size)).astype(np.float32)
        # Wagi Warstwa Ukryta -> Wartość V(s)
        self.w_v = np.random.uniform(-0.1, 0.1, config.hidden_size).astype(np.float32)

        # Ślady dla propagacji błędu
        self.e_h = np.zeros((state_size, config.hidden_size), dtype=np.float32)
        self.e_v = np.zeros(config.hidden_size, dtype=np.float32)
        self._trace_decay = np.exp(-self.config.dt / self.config.tau_e)
        # Poprawny biologiczny zanik membrany: exp(-dt/tau_hidden) zamiast hardcoded 0.8
        self._mem_decay: float = float(np.exp(-self.config.dt / self.config.tau_hidden))

        # Potencjał dla neuronów ukrytych (uproszczony LIF dla krytyka)
        self.v_hidden = np.zeros(config.hidden_size, dtype=np.float32)
        self.last_hidden_spikes = np.zeros(config.hidden_size, dtype=np.float32)
        self.hidden_firing_rate = np.zeros(config.hidden_size, dtype=np.float32)
        self.fr_decay = 0.9  # Stała wygładzania dla Krytyka

    def forward(self, state_spikes: np.ndarray) -> float:
        state_f32 = state_spikes.astype(np.float32)

        # 1. Integracja warstwy ukrytej
        self.v_hidden = self.v_hidden * self._mem_decay + np.dot(state_f32, self.w_h)
        spikes = (self.v_hidden > 0.5).astype(np.float32)
        self.v_hidden[spikes > 0] = 0.0

        # 2. WYGŁADZANIE (Zamiast surowych spike'ów do V(s))
        # To sprawia, że V(s) jest różniczkowalne i stabilne
        self.hidden_firing_rate = self.hidden_firing_rate * self.fr_decay + spikes * (1 - self.fr_decay)

        # 3. Ślady oparte na wygładzonej aktywności
        self.e_h = self.e_h * self._trace_decay + np.outer(state_f32, self.hidden_firing_rate)
        self.e_v = self.e_v * self._trace_decay + self.hidden_firing_rate

        return float(np.dot(self.w_v, self.hidden_firing_rate))

    def update(self, td_error: float) -> None:
        """Propagacja wsteczna błędu TD przez warstwy ukryte za pomocą śladów."""
        # Normalizacja śladów kwalifikowalności: zapobiega katastrofalnym
        # aktualizacjom, gdy ślady akumulują się przez długi epizod.
        # Biologiczny odpowiednik: ograniczona pojemność znaczników synaptycznych.
        e_v_scale = max(np.linalg.norm(self.e_v), 1.0)
        e_h_scale = max(np.linalg.norm(self.e_h), 1.0)

        # Update warstwy wyjściowej
        dw_v = self.config.critic_lr * td_error * (self.e_v / e_v_scale)

        # Odsprzęgnięty backprop do warstwy ukrytej (przybliżenie liniowe)
        backward_error = td_error * self.w_v
        dw_h = self.config.critic_lr * (self.e_h / e_h_scale) * backward_error[np.newaxis, :]

        self.w_v += dw_v
        self.w_h += dw_h
        np.clip(self.w_v, -10.0, 10.0, out=self.w_v)
        np.clip(self.w_h, -10.0, 10.0, out=self.w_h)


    def reset_state(self) -> None:
        """Reset transient state (membrane, traces). Weights are preserved."""
        self.v_hidden.fill(0.0)
        self.last_hidden_spikes.fill(0.0)
        self.hidden_firing_rate.fill(0.0)
        self.e_h.fill(0.0)
        self.e_v.fill(0.0)

    def set_plasticity_timescales(self, ne: float) -> None:
        """
        Dynamicznie dostosowuje stałe czasowe śladów kwalifikowalności
        w odpowiedzi na poziom noradrenaliny.

        Wysoka NE (wykrycie zmiany kontekstu) → kompresja tau_e →
        stare korelacje zanikają szybciej → szybsze odłączenie od
        poprzedniej polityki.

        Niska NE (stabilne środowisko) → pełne tau_e → konsolidacja.
        """
        ne = float(np.clip(ne, 0.0, 1.0))
        ne_factor = 1.0 + ne * (self.config.tau_ne_compression - 1.0)
        eff_tau_e = self.config.tau_e / ne_factor
        self._trace_decay = float(np.exp(-self.config.dt / eff_tau_e))


class SNNContinuousActor:
    """
    Ciągły Aktor. Generuje wektor akcji [motor_actions + internal_actions].
    Uczy się poprzez korelację szumu eksploracyjnego z błędem TD.
    """

    def __init__(self, state_size: int, motor_dim: int, internal_dim: int, config: ContinuousBGConfig):
        self.config = config
        self.action_dim = motor_dim + internal_dim
        self.motor_dim = motor_dim

        # Wagi mapujące stan na średnią akcji (mu)
        self.w_mu = np.random.uniform(-0.01, 0.01, (state_size, self.action_dim)).astype(np.float32)

        # Ślad STDP korelujący stan z dodanym szumem (Policy Gradient Trace)
        self.e_actor = np.zeros((state_size, self.action_dim), dtype=np.float32)
        self._trace_decay = np.exp(-self.config.dt / self.config.tau_e)

    def forward(self, state_spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Zwraca akcje motoryczne (dla robota) i wewnętrzne (dla pamięci)."""
        state_f32 = state_spikes.astype(np.float32)

        # Generowanie deterministycznej średniej (mu)
        mu = np.dot(state_f32, self.w_mu)

        # Eksploracja (Szum Gaussa)
        noise = np.random.normal(0, self.config.exploration_noise, self.action_dim)
        raw_action = mu + noise

        # Ograniczenie przestrzeni ciągłej (np. do tanh dla robota [-1, 1])
        bounded_action = np.tanh(raw_action)

        # Ślad kwalifikowalności: koreluje aktywny stan z ZASTOSOWANYM SZUMEM.
        # Jeśli szum na plusie dał nagrodę, wagi powinny wzrosnąć, by mu poszło w stronę szumu.
        self.e_actor *= self._trace_decay
        self.e_actor += np.outer(state_f32, noise)

        # Rozdzielenie akcji
        motor_action = bounded_action[:self.motor_dim]
        # Przeskalowanie akcji wewnętrznych (bramka WM) do [0, 1]
        internal_action = (bounded_action[self.motor_dim:] + 1.0) / 2.0

        return motor_action, internal_action

    def update(self, td_error: float) -> None:
        """Ciągłe STDP: Przesuwa średnią akcji w stronę udanej eksploracji."""
        dw = self.config.actor_lr * td_error * self.e_actor
        self.w_mu += dw
        np.clip(self.w_mu, -5.0, 5.0, out=self.w_mu)


    def reset_state(self) -> None:
        """Reset transient traces. Weights are preserved."""
        self.e_actor.fill(0.0)

    def set_plasticity_timescales(self, ne: float) -> None:
        """
        Dostosowuje zanik śladu polityki do poziomu NE.
        Wysoka NE → krótsze okno korelacji szum→nagroda → szybsze przełączanie polityki.
        """
        ne = float(np.clip(ne, 0.0, 1.0))
        ne_factor = 1.0 + ne * (self.config.tau_ne_compression - 1.0)
        eff_tau_e = self.config.tau_e / ne_factor
        self._trace_decay = float(np.exp(-self.config.dt / eff_tau_e))


class BasalGangliaAGISystem:
    def __init__(self, state_size: int, motor_dim: int, internal_dim: int = 1,
                 config: ContinuousBGConfig | None = None):
        self.config = config or ContinuousBGConfig()
        self.critic = SNNDeepCritic(state_size, self.config)
        self.actor = SNNContinuousActor(state_size, motor_dim, internal_dim, self.config)
        self.last_v = 0.0

    def step(self, state_spikes: np.ndarray, reward: float, is_terminal: bool = False) -> Tuple[
        np.ndarray, np.ndarray, float]:
        current_v = self.critic.forward(state_spikes)

        # TD Error
        if is_terminal:
            td_error = reward - self.last_v
        else:
            td_error = reward + self.config.gamma * current_v - self.last_v

        # Aktualizacja
        self.critic.update(td_error)
        self.actor.update(td_error)

        # Akcja na kolejny krok
        motor_action, internal_action = self.actor.forward(state_spikes)
        self.last_v = 0.0 if is_terminal else current_v

        return motor_action, internal_action, td_error

    def reset_state(self) -> None:
        """Reset all transient state between episodes. Learned weights are preserved."""
        self.last_v = 0.0
        self.critic.reset_state()
        self.actor.reset_state()

    def set_plasticity_timescales(self, ne: float) -> None:
        """
        Propaguje poziom NE do Krytyka i Aktora — zamknięta pętla:
          TD-error → NeuromodulatorSystem → NE → set_plasticity_timescales()
        Pozwala BG samodzielnie dostosować tempo uczenia do zmienności środowiska.
        """
        self.critic.set_plasticity_timescales(ne)
        self.actor.set_plasticity_timescales(ne)