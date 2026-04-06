import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, kw_only=True)
class ContinuousBGConfig:
    gamma: float = 0.99
    critic_lr: float = 7e-3   # Tempo uczenia Krytyka — szybkie z krótkim tau_e_critic=5
    actor_lr: float = 5e-3    # Aktor z policy gradient trace — wyższe lr z konsolidacją
    tau_e: float = 20.0       # Bazowe tau dla aktora (NMDA/Ca²⁺, ~1s)
    tau_e_critic: float = 5.0 # Krótsze tau dla krytyka — szybsza konwergencja V(s)
    tau_hidden: float = 2.0   # Stała czasowa membrany ukrytej — krótka dla szybkiej odpowiedzi
    dt: float = 1.0
    exploration_noise: float = 0.3  # Temperatura softmax (wyższa na starcie → eksploracja)
    hidden_size: int = 128  # Rozmiar warstwy ukrytej Krytyka
    tau_ne_compression: float = 4.0  # Max trace-tau compression factor at NE=1.0
    w_clip: float = 3.0      # Synaptic saturation: max |w| (receptor density / spine volume bound)


class SNNDeepCritic:
    """
    Głęboki Krytyk (Nieliniowy). Posiada jedną warstwę ukrytą LIF,
    aby poprawnie estymować funkcję wartości dla złożonych, nieliniowych
    przestrzeni (rozwiązanie problemu XOR/szachownicy).
    """

    def __init__(self, state_size: int, config: ContinuousBGConfig):
        self.config = config
        self._state_size = state_size

        # --- Xavier-like init: std = 1/sqrt(fan_in) ---
        # Zapewnia, że wariancja aktywacji jest ~1 niezależnie od state_size.
        h_std = 1.0 / np.sqrt(state_size)
        self.w_h = np.random.uniform(-h_std, h_std, (state_size, config.hidden_size)).astype(np.float32)

        v_std = 1.0 / np.sqrt(config.hidden_size)
        self.w_v = np.random.uniform(-v_std, v_std, config.hidden_size).astype(np.float32)

        # Ślady dla propagacji błędu
        self.e_h = np.zeros((state_size, config.hidden_size), dtype=np.float32)
        self.e_v = np.zeros(config.hidden_size, dtype=np.float32)
        self._trace_decay = np.exp(-self.config.dt / self.config.tau_e_critic)
        self._mem_decay: float = float(np.exp(-self.config.dt / self.config.tau_hidden))

        # Potencjał membrany neuronów ukrytych
        self.v_hidden = np.zeros(config.hidden_size, dtype=np.float32)
        # Ciągła aktywacja (rate-code)
        self.activation = np.zeros(config.hidden_size, dtype=np.float32)

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def _graded_activation(self, v: np.ndarray) -> np.ndarray:
        """Graded rate-code activation.

        tanh zapewnia bounded [-1,1] output. Xavier init + krótkie
        tau_hidden utrzymują |v| w zakresie liniowym tanh (~1–2)
        bez potrzeby ręcznego gain.
        """
        return np.tanh(v)

    # ------------------------------------------------------------------
    # Forward / Peek
    # ------------------------------------------------------------------

    def forward(self, state_spikes: np.ndarray) -> float:
        state_f32 = state_spikes.astype(np.float32)

        # 1. Integracja membrany (leaky integrator)
        self.v_hidden = self.v_hidden * self._mem_decay + np.dot(state_f32, self.w_h)

        # 2. Ciągła aktywacja z adaptacyjnym gain
        self.activation = self._graded_activation(self.v_hidden)

        # 3. Ślady kwalifikowalności (accumulating traces)
        # Akumulacja wzmacnia sygnał korelacyjny: e_ss ≈ act/(1-decay) ≈ tau_e × act.
        # Stabilność zapewnia w_clip (synaptic saturation).
        self.e_h = self.e_h * self._trace_decay + np.outer(state_f32, self.activation)
        self.e_v = self.e_v * self._trace_decay + self.activation
        # Ca²⁺ saturation — ślady ograniczone pojemnością kolca dendrytycznego.
        # Zapobiega akumulacji przy bardzo długich epizodach.
        np.clip(self.e_h, -2.0, 2.0, out=self.e_h)
        np.clip(self.e_v, -2.0, 2.0, out=self.e_v)

        return float(np.dot(self.w_v, self.activation))

    def peek(self, state_spikes: np.ndarray) -> float:
        """Estimate V(s') without modifying internal state (membrane, traces)."""
        state_f32 = state_spikes.astype(np.float32)
        v_hid = self.v_hidden * self._mem_decay + np.dot(state_f32, self.w_h)
        act = self._graded_activation(v_hid)
        return float(np.dot(self.w_v, act))

    def update(self, td_error: float) -> None:
        """Biologiczna reguła uczenia: δ × ślad kwalifikowalności.

        Warstwa wyjściowa (w_v): prosta reguła Hebba modulowana dopaminą:
            Δw_v = lr × δ × e_v

        Warstwa ukryta (w_h): TD error jest propagowany wstecz przez w_v,
        analogicznie do sygnału z jądra podwzgórzowego (STN) modulującego
        plastyczność korowo-prążkowiową.  Pochodna tanh' modeluje
        nieliniową odpowiedź dendrytyczną neuronu postsynaptycznego.

        Stabilność zapewniają:
        - bounded activation (tanh ∈ [-1,1]) → bounded traces
        - weight decay (homeostaza synaptyczna / synaptic scaling)
        - TD clipping (ograniczony wpływ ekstremalnych zdarzeń)
        """
        td_error = float(np.clip(td_error, -10.0, 10.0))

        # Warstwa wyjściowa: prosta reguła δ×e
        self.w_v += self.config.critic_lr * td_error * self.e_v

        # Warstwa ukryta: δ propagowany wstecz przez w_v z pochodną aktywacji.
        # tanh'(x) = 1 - tanh(x)² — modeluje nieliniową odpowiedź dendrytu.
        activation_deriv = 1.0 - self.activation ** 2
        feedback = self.w_v * activation_deriv   # (hidden,)
        self.w_h += self.config.critic_lr * td_error * self.e_h * feedback[np.newaxis, :]

        # Synaptic saturation: max weight bounded by receptor density / spine volume.
        wc = self.config.w_clip
        np.clip(self.w_v, -wc, wc, out=self.w_v)
        np.clip(self.w_h, -wc, wc, out=self.w_h)


    def reset_state(self) -> None:
        """Reset transient state (membrane, traces). Weights are preserved."""
        self.v_hidden.fill(0.0)
        self.activation.fill(0.0)
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
        eff_tau_e = self.config.tau_e_critic / ne_factor
        self._trace_decay = float(np.exp(-self.config.dt / eff_tau_e))


class SNNContinuousActor:
    """
    Aktor oparty na polityce softmax z mechanizmem dopaminowym.

    Biologicznie odpowiada ścieżce korowo-prążkowiowej:
    - Wagi w_mu: synapsy korowo-prążkowiowe (cortex → striatum)
    - Logity: aktywacja neuronów MSN (medium spiny neurons)
    - Softmax: kompetycja lateralna w prążkowiu (GPe/GPi)
    - Ślad kwalifikowalności: NMDA/Ca²⁺ ślad na aktywnej synapsie
    - TD error → Dopamina: moduluje plastyczność STDP

    Reguła uczenia (policy gradient z eligibility trace):
      e_t = decay * e_{t-1} + ∇_θ log π(a|s)
      Δw = lr × δ × e_t

    gdzie ∇ log π(a|s) = state ⊗ (one_hot(a) - π(·|s))
    Biologicznie: wybrany neuron MSN (D1) ma pozytywny ślad,
    niewybrany (D2) — negatywny proporcjonalnie do π.
    """

    def __init__(self, state_size: int, motor_dim: int, internal_dim: int, config: ContinuousBGConfig):
        self.config = config
        self.action_dim = motor_dim + internal_dim
        self.motor_dim = motor_dim

        # Umiarkowane losowe init — wystarczające do zróżnicowania logitów,
        # ale nie na tyle duże by stworzyć silne początkowe bias.
        # Biologicznie: niezorganizowane synapsy przed treningiem.
        self.w_mu = np.random.uniform(-0.1, 0.1, (state_size, self.action_dim)).astype(np.float32)

        # Ślad kwalifikowalności polityki (policy gradient trace)
        self.e_actor = np.zeros((state_size, self.action_dim), dtype=np.float32)
        self._trace_decay = np.exp(-self.config.dt / self.config.tau_e)

        # Temperatura eksploracji — mutowalny mnożnik:
        # maleje po sukcesach (serotonina wysoka), rośnie po porażkach.
        self.noise_scale: float = 1.0

        # Zapamiętujemy ostatnią politykę i akcję dla update
        self._last_probs: np.ndarray | None = None
        self._last_action: int = -1
        self._last_state: np.ndarray | None = None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerycznie stabilna softmax."""
        x = logits - logits.max()
        e = np.exp(x)
        return e / e.sum()

    def forward(self, state_spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Oblicza politykę i próbkuje akcję."""
        state_f32 = state_spikes.astype(np.float32)
        self._last_state = state_f32

        # Logity = stan × wagi  (aktywacja MSN)
        logits = np.dot(state_f32, self.w_mu)

        # Temperatura eksploracji: noise_scale > 1 → bardziej losowa;
        # noise_scale → 0 → greedy.  Odpowiada inwersji temperatury β = 1/T.
        temperature = max(self.config.exploration_noise * self.noise_scale, 1e-4)
        probs = self._softmax(logits[:self.motor_dim] / temperature)
        self._last_probs = probs

        # Próbkowanie akcji (ε-softmax)
        action = int(np.random.choice(self.motor_dim, p=probs))
        self._last_action = action

        # Ślad kwalifikowalności (policy gradient):
        # ∇ log π(a|s) = state ⊗ (one_hot(a) - π(·|s))
        # Biologicznie: D1 MSN wybranej akcji = +1,
        # wszystkie MSN hamowane proporcjonalnie do π (kompetycja GPe).
        grad_log_pi = np.zeros(self.action_dim, dtype=np.float32)
        one_hot = np.zeros(self.motor_dim, dtype=np.float32)
        one_hot[action] = 1.0
        grad_log_pi[:self.motor_dim] = one_hot - probs

        self.e_actor = self.e_actor * self._trace_decay + np.outer(state_f32, grad_log_pi)
        # Ca²⁺ saturation — trace per synapse is bounded by dendritic spine capacity
        np.clip(self.e_actor, -1.0, 1.0, out=self.e_actor)

        # Motor output jako ciągły wektor (kompatybilność z continuous API)
        motor_action = probs * 2.0 - 1.0   # map [0,1] → [-1,1]

        # Akcje wewnętrzne (bramka WM): drugi segment logitów
        if self.action_dim > self.motor_dim:
            internal_logits = logits[self.motor_dim:]
            internal_action = 1.0 / (1.0 + np.exp(-internal_logits))  # sigmoid → [0,1]
        else:
            internal_action = np.array([], dtype=np.float32)

        return motor_action, internal_action

    def update(self, td_error: float) -> None:
        """Policy gradient: δ × eligibility trace."""
        td_error = float(np.clip(td_error, -10.0, 10.0))

        self.w_mu += self.config.actor_lr * td_error * self.e_actor
        np.clip(self.w_mu, -self.config.w_clip, self.config.w_clip, out=self.w_mu)

    def get_action(self) -> int:
        """Zwraca ostatnio wybraną dyskretną akcję."""
        return self._last_action

    def reset_state(self) -> None:
        """Reset transient traces. Weights are preserved."""
        self.e_actor.fill(0.0)
        self._last_probs = None
        self._last_action = -1
        self._last_state = None

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