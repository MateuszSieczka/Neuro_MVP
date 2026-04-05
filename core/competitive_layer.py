import numpy as np
from .config import KWTAConfig
from .neuron import LIFLayer


class CompetitiveLIFLayer(LIFLayer):
    """
    Extends LIFLayer with windowed k-Winners-Take-All (k-WTA) lateral inhibition
    to enforce Sparse Distributed Representations (SDR).
    """

    def __init__(self, num_inputs: int, num_neurons: int = 20, config: KWTAConfig | None = None) -> None:
        self.kwta_config = config or KWTAConfig()
        super().__init__(num_inputs, num_neurons, self.kwta_config)

        # Bufor zliczający impulsy w aktualnym oknie czasowym
        self.window_spike_counts: np.ndarray = np.zeros(num_neurons, dtype=np.int32)
        self.current_step: int = 0
        self.last_winners: np.ndarray = np.array([], dtype=np.int32)

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        Executes standard LIF integration, tracks spikes, and applies
        lateral inhibition at the end of the evaluation window.
        """
        # 1. Standardowa integracja błony
        spikes = super().forward(pre_spikes)

        # 2. Akumulacja impulsów
        self.window_spike_counts += spikes.astype(np.int32)
        self.current_step += 1

        # 3. Ewaluacja k-WTA na koniec okna
        if self.current_step >= self.kwta_config.window_ms:
            self._apply_lateral_inhibition()
            self._reset_window()

        return spikes

    def _apply_lateral_inhibition(self) -> None:
        """
        Identifies exact top k neurons. Applies hyperpolarizing penalty to losers
        and resets their eligibility traces to block false learning.
        Handles spike count ties deterministically using stable argsort.
        """
        if self.kwta_config.k_winners >= self.num_neurons:
            return

        if np.max(self.window_spike_counts) == 0:
            # Clear last_winners on silent window to prevent ghost credit assignment
            self.last_winners = np.array([], dtype=np.int32)
            return

        # 1. Strict and STABLE k-WTA
        sorted_indices = np.argsort(self.window_spike_counts, kind='stable')
        winner_indices = sorted_indices[-self.kwta_config.k_winners:]

        self.last_winners = winner_indices

        # 2. Tworzymy maskę przegranych
        losers_mask = np.ones(self.num_neurons, dtype=bool)
        losers_mask[winner_indices] = False

        # 3. Zabezpieczenie: Neuron nie może wygrać, jeśli nie strzelił ani razu
        no_spike_mask = self.window_spike_counts == 0
        losers_mask = losers_mask | no_spike_mask

        # 4. Egzekucja kary (hiperpolaryzacja)
        self.v[losers_mask] -= self.kwta_config.i_inh

        # 5. Całkowite wymazanie chemicznej pamięci (blokada fałszywego STDP)
        self.e[:, losers_mask] = 0.0
        self.x_post[losers_mask] = 0.0

        # 6. Zabezpieczenie fizyki błony przed "tarczą refrakcyjną"
        self.refrac_count[losers_mask] = 0

    def _reset_window(self) -> None:
        """Resets the spike counter for the next window."""
        self.window_spike_counts.fill(0)
        self.current_step = 0

    def reset_state(self) -> None:
        """Fully resets layer state including the k-WTA window."""
        super().reset_state()
        self._reset_window()
        self.last_winners = np.array([], dtype=np.int32)