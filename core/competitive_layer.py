import numpy as np
from .config import KWTAConfig, HomeostaticKWTAConfig
from .neuron import LIFLayer


class CompetitiveLIFLayer(LIFLayer):
    def __init__(
            self,
            num_inputs: int,
            num_neurons: int = 20,
            config: KWTAConfig | HomeostaticKWTAConfig | None = None
    ) -> None:
        self.kwta_config = config or HomeostaticKWTAConfig()
        super().__init__(num_inputs, num_neurons, self.kwta_config)

        self._homeostatic_kwta = self._homeostatic
        self._homeostatic = False

        self.window_spike_counts: np.ndarray = np.zeros(num_neurons, dtype=np.int32)
        self.last_winners: np.ndarray = np.array([], dtype=np.int32)

        # Tracks how many steps passed in the CURRENT global window
        self._current_window_size: int = 0
        self._phase_reset_pending: bool = False

    def trigger_phase_reset(self) -> None:
        """Called by NetworkGraph when the global oscillator completes a cycle."""
        self._phase_reset_pending = True

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        # POPRAWKA Bug C: Proaktywna inhibicja PRZED detekcją spike'ów.
        # Neurony, które strzeliły za dużo w bieżącym oknie, dostają karę odejmowaną
        # od v *zanim* zdążą wyemitować kolejny impuls. Dzięki temu k-WTA działa
        # prewencyjnie, a nie jako post-hoc korekta po fakcie.
        self._apply_proactive_inhibition()

        spikes = super().forward(pre_spikes)

        self.window_spike_counts += spikes.astype(np.int32)
        self._current_window_size += 1

        if self._phase_reset_pending:
            self._apply_lateral_inhibition()

            if self._homeostatic_kwta and self._current_window_size > 0:
                self._update_kwta_homeostasis(self._current_window_size)

            self._reset_window()

        return spikes

    def _apply_proactive_inhibition(self) -> None:
        """
        Ciągła inhibicja boczna aplikowana PRZED detekcją spike'ów.

        Dla każdego neuronu oblicza nadwyżkę aktywności ponad jego oczekiwany
        udział (total_spikes * k / N). Neurony z nadwyżką dostają ujemny prąd
        proporcjonalny do tej nadwyżki, co zmniejsza ich szansę na wystrzelenie
        w bieżącym kroku. Neurony poniżej progu k-WTA nie są dotknięte.

        Biologiczny odpowiednik: toniczna inhibicja z interneuronów GABAergicznych,
        które śledzą historię aktywności i aktywnie tłumią „nadaktywnych" sąsiadów.
        """
        if self._current_window_size == 0 or self.kwta_config.k_winners >= self.num_neurons:
            return
        total = int(np.sum(self.window_spike_counts))
        if total == 0:
            return
        # Oczekiwany udział dla k zwycięzców (proporcja k/N całkowitej aktywności)
        expected_per_neuron = total * self.kwta_config.k_winners / self.num_neurons
        excess = np.maximum(0.0, self.window_spike_counts.astype(np.float32) - expected_per_neuron)
        # Inhibicja skalowana przez okno, by siła na krok była stała niezależnie od długości okna
        inhibition = excess * (self.kwta_config.i_inh / self._current_window_size)
        self.v -= inhibition

    def _update_kwta_homeostasis(self, current_window_size: int) -> None:
        # (Keep the existing implementation of this method)
        cfg = self.kwta_config
        effective_spikes = self.window_spike_counts.astype(np.float32)
        if len(self.last_winners) > 0:
            losers_mask = np.ones(self.num_neurons, dtype=bool)
            losers_mask[self.last_winners] = False
            effective_spikes[losers_mask] = 0.0

        spikes_per_step = effective_spikes / current_window_size
        decay = np.exp(-(cfg.dt * current_window_size) / cfg.homeostatic_tau)

        self.avg_rate = (
                self.avg_rate * decay + spikes_per_step * (1.0 - decay)
        )

        rate_error = self.avg_rate - cfg.target_rate
        self.v_thresh_adaptive += (cfg.thresh_adapt_lr * current_window_size) * rate_error

        np.clip(
            self.v_thresh_adaptive,
            cfg.thresh_min,
            cfg.thresh_max,
            out=self.v_thresh_adaptive,
        )

    def _apply_lateral_inhibition(self) -> None:
        # (Keep the existing implementation of this method)
        if self.kwta_config.k_winners >= self.num_neurons:
            return

        if np.max(self.window_spike_counts) == 0:
            self.last_winners = np.array([], dtype=np.int32)
            return

        sorted_indices = np.argsort(self.window_spike_counts, kind='stable')
        winner_indices = sorted_indices[-self.kwta_config.k_winners:]

        self.last_winners = winner_indices

        losers_mask = np.ones(self.num_neurons, dtype=bool)
        losers_mask[winner_indices] = False
        no_spike_mask = self.window_spike_counts == 0
        losers_mask = losers_mask | no_spike_mask

        self.v[losers_mask] -= self.kwta_config.i_inh
        self.e[:, losers_mask] = 0.0
        self.x_post[losers_mask] = 0.0
        self.refrac_count[losers_mask] = 0

    def _reset_window(self) -> None:
        self.window_spike_counts.fill(0)
        self._current_window_size = 0
        self._phase_reset_pending = False

    def reset_state(self) -> None:
        super().reset_state()
        self._reset_window()
        self.last_winners = np.array([], dtype=np.int32)