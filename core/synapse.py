"""
Synapse — conductance-based synaptic models (AMPA, NMDA, GABA-A, GABA-B).

Reference: Jahr & Stevens (1990), Destexhe et al. (1998)

Replaces the flat weight × spike model with biophysically grounded
conductance channels. Each synapse type has:
  - Conductance variable g that decays with a type-specific time constant
  - Rise triggered by presynaptic spike (instantaneous for simplicity)
  - Current: I_syn = g × (V - E_rev) for conductance-based
    or simplified current-based: I_syn = g (charge per spike)

For computational efficiency in large networks, we use the current-based
approximation where weights represent PSP amplitude (mV-equivalent),
but with proper temporal dynamics (dual-exponential or single-exponential
decay per channel type).

NMDA retains explicit voltage dependence via Mg²⁺ block factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SynapseConfig, NeuronConfig
from .simulation_context import SimulationContext, DEFAULT_CONTEXT


class SynapticChannels:
    """Manages AMPA/NMDA/GABA-A/GABA-B conductance traces for a layer.

    Each channel maintains a conductance vector (one per postsynaptic neuron)
    that decays exponentially and receives instantaneous increments on
    presynaptic spikes.

    Usage::

        channels = SynapticChannels(n_post=64, config=SynapseConfig())
        # Each timestep:
        channels.receive_spikes(pre_spikes, w_ampa, w_nmda)
        total_current = channels.compute_current(v_post)
    """

    def __init__(
        self,
        n_post: int,
        config: SynapseConfig | None = None,
    ) -> None:
        self.config = config or SynapseConfig()
        self.n_post = n_post

        # Decay-phase conductance traces per channel (n_post,)
        self.g_ampa: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_nmda: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_gaba_a: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_gaba_b: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)

        # Rise-phase traces (dual-exponential, Destexhe et al. 1998)
        # Effective conductance = g_decay - g_rise after normalisation.
        # Spikes increment both; rise decays faster → difference gives
        # the characteristic rise-then-decay synaptic waveform.
        self.g_ampa_rise: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_nmda_rise: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_gaba_a_rise: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_gaba_b_rise: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)

    def receive_excitatory(
        self,
        pre_spikes: NDArray[np.float32],
        w: NDArray[np.float32],
    ) -> None:
        """Process presynaptic excitatory spikes through AMPA + NMDA channels.

        AMPA/NMDA ratio from config (default 3:1, Myme et al. 2003).
        Spike increments both decay and rise traces by w × N (normalisation
        factor from Destexhe et al. 1998) so that peak conductance = w.

        Args:
            pre_spikes: (n_pre,) spike vector.
            w: (n_pre, n_post) weight matrix.
        """
        drive = pre_spikes.astype(np.float32) @ w  # (n_post,)
        cfg = self.config
        ratio = cfg.ampa_nmda_ratio
        total = ratio + 1.0
        ampa_drive = drive * (ratio / total)
        nmda_drive = drive * (1.0 / total)
        # Both decay and rise traces get the same norm-scaled increment
        self.g_ampa += ampa_drive * cfg.ampa_norm
        self.g_ampa_rise += ampa_drive * cfg.ampa_norm
        self.g_nmda += nmda_drive * cfg.nmda_norm
        self.g_nmda_rise += nmda_drive * cfg.nmda_norm

    def receive_inhibitory(
        self,
        inh_spikes: NDArray[np.float32],
        w_ie: NDArray[np.float32],
        gaba_b_ratio: float = 0.25,
    ) -> None:
        """Process inhibitory spikes through GABA-A + GABA-B channels.

        Args:
            inh_spikes: (n_inh,) inhibitory spike vector.
            w_ie: (n_inh, n_post) inhibitory weight matrix.
            gaba_b_ratio: Fraction of inhibition via GABA-B (Isaacson 2011).
        """
        drive = inh_spikes.astype(np.float32) @ w_ie  # (n_post,)
        cfg = self.config
        ga_drive = drive * (1.0 - gaba_b_ratio)
        gb_drive = drive * gaba_b_ratio
        self.g_gaba_a += ga_drive * cfg.gaba_a_norm
        self.g_gaba_a_rise += ga_drive * cfg.gaba_a_norm
        self.g_gaba_b += gb_drive * cfg.gaba_b_norm
        self.g_gaba_b_rise += gb_drive * cfg.gaba_b_norm

    def decay(self) -> None:
        """Apply exponential decay to all conductance channels.

        Both decay-phase and rise-phase traces decay independently.
        Effective conductance = g_decay - g_rise (computed in
        compute_current).
        """
        cfg = self.config
        self.g_ampa *= cfg.ampa_decay
        self.g_nmda *= cfg.nmda_decay
        self.g_gaba_a *= cfg.gaba_a_decay
        self.g_gaba_b *= cfg.gaba_b_decay
        self.g_ampa_rise *= cfg.ampa_rise_decay
        self.g_nmda_rise *= cfg.nmda_rise_decay
        self.g_gaba_a_rise *= cfg.gaba_a_rise_decay
        self.g_gaba_b_rise *= cfg.gaba_b_rise_decay

    def compute_current(
        self,
        v_post: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute total synaptic current from all channels (Ohm's law).

        Dual-exponential kinetics (Destexhe et al. 1998):
          g_eff = g_decay − g_rise  (difference gives rise-then-decay shape)
        Conductance-based current with reversal potentials:
          I_exc = (g_ampa_eff + g_nmda_eff × B(V)) × (E_exc − V)
          I_inh = (g_gaba_a_eff + g_gaba_b_eff) × (E_inh − V)

        Reference: Jahr & Stevens (1990), Destexhe et al. (1998)

        Args:
            v_post: (n_post,) postsynaptic membrane potential (mV).

        Returns:
            (n_post,) total synaptic current (positive = depolarizing).
        """
        cfg = self.config

        # Effective conductances: decay trace − rise trace (≥ 0 by construction)
        g_ampa_eff = np.maximum(self.g_ampa - self.g_ampa_rise, 0.0)
        g_nmda_eff = np.maximum(self.g_nmda - self.g_nmda_rise, 0.0)
        g_gaba_a_eff = np.maximum(self.g_gaba_a - self.g_gaba_a_rise, 0.0)
        g_gaba_b_eff = np.maximum(self.g_gaba_b - self.g_gaba_b_rise, 0.0)

        # NMDA voltage-dependent Mg²⁺ block (Jahr & Stevens 1990)
        mg_block = SynapseConfig.nmda_mg_block(v_post)

        # Excitatory: I = g × (E_exc − V) → positive at rest (depolarizing)
        g_exc = g_ampa_eff + g_nmda_eff * mg_block
        i_exc = g_exc * (cfg.e_exc - v_post)

        # Inhibitory: I = g × (E_inh − V) → negative at rest (hyperpolarizing)
        g_inh = g_gaba_a_eff + g_gaba_b_eff
        i_inh = g_inh * (cfg.e_inh - v_post)

        return i_exc + i_inh

    def reset(self) -> None:
        """Reset all conductance traces to zero."""
        self.g_ampa.fill(0.0)
        self.g_nmda.fill(0.0)
        self.g_gaba_a.fill(0.0)
        self.g_gaba_b.fill(0.0)
        self.g_ampa_rise.fill(0.0)
        self.g_nmda_rise.fill(0.0)
        self.g_gaba_a_rise.fill(0.0)
        self.g_gaba_b_rise.fill(0.0)
