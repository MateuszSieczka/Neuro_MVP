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

        # Conductance trace per channel (n_post,)
        self.g_ampa: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_nmda: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_gaba_a: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)
        self.g_gaba_b: NDArray[np.float32] = np.zeros(n_post, dtype=np.float32)

    def receive_excitatory(
        self,
        pre_spikes: NDArray[np.float32],
        w: NDArray[np.float32],
    ) -> None:
        """Process presynaptic excitatory spikes through AMPA + NMDA channels.

        AMPA/NMDA ratio from config (default 3:1, Myme et al. 2003).
        Spike → instantaneous conductance increment proportional to weight.

        Args:
            pre_spikes: (n_pre,) spike vector.
            w: (n_pre, n_post) weight matrix.
        """
        # Total excitatory drive
        drive = pre_spikes.astype(np.float32) @ w  # (n_post,)
        ratio = self.config.ampa_nmda_ratio
        total = ratio + 1.0
        self.g_ampa += drive * (ratio / total)
        self.g_nmda += drive * (1.0 / total)

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
        self.g_gaba_a += drive * (1.0 - gaba_b_ratio)
        self.g_gaba_b += drive * gaba_b_ratio

    def decay(self) -> None:
        """Apply exponential decay to all conductance channels."""
        self.g_ampa *= self.config.ampa_decay
        self.g_nmda *= self.config.nmda_decay
        self.g_gaba_a *= self.config.gaba_a_decay
        self.g_gaba_b *= self.config.gaba_b_decay

    def compute_current(
        self,
        v_post: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute total synaptic current from all channels (Ohm's law).

        Conductance-based model with reversal potentials:
          I_exc = (g_ampa + g_nmda × B(V)) × (V - E_exc)
          I_inh = (g_gaba_a + g_gaba_b) × (V - E_inh)
          I_total = I_exc + I_inh

        At V < E_exc (0 mV): excitatory current is negative (depolarizing
        in the convention dV/dt ~ -I, or depolarizing if we add I to V
        with the driving force providing correct sign).

        Reference: Jahr & Stevens (1990), Destexhe et al. (1998)

        Args:
            v_post: (n_post,) postsynaptic membrane potential (mV).

        Returns:
            (n_post,) total synaptic current (positive = depolarizing).
        """
        cfg = self.config

        # NMDA voltage-dependent Mg²⁺ block (Jahr & Stevens 1990)
        mg_block = SynapseConfig.nmda_mg_block(v_post)

        # Excitatory: driving force (V - E_exc), E_exc = 0 mV
        # At rest V=-70 mV → (V - 0) = -70 → g × (-70) is negative
        # We want depolarizing = positive, so negate: I = g × (E_exc - V)
        g_exc = self.g_ampa + self.g_nmda * mg_block
        i_exc = g_exc * (cfg.e_exc - v_post)

        # Inhibitory: driving force toward E_inh = -75 mV
        # At rest V=-70 mV → (E_inh - V) = -75 - (-70) = -5 → hyperpolarizing
        g_inh = self.g_gaba_a + self.g_gaba_b
        i_inh = g_inh * (cfg.e_inh - v_post)

        return i_exc + i_inh

    def reset(self) -> None:
        """Reset all conductance traces to zero."""
        self.g_ampa.fill(0.0)
        self.g_nmda.fill(0.0)
        self.g_gaba_a.fill(0.0)
        self.g_gaba_b.fill(0.0)
