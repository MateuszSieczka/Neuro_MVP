"""
Columnar Architecture — cortical column factory for 2D/3D inputs.

Biological grounding:
  Neocortex is organized into ~150 000 cortical columns, each processing
  a local receptive field.  Columns within a modality share the same
  laminar circuit (L4 → L2/3 → L5/6) but operate on different spatial
  patches of the sensory sheet.

Architecture per column:
  col_i  (PredictiveCodingLayer)  — L4 feedforward encoding
  kwta_i (CompetitiveLIFLayer)    — L2/3 k-WTA competitive selection

All kwta_i outputs converge via concat aggregation into a shared
association layer (higher cortical area).

SpatialAttentionController modulates col_i gains using bottom-up
prediction error (saliency) and top-down association feedback.

Usage::

    from core.columnar import build_columnar_network, split_input

    net, col_names, kwta_names, assoc_name, attn = build_columnar_network(
        input_dim=64,
        receptive_field_size=16,
        neurons_per_column=8,
        assoc_neurons=32,
    )

    full_input = np.random.rand(64).astype(np.float32)
    sensory = split_input(full_input, col_names, 16)
    outputs = net.step(sensory, attention=attn)
"""

from __future__ import annotations

import numpy as np

from .attention import SpatialAttentionController
from .competitive_layer import CompetitiveLIFLayer
from .config import (
    AttentionConfig,
    CompetitiveConfig,
    PredictiveCodingConfig,
)
from .network import NetworkGraph
from .predictive_coding import PredictiveCodingLayer


def build_columnar_network(
    input_dim: int,
    receptive_field_size: int,
    neurons_per_column: int = 8,
    assoc_neurons: int = 32,
    column_config: PredictiveCodingConfig | None = None,
    comp_config: CompetitiveConfig | None = None,
    assoc_config: PredictiveCodingConfig | None = None,
    attn_config: AttentionConfig | None = None,
    net: NetworkGraph | None = None,
    column_prefix: str = "col",
    kwta_prefix: str = "kwta",
    assoc_name: str = "assoc",
) -> tuple[NetworkGraph, list[str], list[str], str, SpatialAttentionController]:
    """
    Build a columnar architecture with k-WTA readout and spatial attention.

    Architecture per column::

        col_i (PC layer: rf_size → neurons_per_column)
          ↓ feedforward
        kwta_i (CompetitiveLIFLayer: neurons_per_column → neurons_per_column)
          ↓ feedforward (concat)
        assoc (PC layer: n_cols × neurons_per_column → assoc_neurons)

    SpatialAttentionController targets col_i layers (bottom-up PE saliency).

    Args:
        input_dim:             Total flat input dimensionality.
        receptive_field_size:  Size of each column's receptive field.
                               input_dim must be divisible by this value.
        neurons_per_column:    Number of neurons per column / k-WTA layer.
        assoc_neurons:         Number of neurons in the association layer.
        column_config:         PredictiveCodingConfig for columns (shared).
        comp_config:           CompetitiveConfig for k-WTA layers (shared).
        assoc_config:          PredictiveCodingConfig for the association layer.
        attn_config:           AttentionConfig for spatial attention controller.
        net:                   Existing NetworkGraph to add to (or None to create new).
        column_prefix:         Prefix for column layer names ("col_0", "col_1", ...).
        kwta_prefix:           Prefix for k-WTA layer names ("kwta_0", "kwta_1", ...).
        assoc_name:            Name for the association layer.

    Returns:
        (net, column_names, kwta_names, assoc_name, attention)
    """
    if input_dim % receptive_field_size != 0:
        raise ValueError(
            f"input_dim ({input_dim}) must be divisible by "
            f"receptive_field_size ({receptive_field_size})."
        )

    n_columns = input_dim // receptive_field_size

    if net is None:
        net = NetworkGraph()

    if column_config is None:
        column_config = PredictiveCodingConfig()
    if comp_config is None:
        comp_config = CompetitiveConfig()
    if assoc_config is None:
        assoc_config = PredictiveCodingConfig()

    # ── Create columns: PC (L4) → k-WTA (L2/3) ──────────────────────
    column_names: list[str] = []
    kwta_names: list[str] = []
    total_kwta_outputs = 0

    for i in range(n_columns):
        col_name = f"{column_prefix}_{i}"
        kwta_name = f"{kwta_prefix}_{i}"

        col = PredictiveCodingLayer(
            num_inputs=receptive_field_size,
            num_neurons=neurons_per_column,
            pc_cfg=column_config,
        )
        net.add_layer(col_name, col)

        kwta = CompetitiveLIFLayer(
            num_inputs=neurons_per_column,
            num_neurons=neurons_per_column,
            comp_cfg=comp_config,
        )
        net.add_layer(kwta_name, kwta)
        net.connect(col_name, kwta_name, connection_type="feedforward")

        column_names.append(col_name)
        kwta_names.append(kwta_name)
        total_kwta_outputs += neurons_per_column

    # ── Association layer — receives concatenated k-WTA outputs ───────
    assoc = PredictiveCodingLayer(
        num_inputs=total_kwta_outputs,
        num_neurons=assoc_neurons,
        pc_cfg=assoc_config,
    )
    net.add_layer(assoc_name, assoc)

    for kwta_name in kwta_names:
        net.connect(
            source=kwta_name,
            target=assoc_name,
            connection_type="feedforward",
            aggregation_mode="concat",
        )

    # ── Spatial attention controller (targets PC columns) ─────────────
    attention = SpatialAttentionController(
        assoc_neurons=assoc_neurons,
        n_columns=n_columns,
        column_names=column_names,
        config=attn_config or AttentionConfig(),
        assoc_name=assoc_name,
    )

    return net, column_names, kwta_names, assoc_name, attention


def split_input(
    flat_input: np.ndarray,
    column_names: list[str],
    receptive_field_size: int,
) -> dict[str, np.ndarray]:
    """
    Split a flat input vector into per-column sensory dicts.

    Args:
        flat_input:            1D array of shape (input_dim,).
        column_names:          Column names from build_columnar_network.
        receptive_field_size:  Size of each column's receptive field.

    Returns:
        Dict mapping column name → receptive field slice.
    """
    sensory: dict[str, np.ndarray] = {}
    for i, name in enumerate(column_names):
        start = i * receptive_field_size
        end = start + receptive_field_size
        sensory[name] = flat_input[start:end].astype(np.float32)
    return sensory
