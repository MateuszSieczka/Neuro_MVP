"""
Columnar Architecture — cortical column factory for 2D/3D inputs.

Biological grounding:
  Neocortex is organized into ~150 000 cortical columns, each processing
  a local receptive field.  Columns within a modality share the same
  laminar circuit (L2/3 → L4 → L5/6) but operate on different spatial
  patches of the sensory sheet.  Higher areas bind column outputs via
  convergent feedforward projections (concat aggregation).

This module provides a factory function that:
  1. Splits a flat input vector into N non-overlapping receptive fields.
  2. Creates one PredictiveCodingLayer (column) per field.
  3. Creates a higher-level association layer whose num_inputs equals
     the sum of all column output widths.
  4. Registers everything in a NetworkGraph with 'concat' aggregation.

Usage::

    from core.columnar import build_columnar_network

    net, column_names, assoc_name = build_columnar_network(
        input_dim=64,          # total flat input size (e.g. 8×8 image)
        receptive_field_size=16,  # each column sees 16 inputs (4×4 patch)
        neurons_per_column=8,
        assoc_neurons=32,
    )

    # Feed sensory input split across columns
    full_input = np.random.rand(64).astype(np.float32)
    sensory = {}
    for i, name in enumerate(column_names):
        start = i * 16
        sensory[name] = full_input[start:start+16]
    outputs = net.step(sensory)
"""

from __future__ import annotations

import numpy as np

from .config import PredictiveCodingConfig
from .network import NetworkGraph
from .predictive_coding import PredictiveCodingLayer


def build_columnar_network(
    input_dim: int,
    receptive_field_size: int,
    neurons_per_column: int = 8,
    assoc_neurons: int = 32,
    column_config: PredictiveCodingConfig | None = None,
    assoc_config: PredictiveCodingConfig | None = None,
    net: NetworkGraph | None = None,
    column_prefix: str = "col",
    assoc_name: str = "assoc",
) -> tuple[NetworkGraph, list[str], str]:
    """
    Build a columnar architecture inside a NetworkGraph.

    Args:
        input_dim:             Total flat input dimensionality.
        receptive_field_size:  Size of each column's receptive field.
                               input_dim must be divisible by this value.
        neurons_per_column:    Number of neurons per column.
        assoc_neurons:         Number of neurons in the association layer.
        column_config:         PredictiveCodingConfig for columns (shared).
        assoc_config:          PredictiveCodingConfig for the association layer.
        net:                   Existing NetworkGraph to add to (or None to create new).
        column_prefix:         Prefix for column layer names ("col_0", "col_1", ...).
        assoc_name:            Name for the association layer.

    Returns:
        (net, column_names, assoc_name)
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
        column_config = PredictiveCodingConfig(
            k_winners=max(1, neurons_per_column // 2),
        )
    if assoc_config is None:
        assoc_config = PredictiveCodingConfig(
            k_winners=max(1, assoc_neurons // 4),
        )

    # Create columns
    column_names: list[str] = []
    total_column_outputs = 0
    for i in range(n_columns):
        name = f"{column_prefix}_{i}"
        col = PredictiveCodingLayer(
            num_inputs=receptive_field_size,
            num_neurons=neurons_per_column,
            pc_cfg=column_config,
        )
        net.add_layer(name, col)
        column_names.append(name)
        total_column_outputs += neurons_per_column

    # Create association layer — receives concatenated column outputs
    assoc = PredictiveCodingLayer(
        num_inputs=total_column_outputs,
        num_neurons=assoc_neurons,
        pc_cfg=assoc_config,
    )
    net.add_layer(assoc_name, assoc)

    # Connect all columns → association layer with concat aggregation
    for name in column_names:
        net.connect(
            source=name,
            target=assoc_name,
            connection_type="feedforward",
            aggregation_mode="concat",
        )

    return net, column_names, assoc_name


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
