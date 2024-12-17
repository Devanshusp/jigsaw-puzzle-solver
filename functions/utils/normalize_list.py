"""
normalize_list.py - Normalize a list of values to range [0, 1]
"""

import numpy as np


def normalize_list(values, reverse=False, factor=2.0, apply_non_linear=False):
    """
    Normalize values to range [0, 1].
    Reverse if higher is better.
    Apply non-linear scaling to emphasize "good" values if specified.

    Args:
    - values: List of values to normalize
    - reverse: If True, higher values are better
    - factor: Factor controlling the strength of non-linear scaling
    - apply_non_linear: If True, applies non-linear scaling

    Returns:
    - A list of normalized values
    """
    min_val = min(values)
    max_val = max(values)

    # Avoid division by zero if all values are the same
    if min_val == max_val:
        return [0 for _ in values]

    # Linear normalization
    norm_values = [(val - min_val) / (max_val - min_val) for val in values]

    if apply_non_linear:
        # Apply non-linear scaling to amplify higher values
        enhanced_values = [1 - np.exp(-factor * v) for v in norm_values]
    else:
        # If not applying non-linear scaling, return linear values
        enhanced_values = norm_values

    if reverse:
        return enhanced_values
    else:
        return [1 - v for v in enhanced_values]
