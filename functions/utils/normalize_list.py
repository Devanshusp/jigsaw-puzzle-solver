"""
normalize_list.py - Normalize a list of values to range [0, 1]
"""


def normalize_list(values, reverse=False):
    """Normalize values to range [0, 1]. Reverse if higher is better."""
    min_val = min(values)
    max_val = max(values)
    if reverse:
        return [(val - min_val) / (max_val - min_val) for val in values]
    else:
        return [(max_val - val) / (max_val - min_val) for val in values]
