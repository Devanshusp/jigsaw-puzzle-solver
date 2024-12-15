"""
normalize_list.py - Normalize a list of values to range [0, 1]
"""


def normalize_list(values, reverse=False):
    """Normalize values to range [0, 1]. Reverse if higher is better."""
    min_val = min(values)
    max_val = max(values)

    # If all values are the same (min == max), return a list of zeros
    if min_val == max_val:
        return [0 for _ in values]

    if reverse:
        return [(val - min_val) / (max_val - min_val) for val in values]
    else:
        return [(max_val - val) / (max_val - min_val) for val in values]
