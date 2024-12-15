"""
solve_border.py - This file contains the logic to solve the border of a puzzle.
"""

from typing import List

from functions.utils.save_data import get_data_for_piece


def solve_border(
    pieces_path: str,
    corner_piece_indices: List[int],
    edge_piece_indices: List[int],
    display_steps: bool = True,
):
    unsolved_piece_indices = set(corner_piece_indices + edge_piece_indices)

    # 1. Select a corner piece to begin with.
    selected_corner_piece_index = corner_piece_indices[0]
    unsolved_piece_indices.remove(selected_corner_piece_index)

    pass
