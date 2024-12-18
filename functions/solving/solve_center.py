"""
solve_center.py - This file contains the logic to solve the center of a puzzle.
"""

import copy
from pprint import pprint
from typing import Any, Dict, List, Tuple

from functions.solving.match_pieces import match_pieces
from functions.solving.solve_border import get_piece_relative_side
from functions.utils.normalize_list import normalize_list
from functions.utils.visualize_solution import visualize_solution


def solve_center(
    pieces_path: str,
    solved_border_matrix: List[List[Tuple[str, Any]]],
    middle_piece_indices: List[int],
    display_steps: bool = True,
    save: bool = False,
    save_name: str | None = None,
    visualize_each_step: bool = False,
):
    """
    Solve the center pieces of the puzzle using the solved border matrix as a guide.

    Args:
        pieces_path (str): Path to the pieces data
        solved_border_matrix (List[List[Dict]]): Matrix of solved border pieces
        middle_piece_indices (List[int]): Indices of pieces in the center area
    """
    # Create a copy of the solution matrix to modify
    solution_matrix = copy.deepcopy(solved_border_matrix)
    unsolved_piece_indices = set(middle_piece_indices)

    # Find the start of center solving (first empty cell after border)
    start_row, start_col = find_center_start(solution_matrix)
    curr_row, curr_col = start_row, start_col

    # Loop until all center pieces are solved
    while unsolved_piece_indices:
        print(f"Solving for {curr_col}, {curr_row}")

        # Get reference pieces (top and left neighbors)
        top_neighbor = solution_matrix[curr_row - 1][curr_col] if curr_row > 0 else None
        left_neighbor = (
            solution_matrix[curr_row][curr_col - 1] if curr_col > 0 else None
        )

        # Prepare potential matching sides and piece indices
        potential_piece_indices = list(unsolved_piece_indices)
        top_matching_side = get_piece_relative_side(
            top_neighbor["piece_top_side"], "down"  # type: ignore
        )  # Bottom of the top neighbor
        left_matching_side = get_piece_relative_side(
            left_neighbor["piece_top_side"], "right"  # type: ignore
        )  # Right of the left neighbor

        potential_matches = [
            (piece_index, side)
            for piece_index in potential_piece_indices
            for side in ["A", "B", "C", "D"]
        ]

        print(f"Potential matches: {potential_matches}")

        # No more potential matches found
        if not potential_matches and len(unsolved_piece_indices) > 0:
            break

        # Calculate match scores and errors
        match_error = calculate_match_error(
            pieces_path,
            top_neighbor["piece_index"],  # type: ignore
            top_matching_side,
            left_neighbor["piece_index"],  # type: ignore
            left_matching_side,
            potential_matches,
            visualize_each_step=visualize_each_step,
        )

        # Select best match
        best_match = match_error[0][0]
        best_match_index, best_match_side = best_match

        # Add best match to solution matrix
        solution_matrix[curr_row][curr_col] = {  # type: ignore
            "piece_index": best_match_index,
            "piece_top_side": best_match_side,
        }

        # Remove matched piece from unsolved pieces
        unsolved_piece_indices.remove(best_match_index)

        # Visualize solution
        if visualize_each_step:
            visualize_solution(
                pieces_path, solution_matrix, save=False, display_steps=True  # type: ignore
            )

        # Move to next cell
        curr_col, curr_row = move_to_next_cell(solution_matrix, curr_col, curr_row)  # type: ignore

    # Final visualization
    visualize_solution(
        pieces_path,
        solution_matrix,  # type: ignore
        display_steps=display_steps,
        save_name=save_name,
        save=save,
    )

    return solution_matrix


def find_center_start(solution_matrix: List[List[Dict[str, Any]]]) -> Tuple[int, int]:
    """Find the starting cell for solving center pieces."""
    for row in range(len(solution_matrix)):
        for col in range(len(solution_matrix[row])):
            if solution_matrix[row][col] is None:
                return row, col
    raise ValueError("No empty cell found in solution matrix")


def calculate_match_error(
    pieces_path: str,
    top_piece_index: int,
    top_piece_match_side: str,
    left_piece_index: int,
    left_piece_match_side: str,
    potential_matches: List[Tuple[int, str]],
    visualize_each_step: bool = False,
) -> List[Tuple[Tuple[int, str], float]]:
    """Calculate match errors for potential pieces."""
    hausdorff_distance = []
    hu_moments_distance = []
    corners_distance = []
    color_difference = []

    print("Match scores:")
    for potential_match in potential_matches:
        potential_match_index, potential_match_top_side = potential_match

        potential_match_left_side = get_piece_relative_side(
            potential_match_top_side, "left"  # type: ignore
        )

        # Skip if no neighboring piece to match against
        match_top_score, match_left_score = None, None

        match_top_score = match_pieces(
            pieces_path,
            top_piece_index,
            potential_match_index,
            top_piece_match_side,  # type: ignore
            potential_match_top_side,  # type: ignore
            display_steps=visualize_each_step,
        )

        match_left_score = match_pieces(
            pieces_path,
            left_piece_index,
            potential_match_index,
            left_piece_match_side,  # type: ignore
            potential_match_left_side,  # type: ignore
            display_steps=visualize_each_step,
        )

        total_match_score = {}
        for key in set(match_top_score) | set(match_left_score):
            total_match_score[key] = match_top_score.get(key, 0) + match_left_score.get(
                key, 0
            )

        print(potential_match)
        pprint(total_match_score)
        hausdorff_distance.append(total_match_score["Hausdorff Distance"])
        hu_moments_distance.append(total_match_score["Hu Moments Distance"])
        corners_distance.append(total_match_score["Corners Distance"])
        color_difference.append(total_match_score["Color Difference"])

    # Normalize match scores
    normalized_hausdorff_distance = normalize_list(hausdorff_distance)
    normalized_hu_moments_distance = normalize_list(hu_moments_distance)
    normalized_corners_distance = normalize_list(corners_distance)
    normalized_color_difference = normalize_list(color_difference)

    # Weights for match error
    hausdorff_weight = 2
    hu_moments_weight = 0
    corner_dist_weight = 1
    color_weight = 0
    sum_weights = (
        hausdorff_weight + hu_moments_weight + corner_dist_weight + color_weight
    )

    print("Match sides shown are always the potential 'top' side")

    # Calculate match error
    match_error = [
        (
            match,
            hausdorff_weight * hausdorff
            + hu_moments_weight * hu_moments
            + corner_dist_weight * corner_dist
            + color_weight * color,
        )
        for match, hausdorff, hu_moments, corner_dist, color in zip(
            potential_matches,
            normalized_hausdorff_distance,
            normalized_hu_moments_distance,
            normalized_corners_distance,
            normalized_color_difference,
        )
    ]

    # Sort potential matches by match score
    match_error.sort(key=lambda x: x[1], reverse=True)

    return match_error


def move_to_next_cell(
    solution_matrix: List[List[Dict[str, Any] | None]], curr_col: int, curr_row: int
) -> Tuple[int, int] | None:
    """Move to the next empty cell in the solution matrix."""
    # First try to move right
    if (
        curr_col + 1 < len(solution_matrix[curr_row])
        and solution_matrix[curr_row][curr_col + 1] is None
    ):
        return curr_col + 1, curr_row

    # If right is filled, move to next row
    for row in range(curr_row, len(solution_matrix)):
        for col in range(len(solution_matrix[row])):
            if solution_matrix[row][col] is None:
                return col, row

    return -1, -1
