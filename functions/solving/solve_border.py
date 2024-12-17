"""
solve_border.py - This file contains the logic to solve the border of a puzzle.
"""

import copy
from pprint import pprint
from typing import List, Literal

from functions.solving.match_pieces import match_pieces
from functions.solving.potential_piece_matches import get_potential_piece_matches
from functions.utils.normalize_list import normalize_list
from functions.utils.save_data import get_data_for_piece
from functions.utils.visualize_solution import visualize_solution


def solve_border(
    pieces_path: str,
    corner_piece_indices: List[int],
    edge_piece_indices: List[int],
    display_steps: bool = True,
    start_corner_index: int = 3,
    backtracking_threshold: float = 0.1,
    backtracking_enabled: bool = False,
    visualize_each_step: bool = False,
    save: bool = False,
    save_name: str | None = None,
):
    unsolved_piece_indices = set(corner_piece_indices + edge_piece_indices)

    # 1. Select a corner piece to begin with.
    selected_corner_piece_index = corner_piece_indices[start_corner_index]
    unsolved_piece_indices.remove(selected_corner_piece_index)

    # 2. Orient the selected corner piece.
    selected_corner_piece_side_data = get_data_for_piece(
        pieces_path, selected_corner_piece_index, "piece_side_data"
    )

    selected_corner_piece_flat_sides = []

    for side in ["A", "B", "C", "D"]:
        side_data = selected_corner_piece_side_data[side]
        side_classification = side_data["classification"]
        if side_classification == "FLT":
            selected_corner_piece_flat_sides.append(side)

    selected_corner_piece_flat_sides.sort()
    selected_corner_piece_top = selected_corner_piece_flat_sides[0]

    if ["A", "D"] == selected_corner_piece_flat_sides:
        selected_corner_piece_top = "D"

    # 3. Extract potential pieces and start matching
    # Initialize solving direction and location
    curr_row, curr_col = 0, 0
    directions = ["right", "down", "left", "up"]  # notice it's in clockwise order
    direction_index = 0

    # Space for target piece and solving data
    curr_piece_index = selected_corner_piece_index
    curr_piece_top_side = selected_corner_piece_top
    curr_direction = directions[direction_index]

    # Initialize solving data
    solution_matrix = add_to_solution_matrix(
        [], curr_row, curr_col, curr_piece_index, curr_piece_top_side
    )

    # Initialize dictionary for backtracking to best matches
    backtrack_data = {}

    # Loop until all pieces are solved
    while unsolved_piece_indices:
        # Update current row and column based on current direction
        if curr_direction == "right":
            curr_col += 1
        elif curr_direction == "down":
            curr_row += 1
        elif curr_direction == "left":
            curr_col -= 1
        elif curr_direction == "up":
            curr_row -= 1

        print(f"Solving for {curr_col}, {curr_row}")

        # Get current piece's relative matching side
        curr_piece_matching_side = get_piece_relative_side(
            curr_piece_top_side, curr_direction  # type: ignore
        )

        # Extract potential pieces
        potential_piece_indices = list(unsolved_piece_indices)

        # Flat direction of current piece (which side it's located)
        flat_direction = None
        if curr_direction == "right":
            flat_direction = "up"
        elif curr_direction == "down":
            flat_direction = "right"
        elif curr_direction == "left":
            flat_direction = "down"
        elif curr_direction == "up":
            flat_direction = "left"

        # Get potential piece matches
        potential_matches = get_potential_piece_matches(
            pieces_path,
            curr_piece_index,
            curr_piece_matching_side,  # type: ignore
            get_piece_relative_side(
                curr_piece_top_side,  # type: ignore
                flat_direction,  # type: ignore
            ),
            potential_piece_indices,
        )

        print(
            f"Potential matches for {curr_piece_index}{curr_piece_matching_side}: "
            f"{potential_matches}"
        )

        # No more potential matches found, break, could lead to issues and unsolved
        # pieces!
        if not potential_matches and len(unsolved_piece_indices) > 0:
            # TODO: Add backtracking
            break

        # Save potential matches and match scores
        hausdorff_distance = []
        hu_moments_distance = []
        corners_distance = []
        color_difference = []
        color_shape_distance = []

        print("Match scores:")
        for potential_match in potential_matches:
            potential_match_index, potential_match_side = potential_match
            match_score = match_pieces(
                pieces_path,
                curr_piece_index,
                potential_match_index,
                curr_piece_matching_side,  # type: ignore
                potential_match_side,
                display_steps=False,
            )

            print(potential_match)
            pprint(match_score)
            hausdorff_distance.append(match_score["Hausdorff Distance"])
            hu_moments_distance.append(match_score["Hu Moments Distance"])
            corners_distance.append(match_score["Corners Distance"])
            color_difference.append(match_score["Color Difference"])
            color_shape_distance.append(match_score["Color Shape Distance"])

        # Normalize match scores
        normalized_hausdorff_distance = normalize_list(
            hausdorff_distance, apply_non_linear=True
        )
        normalized_hu_moments_distance = normalize_list(
            hu_moments_distance, apply_non_linear=True
        )
        normalized_corners_distance = normalize_list(
            corners_distance, apply_non_linear=True
        )
        normalized_color_difference = normalize_list(
            color_difference, apply_non_linear=True
        )
        normalized_color_shape_distance = normalize_list(color_shape_distance)

        # Create wieghts for match error
        hausdorff_weight = 1
        hu_moments_weight = 1
        corner_dist_weight = 1
        color_weight = 1
        color_shape_weight = 0
        sum_weights = (
            hausdorff_weight
            + hu_moments_weight
            + corner_dist_weight
            + color_weight
            + color_shape_weight
        )

        # Calculate match error
        match_error = [
            (
                match,
                # Sum weighted error scores from each similarity measure
                hausdorff_weight * hausdorff
                + hu_moments_weight * hu_moments
                + corner_dist_weight * corner_dist
                + color_weight * color
                + color_shape_weight * color_shape,
            )
            for match, hausdorff, hu_moments, corner_dist, color, color_shape in zip(
                potential_matches,
                normalized_hausdorff_distance,
                normalized_hu_moments_distance,
                normalized_corners_distance,
                normalized_color_difference,
                normalized_color_shape_distance,
            )
        ]

        # Sort potential matches by match score
        match_error.sort(key=lambda x: x[1], reverse=True)

        # over here, save a dict that allows for backtracking
        # we need to save the piece we we're on before we went to the next one
        # we need to save a list of potential matches, these essentially skip the
        # "matching phase" and directly get accepted as the solution
        # one way to do this is by doing the following:

        pprint(match_error)

        # Get best match piece data
        best_match = match_error[0][0]
        best_match_index, best_match_side = best_match

        # Save the next few best matches; the next few best matches are the best pieces
        # (< backtracking_threshold % match score diff from original "best match")
        if len(backtrack_data) == 0:
            print("Since backtrack_data is empty, trying to save next best matches...")
            solution_matrix_copy = copy.deepcopy(solution_matrix)
            unsolved_piece_indices_copy = copy.deepcopy(unsolved_piece_indices)

            next_best_matches = [
                matches[0]
                for matches in match_error[1:]
                if (abs(match_error[0][1] - matches[1]) / sum_weights)
                <= backtracking_threshold
            ]

            if len(next_best_matches) == 0:
                print("No next best matches found!")
            else:
                next_best_matches.append(match_error[0][0])
                print(
                    f"Found {len(next_best_matches)} next best matches "
                    f"(< {backtracking_threshold}%): {next_best_matches}"
                )

                backtrack_data = {
                    "curr_piece_index": curr_piece_index,
                    "curr_piece_top_side": curr_piece_top_side,
                    "curr_direction": directions[direction_index],
                    "direction_index": direction_index,
                    "curr_row": curr_row,
                    "curr_col": curr_col,
                    "unsolved_piece_indices": unsolved_piece_indices_copy,
                    "solution_matrix": solution_matrix_copy,
                    "potential_matches": next_best_matches,
                }

        # Update the current piece data; notice to get the top side of the best match
        # piece, we need to pass the matched piece's side and the direction that was
        # used to match it (which is the opposite of the current matching direction).
        curr_piece_index = best_match_index
        curr_piece_top_side = get_piece_top_side(
            best_match_side, directions[(direction_index + 2) % 4]  # type: ignore
        )

        # Add best match to solution matrix
        try:
            solution_matrix = add_to_solution_matrix(
                solution_matrix,
                curr_row=curr_row,
                curr_col=curr_col,
                piece_index=best_match_index,
                piece_top_side=curr_piece_top_side,  # type: ignore
            )
        except ValueError as ve:
            if len(backtrack_data) == 0 or backtracking_enabled is False:
                print(
                    "No backtrack data found to backtrack (or it is disabled)! "
                    "Stopping on error: ",
                    ve,
                )
            else:
                print("Backtracking to previous best match...")
                print("Current backtrack data: ")
                pprint(backtrack_data)
                next_best_match = backtrack_data["potential_matches"][0]

                curr_direction = backtrack_data["curr_direction"]
                direction_index = backtrack_data["direction_index"]

                best_match_index = next_best_match[0]
                curr_piece_index = best_match_index
                curr_piece_top_side = get_piece_top_side(
                    next_best_match[1],
                    directions[(direction_index + 2) % 4],  # type: ignore
                )

                curr_row = backtrack_data["curr_row"]
                curr_col = backtrack_data["curr_col"]

                solution_matrix = copy.deepcopy(backtrack_data["solution_matrix"])
                print("Current solution matrix: ")
                pprint(solution_matrix)
                solution_matrix = add_to_solution_matrix(
                    solution_matrix,
                    curr_row=curr_row,
                    curr_col=curr_col,
                    piece_index=best_match_index,
                    piece_top_side=curr_piece_top_side,  # type: ignore
                )

                unsolved_piece_indices = copy.deepcopy(
                    backtrack_data["unsolved_piece_indices"]
                )
                backtrack_data["potential_matches"] = backtrack_data[
                    "potential_matches"
                ][1:]

                if len(backtrack_data["potential_matches"]) == 0:
                    print("Resetting backtrack data!")
                    backtrack_data = {}

        # Get best match's piece classification
        best_match_classification = get_data_for_piece(
            pieces_path, best_match_index, "classification"
        )
        # If best match is a corner, change direction
        if best_match_classification == "CNR":
            direction_index = (direction_index + 1) % 4
            curr_direction = directions[direction_index]  # type: ignore
            print(f"Hit a corner! Changed direction to {curr_direction}.")

        if visualize_each_step:
            visualize_solution(
                pieces_path, solution_matrix, save=False, display_steps=True
            )

        # Remove matched piece from unsolved pieces
        unsolved_piece_indices.remove(best_match_index)

    pprint(solution_matrix)

    visualize_solution(
        pieces_path,
        solution_matrix,
        display_steps=display_steps,
        save_name=save_name,
        save=save,
    )

    return solution_matrix


def get_piece_relative_side(
    piece_top: Literal["A", "B", "C", "D"],
    piece_side_orientation: Literal["right", "down", "left", "up"],
) -> str:
    if piece_side_orientation == "up":
        return piece_top

    sides = ["D", "C", "B", "A"]
    side_index = sides.index(piece_top)

    if piece_side_orientation == "down":
        return sides[(side_index + 2) % 4]

    if piece_side_orientation == "left":
        return sides[(side_index + 3) % 4]

    if piece_side_orientation == "right":
        return sides[(side_index + 1) % 4]

    raise ValueError(f"Invalid piece_side_orientation: {piece_side_orientation}")


def get_piece_top_side(
    piece_side: Literal["A", "B", "C", "D"],
    piece_side_orientation: Literal["right", "down", "left", "up"],
) -> str:

    if piece_side_orientation == "up":
        return piece_side

    sides = ["D", "C", "B", "A"]
    side_index = sides.index(piece_side)

    if piece_side_orientation == "right":
        return sides[(side_index + 3) % 4]

    if piece_side_orientation == "down":
        return sides[(side_index + 2) % 4]

    if piece_side_orientation == "left":
        return sides[(side_index + 1) % 4]

    raise ValueError(f"Invalid piece_side_orientation: {piece_side_orientation}")


def add_to_solution_matrix(
    solution_matrix: List[List[dict | None]],
    curr_row: int,
    curr_col: int,
    piece_index: int,
    piece_top_side: Literal["A", "B", "C", "D"],
) -> List[List[dict | None]]:
    # Ensure the matrix has at least one row
    if len(solution_matrix) == 0:
        solution_matrix.append([None] * (curr_col + 1))

    # Expand rows to accommodate the current row
    if curr_row >= len(solution_matrix):
        row_width = len(solution_matrix[0])
        solution_matrix.extend(
            [None] * row_width for _ in range(curr_row - len(solution_matrix) + 1)
        )

    # Expand columns to accommodate the current column
    if curr_col >= len(solution_matrix[0]):
        for row in solution_matrix:
            row.extend([None] * (curr_col - len(row) + 1))

    if solution_matrix[curr_row][curr_col] is not None:
        raise ValueError(
            f"Cell ({curr_row}, {curr_col}) is already filled with a piece!"
        )

    # Add the piece to the specified matrix location
    solution_matrix[curr_row][curr_col] = {
        "piece_index": piece_index,
        "piece_top_side": piece_top_side,
    }

    return solution_matrix
