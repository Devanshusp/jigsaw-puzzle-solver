"""
solve_border.py - This file contains the logic to solve the border of a puzzle.
"""

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
):
    unsolved_piece_indices = set(corner_piece_indices + edge_piece_indices)

    # 1. Select a corner piece to begin with.
    selected_corner_piece_index = corner_piece_indices[0]
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

    # 3. Extract potential pieces to start matching
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
        if not potential_matches:
            break

        # Save potential matches and match scores
        hausdorff_distance = []
        fourier_descriptor_similarity = []
        procrustes_shape_similarity = []
        cosine_similarity = []

        for potential_match in potential_matches:
            potential_match_index, potential_match_side = potential_match
            match_score = match_pieces(
                pieces_path,
                curr_piece_index,
                potential_match_index,
                curr_piece_matching_side,  # type: ignore
                potential_match_side,
            )

            hausdorff_distance.append(match_score["Hausdorff Distance"])
            fourier_descriptor_similarity.append(
                match_score["Fourier Descriptor Similarity"]
            )
            procrustes_shape_similarity.append(
                match_score["Procrustes Shape Similarity"]
            )
            cosine_similarity.append(match_score["Cosine Similarity"])

        # Normalize match scores
        normalized_hausdorff_distance = normalize_list(hausdorff_distance)
        normalized_fourier_descriptor_similarity = normalize_list(
            fourier_descriptor_similarity
        )
        normalized_procrustes_shape_similarity = normalize_list(
            procrustes_shape_similarity, reverse=True
        )
        normalized_cosine_similarity = normalize_list(cosine_similarity, reverse=True)

        # Calculate match error
        match_error = [
            (
                match,
                # Sum weighted error scores from each similarity measure
                0.5 * hausdorff + 0.5 * fourier + 1 * procrustes + 0 * cosine,
            )
            for match, hausdorff, fourier, procrustes, cosine in zip(
                potential_matches,
                normalized_hausdorff_distance,
                normalized_fourier_descriptor_similarity,
                normalized_procrustes_shape_similarity,
                normalized_cosine_similarity,
            )
        ]

        # Sort potential matches by match score
        match_error.sort(key=lambda x: x[1], reverse=True)

        pprint(match_error)

        # Get best match piece data
        best_match = match_error[0][0]
        best_match_index, best_match_side = best_match

        # Update the current piece data; notice to get the top side of the best match
        # piece, we need to pass the matched piece's side and the direction that was
        # used to match it (which is the opposite of the current matching direction).
        curr_piece_index = best_match_index
        curr_piece_top_side = get_piece_top_side(
            best_match_side, directions[(direction_index + 2) % 4]  # type: ignore
        )

        # Add best match to solution matrix
        solution_matrix = add_to_solution_matrix(
            solution_matrix,
            curr_row=curr_row,
            curr_col=curr_col,
            piece_index=best_match_index,
            piece_top_side=curr_piece_top_side,  # type: ignore
        )

        # Get best match's piece classification
        best_match_classification = get_data_for_piece(
            pieces_path, best_match_index, "classification"
        )

        # If best match is a corner, change direction
        if best_match_classification == "CNR":
            direction_index = (direction_index + 1) % 4
            curr_direction = directions[direction_index]  # type: ignore
            print(f"Hit a corner! Changed direction to {curr_direction}.")

        # Remove matched piece from unsolved pieces
        unsolved_piece_indices.remove(best_match_index)

    pprint(solution_matrix)

    visualize_solution(pieces_path, solution_matrix)


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
            f"Cell ({curr_row}, {curr_col}) is already filled with a piece."
        )

    # Add the piece to the specified matrix location
    solution_matrix[curr_row][curr_col] = {
        "piece_index": piece_index,
        "piece_top_side": piece_top_side,
    }

    return solution_matrix
