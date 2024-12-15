"""
potential_piece_matches.py - Utility functions to get the names of puzzle pieces that
might match a given piece.
"""

from typing import List, Literal, Tuple

from functions.utils.save_data import get_data_for_piece


def get_potential_piece_matches(
    pieces_path: str,
    target_piece_index: int,
    target_piece_side: Literal["A", "B", "C", "D"],
    target_flat_side: Literal["A", "B", "C", "D"] | None,
    potential_piece_indices: List[int],
) -> List[Tuple[int, Literal["A", "B", "C", "D"]]]:
    # Load target piece data
    target_piece_data = get_data_for_piece(
        pieces_path, target_piece_index, "piece_side_data"
    )[target_piece_side]

    # Check if target piece is flat
    target_piece_classification = target_piece_data["classification"]

    # If target piece is flat, return empty list, since nothing can be matched
    if target_piece_classification == "FLT":
        return []

    # Initialize list of potential matches
    potential_matches = []

    # Loop through potential pieces
    for potential_piece_index in potential_piece_indices:
        # Load potential piece data
        potential_piece_data = get_data_for_piece(
            pieces_path, potential_piece_index, "piece_side_data"
        )

        # Loop through potential piece sides
        for side in ["A", "B", "C", "D"]:
            # Check if potential piece side is flat
            potential_piece_side_data = potential_piece_data[side]
            potential_piece_side_classification = potential_piece_side_data[
                "classification"
            ]

            # If potential piece side is flat, skip it
            if potential_piece_side_classification == "FLT":
                continue

            # If potential piece side matches target piece classification, skip since
            # we can't match INT to INT and EXT to EXT
            if potential_piece_side_classification == target_piece_classification:
                continue

            if target_flat_side is not None:
                # Get potential flat side
                potential_flat_side = get_potential_flat_side(
                    target_flat_side, target_piece_side, side  # type: ignore
                )

                # If potential flat side doesn't match target flat side, skip
                if potential_flat_side is not None:
                    potential_piece_flat_side_data = potential_piece_data[
                        potential_flat_side
                    ]
                    potential_piece_flat_side_classification = (
                        potential_piece_flat_side_data["classification"]
                    )
                    if potential_piece_flat_side_classification != "FLT":
                        continue

            # Add potential match to list
            potential_matches.append((potential_piece_index, side))

    # Return potential matches
    return potential_matches


def get_potential_flat_side(
    target_flat_side: Literal["A", "B", "C", "D"],
    target_match_side: Literal["A", "B", "C", "D"],
    potential_side: Literal["A", "B", "C", "D"],
) -> str | None:
    sides = ["D", "C", "B", "A"]

    target_flat_side_index = sides.index(target_flat_side)
    target_match_side_index = sides.index(target_match_side)
    potential_side_index = sides.index(potential_side)

    if target_flat_side_index + target_match_side_index % 2 == 0:
        return None

    return sides[
        potential_side_index - ((target_flat_side_index - target_match_side_index) % 4)
    ]
