"""
match_pieces.py -  This takes in two pieces and determines a match score between them.
"""

from typing import Literal

from scipy.spatial.distance import directed_hausdorff

from functions.utils.orient_piece_side import orient_piece_side


def match_pieces(
    pieces_path: str,
    piece1: int,
    piece2: int,
    side1: Literal["A", "B", "C", "D"],
    side2: Literal["A", "B", "C", "D"],
):
    """
    Calculate the match score between two pieces.

    Args:
        pieces_path (str): The path to the pieces data JSON file.
        piece1 (int): The index of the first piece.
        piece2 (int): The index of the second piece.
        side1 (Literal["A", "B", "C", "D"]): The side of the first piece.
        side2 (Literal["A", "B", "C", "D"]): The side of the second piece.

    Returns:
        float: The match score between the two pieces.
    """
    piece1_oriented_contours = orient_piece_side(pieces_path, piece1, side1, "Top")
    piece2_oriented_contours = orient_piece_side(pieces_path, piece2, side2, "Top")

    # Compute Hausdorff distances
    # Hausdorff distance is symmetric, so we'll compute it in both directions
    forward_dist = directed_hausdorff(
        piece1_oriented_contours, piece2_oriented_contours
    )[0]
    reverse_dist = directed_hausdorff(
        piece2_oriented_contours, piece1_oriented_contours
    )[0]

    # Take the maximum of the two directed Hausdorff distances
    hausdorff_dist = max(forward_dist, reverse_dist)

    return hausdorff_dist
