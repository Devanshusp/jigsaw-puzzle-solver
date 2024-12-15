"""
match_pieces.py -  This takes in two pieces and determines a match score between them.
"""

from functions.utils.save_data import get_data_for_piece


def match_pieces(pieces_path: str, piece1: int, piece2: int, side1: str, side2: str):
    piece1_side_data = get_data_for_piece(pieces_path, piece1, "piece_side_data")[side1]
    piece2_side_data = get_data_for_piece(pieces_path, piece2, "piece_side_data")[side2]

    print(piece1_side_data)
    print(piece2_side_data)

    return
