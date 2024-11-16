"""
main.py - The main script to run the Jigsaw Puzzle Solver.
"""

import os

import cv2

from utils import display_pieces, puzzle_to_pieces


def main(PUZZLE_IMAGE_NAME: str):
    PUZZLE_IMAGE_PATH = f"images/puzzles/{PUZZLE_IMAGE_NAME}.png"
    PIECES_SAVE_PATH = f"images/pieces/{PUZZLE_IMAGE_NAME}"

    # make pieces folder if it doesnt exist
    if not os.path.exists(PIECES_SAVE_PATH):
        os.makedirs(PIECES_SAVE_PATH)

    num_pieces, pieces = puzzle_to_pieces(PUZZLE_IMAGE_PATH, kernel_size=(5, 5))

    # Print the result
    print(f"Number of puzzle pieces detected: {num_pieces}")

    # store pieces
    for i, piece in enumerate(pieces):
        piece = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{PIECES_SAVE_PATH}/piece_{i}.png", piece)

    # store pieces in a grid
    display_pieces(pieces, save_name=f"images/pieces_grid/{PUZZLE_IMAGE_NAME}_pieces")


if __name__ == "__main__":
    img_names = [
        "aurora12",
        "aurora30",
        "aurora63",
        "corn12",
        "corn30",
        "corn63",
        "path12",
        "path30",
        "path63",
        "red9",
        "red25",
        "red49",
        "younker12",
        "younker30",
        "younker63",
    ]

    for img_name in img_names:
        main(img_name)
