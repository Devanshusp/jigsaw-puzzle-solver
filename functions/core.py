"""
core.py - This file combines logic from other modules to help process and solve puzzles.
"""

import os

import cv2

from functions.preprocessing.puzzle_to_pieces import puzzle_to_pieces
from functions.utils.display_pieces import display_pieces


def preprocess_puzzle(
    PUZZLE_IMAGE_NAME: str,
    save: bool = False,
    save_folder: str = "images",
    puzzle_folder: str = "puzzles",
):
    """
    Processes a puzzle image to extract its pieces and optionally saves them and a grid
    visualization.

    Args:
        PUZZLE_IMAGE_NAME (str): The name of the puzzle image (without extension) to be
            processed.
        save (bool, optional): Whether to save the extracted pieces and the grid
            visualization. Defaults to False.
        save_folder (str, optional): Directory where the output images will be saved.
            Defaults to "images".
        puzzle_folder (str, optional): Directory where the input puzzle image is
            located. Defaults to "puzzles".

    Steps:
        1. Reads the specified puzzle image.
        2. Extracts individual puzzle pieces using `puzzle_to_pieces`.
        3. Optionally saves each piece to the specified folder.
        4. Displays the pieces in a grid format, with an option to save the
            visualization.

    Returns:
        None
    """
    # Access paths
    SAVE_FOLDER = save_folder
    PUZZLE_IMAGE_PATH = f"{puzzle_folder}/{PUZZLE_IMAGE_NAME}.png"

    # Save paths
    PIECES_SAVE_PATH = f"{SAVE_FOLDER}/pieces/{PUZZLE_IMAGE_NAME}"
    PIECES_GRID_SAVE_PATH = f"{SAVE_FOLDER}/pieces_grid"

    pieces = puzzle_to_pieces(PUZZLE_IMAGE_PATH, kernel_size=(5, 5))

    # Print the result
    print(f"Number of puzzle pieces detected: {len(pieces)}")

    if save:
        # Create directories if they do not exist
        if not os.path.exists(PIECES_SAVE_PATH):
            os.makedirs(PIECES_SAVE_PATH)

        if not os.path.exists(PIECES_GRID_SAVE_PATH):
            os.makedirs(PIECES_GRID_SAVE_PATH)

        # Save individual pieces
        for i, piece in enumerate(pieces):
            piece = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{PIECES_SAVE_PATH}/piece_{i}.png", piece)

        # Update save path for the grid visualization
        PIECES_GRID_SAVE_PATH += f"/{PUZZLE_IMAGE_NAME}_pieces"
    else:
        # Disable saving the grid visualization
        PIECES_GRID_SAVE_PATH = None

    # Display pieces in a grid
    display_pieces(pieces, save_name=PIECES_GRID_SAVE_PATH)
