"""
core.py - This file combines logic from other modules to help process and solve puzzles.
"""

import os
import re

import cv2

from functions.preprocessing.piece_to_polygon import piece_to_polygon
from functions.preprocessing.puzzle_to_pieces import puzzle_to_pieces
from functions.utils.display_pieces import display_pieces
from functions.utils.display_pieces_with_data import display_pieces_with_data
from functions.utils.save_data import (
    get_data_for_piece,
    init_save_file,
    save_data_for_piece,
)


def preprocess_puzzle(
    PUZZLE_IMAGE_NAME: str,
    save: bool = False,
    save_folder: str = "images",
    puzzle_folder: str = "puzzles",
    display_steps: bool = True,
) -> None:
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

    pieces, centers = puzzle_to_pieces(
        PUZZLE_IMAGE_PATH, kernel_size=(5, 5), display_steps=display_steps
    )

    # Print the result
    print(f"Number of puzzle pieces detected: {len(pieces)}")

    if save:
        # Create directories if they do not exist
        if not os.path.exists(PIECES_SAVE_PATH):
            os.makedirs(PIECES_SAVE_PATH)

        if not os.path.exists(PIECES_GRID_SAVE_PATH):
            os.makedirs(PIECES_GRID_SAVE_PATH)

        init_save_file(PIECES_SAVE_PATH, len(pieces))

        # Save individual pieces
        for i, piece in enumerate(pieces):
            # Save clean crops of pieces
            piece = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{PIECES_SAVE_PATH}/piece_{i}.png", piece)

            # Save clean crops of pieces with a red dot at the center
            center_coords = (int(centers[i][0]), int(centers[i][1]))
            piece_with_center = cv2.circle(
                piece,
                center_coords,
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )
            cv2.imwrite(
                f"{PIECES_SAVE_PATH}/piece_{i}_with_center.png", piece_with_center
            )

            # Save center data for each piece
            save_data_for_piece(PIECES_SAVE_PATH, i, "center_coords", center_coords)

        # Update save path for the grid visualization
        PIECES_GRID_SAVE_PATH += f"/{PUZZLE_IMAGE_NAME}_pieces"

    # Display pieces in a grid
    display_pieces(
        pieces, save_name=PIECES_GRID_SAVE_PATH, save=save, display_steps=display_steps
    )


def preprocess_pieces(
    PUZZLE_IMAGE_NAME: str,
    save: bool = False,
    save_folder: str = "images",
    display_steps: bool = True,
) -> None:
    """
    Processes a folder of puzzle pieces to extract their polygons and optionally saves
    them.

    Args:
        PUZZLE_IMAGE_NAME (str): The name of puzzle to be processed.
        save (bool, optional): Whether to save the extracted polygons. Defaults to
            False.
        save_folder (str, optional): Directory where the output images will be saved.
            Defaults to "images".
    Steps:
        1. Reads the specified puzzle image.
        2. Extracts individual puzzle pieces using `puzzle_to_pieces`.
        3. Optionally saves each piece to the specified folder.
        4. Displays the pieces in a grid format, with an option to save the
            visualization.

    Returns:
        None
    """

    # Define the folder path
    PIECES_SAVE_PATH = f"{save_folder}/pieces/{PUZZLE_IMAGE_NAME}"
    PIECES_DATA_GRID_SAVE_PATH = f"{save_folder}/pieces_grid"

    # Regex pattern to match files named piece_{i}.png
    piece_pattern = re.compile(r"^piece_(\d+)\.png$")

    # List to store matching file paths
    piece_files = []

    # Loop through all files in the directory
    for filename in os.listdir(PIECES_SAVE_PATH):
        if piece_pattern.match(filename):
            piece_files.append(os.path.join(PIECES_SAVE_PATH, filename))

    # Sort files by the piece number
    piece_files.sort(
        key=lambda x: int(
            piece_pattern.search(os.path.basename(x)).group(1)  # type: ignore
        )
    )

    piece_data = []

    # Process each piece file
    for piece_file in piece_files:
        # Extract piece index value
        match = piece_pattern.search(os.path.basename(piece_file))
        piece_index = int(match.group(1))  # type: ignore

        print(f"Processing piece {piece_index}")

        # Get saved center coordinates for the piece
        center_coords = get_data_for_piece(
            PIECES_SAVE_PATH,
            piece_index,
            "center_coords",
        )

        # Extracting piece data.
        corners, piece_classification = piece_to_polygon(
            piece_file, center_coords, epsilon_ratio=0.02, display_steps=display_steps
        )

        # Saving extracted piece data.
        if save:
            save_data_for_piece(PIECES_SAVE_PATH, piece_index, "corners", corners)
            save_data_for_piece(
                PIECES_SAVE_PATH,
                piece_index,
                "classification",
                piece_classification,
            )

        # Saving data here for each piece to display in grid format later.
        piece_data.append(
            {
                "piece_index": piece_index,
                "center_coords": center_coords,
                "corners": corners,
                "piece_classification": piece_classification,
            }
        )

        print(piece_file, corners)

    # Display pieces with data in a grid.
    display_pieces_with_data(
        piece_data,
        pieces_path=PIECES_SAVE_PATH,
        save=save,
        save_name=f"{PIECES_DATA_GRID_SAVE_PATH}/{PUZZLE_IMAGE_NAME}_pieces_with_data",
        display_steps=display_steps,
    )
