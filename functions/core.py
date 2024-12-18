"""
core.py - This file combines logic from other modules to help process and solve puzzles.
"""

import os
import re

import cv2

from functions.preprocessing.piece_to_polygon import piece_to_polygon
from functions.preprocessing.puzzle_to_pieces import puzzle_to_pieces
from functions.solving.solve_border import solve_border
from functions.solving.solve_center import solve_center
from functions.utils.color_data import piece_color_data
from functions.utils.display_piece_corners import display_piece_corners
from functions.utils.display_pieces import display_pieces
from functions.utils.piece_file_names import get_piece_file_names
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
    add_random_rotation: bool = False,
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
        display_steps (bool, optional): Flag to display intermediate steps.
        add_random_rotation (bool, optional): Flag to apply random rotation to
            puzzle pieces.

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

    pieces = puzzle_to_pieces(
        PUZZLE_IMAGE_PATH,
        kernel_size=(5, 5),
        display_steps=display_steps,
        add_random_rotation=add_random_rotation,
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

        # Update save path for the grid visualization
        PIECES_GRID_SAVE_PATH += f"/{PUZZLE_IMAGE_NAME}"

    # Display pieces in a grid
    display_pieces(
        pieces, save_name=PIECES_GRID_SAVE_PATH, save=save, display_steps=display_steps
    )


def preprocess_pieces(
    PUZZLE_IMAGE_NAME: str,
    save: bool = False,
    save_folder: str = "images",
    display_steps: bool = True,
    display_non_essential_steps: bool = False,
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
        display_steps (bool, optional): Flag to display intermediate steps.
        display_non_essential_steps (bool, optional): Flag to display non-essential
            steps. Defaults to False.
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

    # Store matching file paths
    piece_files = get_piece_file_names(PIECES_SAVE_PATH, piece_pattern)

    # List to store piece data
    piece_data = []

    # Process each piece file
    for piece_file in piece_files:
        # Extract piece index value
        match = piece_pattern.search(os.path.basename(piece_file))
        piece_index = int(match.group(1))  # type: ignore

        print(f"Processing piece {piece_index}")

        # Extracting piece data.
        corners, center_coords, piece_classification, piece_side_data = (
            piece_to_polygon(
                piece_file,
                epsilon_ratio=0.01,
                # distance from center and corner points (higher is better)
                corner_distance_weight=0.7,
                # angle error (difference from 2x45 and 90) between corner and adjacent
                # points (lower is better)
                corner_angle_weight=1.5,
                # angle error (difference from 90) between center and corner points
                # (lower is better)
                center_angle_weight=1,
                # distance error (difference from average distance) between center and
                # corner points (lower is better)
                center_distance_weight=1,
                intrusion_threshold=0.7,
                display_steps=display_steps,
            )
        )

        # Saving extracted piece data.
        if save:
            save_data_for_piece(PIECES_SAVE_PATH, piece_index, "corners", corners)
            save_data_for_piece(
                PIECES_SAVE_PATH, piece_index, "center_coords", center_coords
            )
            save_data_for_piece(
                PIECES_SAVE_PATH,
                piece_index,
                "classification",
                piece_classification,
            )
            save_data_for_piece(
                PIECES_SAVE_PATH,
                piece_index,
                "piece_side_data",
                piece_side_data,
                avoid_print=True,
            )

        # Adding color data to piece_side_data
        piece_side_data_with_color_data = piece_side_data

        # Extracting additional color data from each piece.
        for side in ["A", "B", "C", "D"]:
            piece_side_data_with_color_data[side]["color_data"] = piece_color_data(
                PIECES_SAVE_PATH,
                piece_index,
                side,
                display_steps=display_non_essential_steps,
            )

        # Saving extracted color data.
        if save:
            save_data_for_piece(
                PIECES_SAVE_PATH,
                piece_index,
                "piece_side_data",
                piece_side_data_with_color_data,
                avoid_print=True,
            )

        # Saving data here for each piece to display in grid format later.
        piece_data.append(
            {
                "piece_index": piece_index,
                "center_coords": center_coords,
                "corners": corners,
                "piece_classification": piece_classification,
                "piece_side_data": piece_side_data,
            }
        )

        print(piece_file, corners)

    # Display pieces with data in a grid.
    display_piece_corners(
        piece_data,
        pieces_path=PIECES_SAVE_PATH,
        save=save,
        save_name=f"{PIECES_DATA_GRID_SAVE_PATH}/{PUZZLE_IMAGE_NAME}_1_corners",
        display_steps=display_steps,
    )


def solve_puzzle(
    PUZZLE_IMAGE_NAME: str,
    save: bool = False,
    save_folder: str = "images",
    display_steps: bool = True,
    display_non_essential_steps: bool = False,
):
    """
    Solves a puzzle and displays the solution.

    Args:
        PUZZLE_IMAGE_NAME (str): The name of the puzzle image to be solved.
        save (bool, optional): Whether to save the solution. Defaults to False.
        save_folder (str, optional): Directory where the output images will be saved.
            Defaults to "images".
        display_steps (bool, optional): Flag to display intermediate steps.
        display_non_essential_steps (bool, optional): Flag to display non-essential
            steps. Defaults to False.

    Returns:
        None
    """
    # Define the folder path
    PIECES_SAVE_PATH = f"{save_folder}/pieces/{PUZZLE_IMAGE_NAME}"
    PIECES_DATA_GRID_SAVE_PATH = f"{save_folder}/pieces_grid"

    # Regex pattern to match files named piece_{i}.png
    piece_pattern = re.compile(r"^piece_(\d+)\.png$")

    # Store matching file paths
    piece_files = get_piece_file_names(PIECES_SAVE_PATH, piece_pattern)

    # List to store piece data
    corner_piece_indices = []
    corner_pieces = []
    edge_piece_indices = []
    edge_pieces = []
    non_border_piece_indices = []
    non_border_pieces = []

    # Process each piece file
    for piece_file in piece_files:
        # Extract piece index value
        match = piece_pattern.search(os.path.basename(piece_file))
        piece_index = int(match.group(1))  # type: ignore

        piece_classification = get_data_for_piece(
            PIECES_SAVE_PATH, piece_index, "classification"
        )

        img = cv2.imread(PIECES_SAVE_PATH + f"/piece_{piece_index}.png")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if piece_classification == "CNR":
            corner_piece_indices.append(piece_index)
            corner_pieces.append(img_rgb)
            print(f"Classified piece {piece_index} as corner")
        elif piece_classification == "EDG":
            edge_piece_indices.append(piece_index)
            edge_pieces.append(img_rgb)
            print(f"Classified piece {piece_index} as edge")
        else:
            non_border_piece_indices.append(piece_index)
            non_border_pieces.append(img_rgb)
            print(f"Classified piece {piece_index} as non-border")

    # Display corner pieces in a grid
    display_pieces(
        corner_pieces,
        save_name=PIECES_DATA_GRID_SAVE_PATH + f"/{PUZZLE_IMAGE_NAME}_2_corner",
        save=save,
        display_steps=display_steps,
    )

    # Display edge pieces in a grid
    display_pieces(
        edge_pieces,
        save_name=PIECES_DATA_GRID_SAVE_PATH + f"/{PUZZLE_IMAGE_NAME}_3_border",
        save=save,
        display_steps=display_steps,
    )

    # Display non-border pieces in a grid
    display_pieces(
        non_border_pieces,
        save_name=PIECES_DATA_GRID_SAVE_PATH + f"/{PUZZLE_IMAGE_NAME}_4_non_border",
        save=save,
        display_steps=display_steps,
    )

    # Set a starting corner index
    for start_corner_index in range(len(corner_piece_indices)):
        try:
            # 2. Pass the border pieces to solve_border
            solved_border_matrix = solve_border(
                pieces_path=PIECES_SAVE_PATH,
                corner_piece_indices=corner_piece_indices,
                edge_piece_indices=edge_piece_indices,
                start_corner_index=start_corner_index,
                display_steps=display_steps,
                save=save,
                save_name=PIECES_DATA_GRID_SAVE_PATH
                + f"/{PUZZLE_IMAGE_NAME}_5_border_{start_corner_index}",
                visualize_each_step=display_non_essential_steps,
            )
        except Exception as e:
            print(
                f"Error solving border pieces: {e} "
                f"with corner index {start_corner_index}"
            )
            continue

        try:
            # 3. Pass the center pieces to solve_center
            solve_center(
                PIECES_SAVE_PATH,
                solved_border_matrix=solved_border_matrix,  # type: ignore
                middle_piece_indices=non_border_piece_indices,
                display_steps=display_steps,
                save=save,
                save_name=PIECES_DATA_GRID_SAVE_PATH
                + f"/{PUZZLE_IMAGE_NAME}_6_solution_{start_corner_index}",
                visualize_each_step=display_non_essential_steps,
            )
        except Exception as e:
            print(f"Error solving center pieces: {e}")
