"""
display_piece_corners.py - Display a list of puzzle piece images in a grid layout
with associated data.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def display_piece_corners(
    piece_data: List[dict],
    pieces_path: str,
    save_name: str,
    figsize: tuple = (10, 10),
    save: bool = True,
    display_steps: bool = True,
):
    """
    Display a list of puzzle piece images in a grid layout with extracted data.

    Args:
        piece_data (List[dict]): A list of dictionaries containing piece data.
        pieces_path (str): Path to the pieces data.
        save_name (str): Name of the saved image.
        figsize (tuple, optional): Figure size. Defaults to (10, 10).
        save (bool, optional): Flag to save the visualization. Defaults to True.
        display_steps (bool, optional): Flag to display intermediate steps.
            Defaults to True.

    Returns:
        None
    """
    # Calculate number of pieces.
    num_pieces = len(piece_data)

    # Calculate grid dimensions (rows and columns).
    rows = int(np.ceil(np.sqrt(num_pieces)))
    cols = int(np.ceil(num_pieces / rows))

    # Create a new figure for displaying all pieces.
    plt.figure(figsize=figsize)

    # Add each piece to the grid.
    for i, piece in enumerate(piece_data):
        # Extract piece data.
        piece_index = piece["piece_index"]
        center_coords = piece["center_coords"]
        corners = piece["corners"]
        piece_classification = piece["piece_classification"]
        piece_side_data = piece["piece_side_data"]

        side_a = piece_side_data["A"]["points"]
        side_b = piece_side_data["B"]["points"]
        side_c = piece_side_data["C"]["points"]
        side_d = piece_side_data["D"]["points"]

        side_a_classification = piece_side_data["A"]["classification"]
        side_b_classification = piece_side_data["B"]["classification"]
        side_c_classification = piece_side_data["C"]["classification"]
        side_d_classification = piece_side_data["D"]["classification"]

        # Load the piece image.
        piece_img = plt.imread(f"{pieces_path}/piece_{piece_index}.png")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(piece_img)

        # Plot the center point in red
        plt.plot(center_coords[0], center_coords[1], "ro")

        # Plot each corner in blue and draw red lines to center
        for corner in corners:
            plt.plot(corner[0], corner[1], "bo")
            plt.plot([center_coords[0], corner[0]], [center_coords[1], corner[1]], "r-")

        # Plot sides with different colors and labels
        # Side A: Green
        plt.plot(
            [side_a[0][0], side_a[1][0]],
            [side_a[0][1], side_a[1][1]],
            "g-",
            linewidth=2,
        )
        plt.text(
            np.mean([side_a[0][0], side_a[1][0]]),  # type: ignore
            np.mean([side_a[0][1], side_a[1][1]]),  # type: ignore
            f"A ({side_a_classification})",
            color="green",
            fontweight="bold",
            fontsize=10,
        )

        # Side B: Purple
        plt.plot(
            [side_b[0][0], side_b[1][0]],
            [side_b[0][1], side_b[1][1]],
            "m-",
            linewidth=2,
        )
        plt.text(
            np.mean([side_b[0][0], side_b[1][0]]),  # type: ignore
            np.mean([side_b[0][1], side_b[1][1]]),  # type: ignore
            f"B ({side_b_classification})",
            color="magenta",
            fontweight="bold",
            fontsize=10,
        )

        # Side C: Cyan
        plt.plot(
            [side_c[0][0], side_c[1][0]],
            [side_c[0][1], side_c[1][1]],
            "c-",
            linewidth=2,
        )
        plt.text(
            np.mean([side_c[0][0], side_c[1][0]]),  # type: ignore
            np.mean([side_c[0][1], side_c[1][1]]),  # type: ignore
            f"C ({side_c_classification})",
            color="cyan",
            fontweight="bold",
            fontsize=10,
        )

        # Side D: Orange
        plt.plot(
            [side_d[0][0], side_d[1][0]],
            [side_d[0][1], side_d[1][1]],
            color="orange",
            linewidth=2,
        )
        plt.text(
            np.mean([side_d[0][0], side_d[1][0]]),  # type: ignore
            np.mean([side_d[0][1], side_d[1][1]]),  # type: ignore
            f"D ({side_d_classification})",
            color="orange",
            fontweight="bold",
            fontsize=10,
        )

        plt.title(f"Piece {piece_index} ({piece_classification})")
        plt.axis("off")

    plt.tight_layout()

    # Save the figure as an image file if needed.
    if save:
        plt.savefig(f"{save_name}.png", dpi=300)

    if display_steps:
        plt.show()

    # Clear the figure after saving and showing to avoid issues.
    plt.clf()
    plt.close()
