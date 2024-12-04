"""
display_pieces_with_data.py - Display a list of puzzle piece images in a grid layout
with associated data.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def display_pieces_with_data(
    piece_data: List[dict],
    pieces_path: str,
    save_name: str,
    figsize: tuple = (10, 10),
    save: bool = True,
    display_steps: bool = True,
):
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

        # Load the piece image.
        piece_img = plt.imread(f"{pieces_path}/piece_{piece_index}.png")

        # Create subplot for each piece.
        plt.subplot(rows, cols, i + 1)

        # Display the piece image with extracted data.
        plt.imshow(piece_img)
        plt.plot(center_coords[0], center_coords[1], "ro")
        for corner in corners:
            plt.plot(corner[0], corner[1], "bo")
        plt.title(f"Piece {piece_index} ({piece_classification})")
        plt.axis("off")

    # Adjust layout to avoid overlap.
    plt.tight_layout()

    # Save the figure as an image file if needed.
    if save:
        plt.savefig(f"{save_name}.png", dpi=300)

    if display_steps:
        # Show the plot.
        plt.show()

        # Clear the figure after saving and showing to avoid issues.
        plt.clf()
