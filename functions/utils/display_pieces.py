"""
diplay_pieces.py - Display a list of puzzle piece images in a grid layout.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def display_pieces(
    pieces: List[np.ndarray],
    figsize: tuple = (10, 10),
    save: bool = True,
    save_name: str | None = None,
    display_steps: bool = True,
) -> None:
    """
    Display a list of puzzle piece images in a grid layout.

    Args:
        pieces: A list of image pieces (numpy arrays) to be displayed.

    Returns:
        None: It shows the pieces in a grid layout using Matplotlib.
    """
    # Calculate number of pieces.
    num_pieces = len(pieces)

    # Calculate grid dimensions (rows and columns).
    try:
        rows = int(np.ceil(np.sqrt(num_pieces)))
        cols = int(np.ceil(num_pieces / rows))
    except ZeroDivisionError:
        print("Not enough pieces to display in a grid.")
        return

    # Create a new figure for displaying all pieces.
    plt.figure(figsize=figsize)

    # Add each piece to the grid.
    for i, piece in enumerate(pieces):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(piece)
        plt.title(f"Piece {i}")
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
    plt.close()
