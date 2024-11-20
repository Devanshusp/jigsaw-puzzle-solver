"""
diplay_pieces.py - Display a list of puzzle piece images in a grid layout.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def display_pieces(
    pieces: List[np.ndarray], figsize: tuple = (10, 10), save_name: str | None = None
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
    rows = int(np.ceil(np.sqrt(num_pieces)))
    cols = int(np.ceil(num_pieces / rows))

    # Create a new figure for displaying all pieces.
    plt.figure(figsize=figsize)

    # Add each piece to the grid.
    for i, piece in enumerate(pieces):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(piece)
        plt.title(f"Piece {i+1}")
        plt.axis("off")

    # Adjust layout to avoid overlap.
    plt.tight_layout()

    # Save the figure as an image file if needed.
    if save_name is not None:
        plt.savefig(f"{save_name}.png", dpi=300)

    # Show the plot.
    plt.show()

    # Clear the figure after saving and showing to avoid issues.
    plt.clf()
