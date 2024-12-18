"""
color_data.py - Extract colors along an edge of a puzzle piece.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from functions.utils.save_data import get_data_for_piece


def piece_color_data(
    piece_path: str,
    piece_number: int,
    piece_side: str,
    display_steps: bool = True,
) -> dict:
    """
    Extract colors along an edge of a puzzle piece.

    Args:
        piece_path (str): Path to the pieces data
        piece_number (int): Index of the piece
        piece_side (str): Side of the piece
        display_steps (bool, optional): Flag to display intermediate steps.
            Defaults to True.

    Returns:
        dict: Dictionary of color data
    """
    piece_img = cv2.imread(piece_path + f"/piece_{piece_number}.png")

    # Initialize save dictionary
    color_data = {}

    # Get contours and centroid data for the piece
    piece_side_data = get_data_for_piece(piece_path, piece_number, "piece_side_data")
    piece_side_contours = piece_side_data[piece_side]["contour_points"]
    piece_side_centroid = get_data_for_piece(piece_path, piece_number, "center_coords")

    # Convert the contours and centroid data to numpy arrays
    piece_side_contours = np.array(piece_side_contours)
    piece_side_centroid = np.array(piece_side_centroid)

    # Compute the gradients along the contour
    dx = np.gradient(piece_side_contours[:, 0])
    dy = np.gradient(piece_side_contours[:, 1])

    # Compute the normals by rotating the gradient vectors 90 degrees
    normals = np.stack((-dy, dx), axis=-1)

    # Normalize the normals to unit vectors
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Define the inward movement distance
    inward_pixels = 3  # len(piece_side_contours) * 0.1

    # Move contour points inward along their normals
    contours = piece_side_contours - normals * inward_pixels

    # Ensure the points stay within image bounds
    contours = np.clip(
        contours, [0, 0], [piece_img.shape[1] - 1, piece_img.shape[0] - 1]
    )

    # Extract color data along the contour
    colors = []
    for x, y in contours:
        x = int(x)
        y = int(y)
        if 0 <= y < piece_img.shape[0] and 0 <= x < piece_img.shape[1]:
            color = piece_img[y, x]
            colors.append(color)

    # Visualize the extracted colors as a color strip
    color_strip = np.array(colors)
    R, G, B = color_strip[:, 0], color_strip[:, 1], color_strip[:, 2]

    def summary_statistics(channel):
        return {
            "mean": int(np.mean(channel)),
            "median": int(np.median(channel)),
            "range": (int(np.min(channel)), int(np.max(channel))),
        }

    # Store summary statistics for each channel
    R_stats = summary_statistics(R)
    G_stats = summary_statistics(G)
    B_stats = summary_statistics(B)

    color_strip_int = color_strip.astype(int)
    color_strip_list = color_strip_int.tolist()

    color_data = {
        "red": R_stats,
        "green": G_stats,
        "blue": B_stats,
        "color_strip": color_strip_list,
    }

    if display_steps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        piece_img = cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB)

        ax1.imshow(piece_img)
        ax1.scatter(contours[:, 0], contours[:, 1], color="red", s=10)
        ax1.set_title("Piece Image with Contour")
        ax1.axis("off")

        strip_height = 50
        thick_color_strip = np.tile(color_strip[np.newaxis, :, :], (strip_height, 1, 1))

        thick_color_strip = cv2.cvtColor(thick_color_strip, cv2.COLOR_BGR2RGB)

        ax2.imshow(thick_color_strip)
        ax2.set_title("Color Strip")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    return color_data
