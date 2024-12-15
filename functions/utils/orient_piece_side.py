"""
orient_piece_side.py - Utility function to rotate a piece's side and get its contours.
"""

import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from functions.utils.save_data import get_data_for_piece


def orient_piece_side(
    piece_path: str,
    piece_index: int,
    side: Literal["A", "B", "C", "D"],
    orientation: Literal["Top", "Bottom"],
    visualize: bool = False,
):
    """
    Rotate a piece's side and get its contours.

    Args:
        piece_path (str): The path to the piece data JSON file.
        piece_index (int): The index of the piece to be rotated.
        side (Literal["A", "B", "C", "D"]): The side of the piece to be rotated.
        orientation (Literal["Top", "Bottom"]): The orientation of the piece to be
            rotated.
        visualize (bool, optional): Whether to visualize the rotated piece. Default is
            False.

    Returns:
        List[List[float]]: The contours of the rotated piece.
    """
    # Extract piece data
    side_data = get_data_for_piece(piece_path, piece_index, "piece_side_data")[side]
    center_coords = get_data_for_piece(piece_path, piece_index, "center_coords")

    # Extract side data
    side_points = side_data["points"]
    side_contours = side_data["contour_points"]

    # Calculate rotation angle necessary to orient piece correctly
    rotation_angle_deg = calculate_rotation_angle(side_points)
    rotation_angle_rad = np.radians(rotation_angle_deg)

    # Rotate contours by calculated degrees (pi/4 radians)
    rotated_contours = [
        rotate_point(pt, center_coords, rotation_angle_rad) for pt in side_contours
    ]

    # If the piece must be oriented to the top, rotate it if not already
    if orientation.lower() == "top" and rotated_contours[0][1] > center_coords[1]:
        rotated_contours = [
            rotate_point(pt, center_coords, math.pi) for pt in rotated_contours
        ]

    # If the piece must be oriented to the bottom, rotate it if not already
    if orientation.lower() == "bottom" and rotated_contours[0][1] < center_coords[1]:
        rotated_contours = [
            rotate_point(pt, center_coords, math.pi) for pt in rotated_contours
        ]

    # Normalize contours to start at (0, 0)
    # Find the minimum x and y coordinates
    min_x = min(pt[0] for pt in rotated_contours)
    min_y = min(pt[1] for pt in rotated_contours)

    # Shift all points to start at (0, 0)
    normalized_contours = [(pt[0] - min_x, pt[1] - min_y) for pt in rotated_contours]

    if visualize:
        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot original
        plot_piece(
            axes[0],
            f"Original Side: {side} (Orientation: {orientation})",
            center_coords,
            side_contours,
            "g",
        )

        # Plot rotated
        plot_piece(
            axes[1],
            f"Rotated Side ({rotation_angle_deg}): {side} (Orientation: {orientation})",
            center_coords,
            rotated_contours,
            "b",
        )

        plt.show()

    return normalized_contours


def rotate_point(point, center, angle_rad):
    """Rotates a point around a center by a given angle in radians."""
    # Calculate vector
    transform = point[0] - center[0], point[1] - center[1]

    # Rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Apply rotation matrix
    rotated = np.dot(transform, rotation_matrix.T).astype(int)
    rotated += center
    return rotated[0], rotated[1]


def calculate_rotation_angle(points):
    # Extract points
    (x1, y1), (x2, y2) = points

    # Calculate angle in radians
    delta_x = x2 - x1
    delta_y = y2 - y1

    # Handle vertical lines
    if delta_x == 0:
        return -90 if delta_y > 0 else 90

    # Calculate angle and convert to degrees
    theta_rad = np.arctan2(delta_y, delta_x)
    theta_deg = np.degrees(theta_rad)

    # Return negative to rotate to the x-axis
    return -theta_deg


def plot_piece(ax, title, center_coords, contours, label_color):
    """Utility to plot a piece on a given axis."""
    ax.plot(center_coords[0], center_coords[1], "ro", label="Center")
    contour_x, contour_y = zip(*contours)
    ax.plot(contour_x, contour_y, f"{label_color}o-", label="Contours")
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    ax.invert_yaxis()  # Flip y-axis for top-left origin
