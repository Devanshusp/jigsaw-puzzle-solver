"""
visualize_solution.py - Visualize a list of puzzle piece images in a grid layout with
associated data.
"""

from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt

from functions.utils.orient_piece_contours import calculate_rotation_angle, rotate_point
from functions.utils.save_data import get_data_for_piece


def visualize_solution(pieces_path: str, solution_matrix: List[List[dict | None]]):
    num_rows = len(solution_matrix)
    num_cols = len(solution_matrix[0]) if num_rows > 0 else 0

    # Create subplots, this will return a single Axes object if num_rows == 1 or
    # num_cols == 1
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    # If there is only one row or column, ax is a single Axes object, not an array
    if num_rows == 1 and num_cols == 1:
        ax = np.array([[ax]])  # Make it 2D for consistency
    elif num_rows == 1:
        ax = np.array([ax])  # Make it a 2D array with one row
    elif num_cols == 1:
        ax = np.array([[a] for a in ax])  # Make it a 2D array with one column

    for i, row in enumerate(solution_matrix):
        for j, piece_info in enumerate(row):
            if piece_info is None:
                ax[i, j].axis("off")
                continue

            piece_index = piece_info["piece_index"]
            piece_top_side = piece_info["piece_top_side"]

            piece_img = cv2.imread(f"{pieces_path}/piece_{piece_index}.png")
            piece_img = cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB)

            piece_center_coords = get_data_for_piece(
                pieces_path, piece_index, "center_coords"
            )
            piece_side_data = get_data_for_piece(
                pieces_path, piece_index, "piece_side_data"
            )
            piece_top_coords = piece_side_data[piece_top_side]["points"]

            rotation_angle_deg = calculate_rotation_angle(piece_top_coords)
            temp_rotated_contours = [
                rotate_point(pt, piece_center_coords, np.radians(rotation_angle_deg))
                for pt in piece_top_coords
            ]

            if temp_rotated_contours[0][1] > piece_center_coords[1]:
                rotation_angle_deg += 180

            # Calculate the new bounding box for the rotated image
            (h, w) = piece_img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D(
                tuple(piece_center_coords), -rotation_angle_deg, 1.0
            )

            # Get the size of the new rotated image bounding box
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])

            # Calculate the new width and height based on the rotation matrix
            new_w = int(h * abs_sin + w * abs_cos)
            new_h = int(h * abs_cos + w * abs_sin)

            # Adjust the rotation matrix to consider the full rotated image
            rotation_matrix[0, 2] += (new_w / 2) - piece_center_coords[0]
            rotation_matrix[1, 2] += (new_h / 2) - piece_center_coords[1]

            # Rotate the image without cropping
            rotated_img = cv2.warpAffine(
                piece_img,
                rotation_matrix,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # Adjust the piece center coordinates to reflect the new center of the
            # rotated image
            new_piece_center_x = new_w / 2
            new_piece_center_y = new_h / 2
            piece_center_coords = (new_piece_center_x, new_piece_center_y)

            ax[i, j].imshow(rotated_img)
            ax[i, j].plot(*piece_center_coords, "ro", markersize=8)
            ax[i, j].axis("off")

    plt.tight_layout()
    plt.show()
