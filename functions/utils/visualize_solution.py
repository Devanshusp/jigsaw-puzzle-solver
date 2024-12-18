"""
visualize_solution.py - Visualize full puzzle pieces with minimal boundary overlap
"""

from typing import List

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from functions.utils.orient_piece_contours import calculate_rotation_angle, rotate_point
from functions.utils.save_data import get_data_for_piece


def visualize_solution(
    pieces_path: str,
    solution_matrix: List[List[dict | None]],
    save_name: str | None = None,
    save: bool = False,
    display_steps: bool = True,
) -> None:
    """
    Visualize the solution matrix as a grid of puzzle pieces with minimal boundary
        overlap.

    Args:
        pieces_path (str): Path to the directory containing puzzle pieces.
        solution_matrix (List[List[dict | None]]): The solution matrix.
        save_name (str | None, optional): Name of the saved image. Defaults to None.
        save (bool, optional): Flag to save the solution. Defaults to False.
        display_steps (bool, optional): Flag to display intermediate steps.
            Defaults to True.

    Returns:
        None
    """
    num_rows = len(solution_matrix)
    num_cols = len(solution_matrix[0]) if num_rows > 0 else 0

    # Restrict figure size to a maximum of 8x8 inches
    max_width, max_height = 8, 8
    fig_width = min(num_cols * 2.5, max_width)
    fig_height = min(num_rows * 2.5, max_height)

    _ = plt.figure(
        figsize=(fig_width, fig_height),
        facecolor="white",
        constrained_layout=True,
    )

    gs = gridspec.GridSpec(num_rows, num_cols, wspace=-0.1, hspace=-0.1)

    for i, row in enumerate(solution_matrix):
        for j, piece_info in enumerate(row):
            ax = plt.subplot(gs[i, j])

            if piece_info is None:
                ax.axis("off")
                continue

            piece_index = piece_info["piece_index"]
            piece_top_side = piece_info["piece_top_side"]

            # Read original image
            piece_img = cv2.imread(
                f"{pieces_path}/piece_{piece_index}.png", cv2.IMREAD_UNCHANGED
            )

            # Convert color spaces
            if piece_img.shape[2] == 3:
                piece_img = cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB)
            elif piece_img.shape[2] == 4:
                piece_img = cv2.cvtColor(piece_img, cv2.COLOR_BGRA2RGBA)

            # Get piece data
            piece_center_coords = get_data_for_piece(
                pieces_path, piece_index, "center_coords"
            )
            piece_side_data = get_data_for_piece(
                pieces_path, piece_index, "piece_side_data"
            )
            piece_top_coords = piece_side_data[piece_top_side]["points"]

            # Calculate rotation angle
            rotation_angle_deg = calculate_rotation_angle(piece_top_coords)
            temp_rotated_top_contours = [
                rotate_point(pt, piece_center_coords, np.radians(rotation_angle_deg))
                for pt in piece_top_coords
            ]

            if temp_rotated_top_contours[0][1] > piece_center_coords[1]:
                rotation_angle_deg += 180

            # Calculate rotation matrix
            (h, w) = piece_img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D(
                tuple(piece_center_coords), -rotation_angle_deg, 1.0
            )

            # Compute new image dimensions to avoid cropping
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            new_w = int(h * abs_sin + w * abs_cos)
            new_h = int(h * abs_cos + w * abs_sin)

            # Adjust rotation matrix to center the image
            rotation_matrix[0, 2] += (new_w / 2) - piece_center_coords[0]
            rotation_matrix[1, 2] += (new_h / 2) - piece_center_coords[1]

            # Rotate the entire image without cropping
            rotated_img = cv2.warpAffine(
                piece_img,
                rotation_matrix,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255, 0),
            )

            ax.imshow(rotated_img)
            ax.axis("off")

    # Remove extra white space and pack tightly
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if save:
        plt.savefig(f"{save_name}.png", dpi=300)

    if display_steps:
        plt.show()

    plt.clf()
    plt.close()
