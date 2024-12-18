"""
match_pieces.py - This file contains the logic to match two pieces using various shape
and color comparison metrics.
"""

from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from functions.utils.orient_piece_contours import orient_piece_contours
from functions.utils.save_data import get_data_for_piece


def hausdorff_color(color1, color2):
    """
    Compute the directed Hausdorff distance between two color strips.

    Args:
        color1: A dictionary containing the color strip data.
        color2: A dictionary containing the color strip data.

    Returns:
        The maximum of the forward and backward Hausdorff distances.
    """
    pixels1 = color1["color_strip"]
    pixels2 = color2["color_strip"]

    pixels1 = np.array(pixels1)
    pixels2 = np.array(pixels2)

    forward_dist = directed_hausdorff(pixels1, pixels2)[0]
    reverse_dist = directed_hausdorff(pixels2, pixels1)[0]

    return max(forward_dist, reverse_dist)


def directed_hausdorff_distance(contour1, contour2):
    """
    Compute the directed Hausdorff distance between two contours.

    Args:
        contour1: A list of points representing the first contour.
        contour2: A list of points representing the second contour.

    Returns:
        The maximum of the forward and reverse Hausdorff distances.
    """
    forward_dist = directed_hausdorff(contour1, contour2)[0]
    reverse_dist = directed_hausdorff(contour2, contour1)[0]

    return max(forward_dist, reverse_dist)


def hu_moments_distance(contour1, contour2):
    """
    Compute the directed Hausdorff distance between two contours (using cv2).

    Args:
        contour1: A list of points representing the first contour.
        contour2: A list of points representing the second contour.

    Returns:
        The maximum of the forward and reverse Hausdorff distances.
    """
    similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)

    return similarity


def corners_similarity(contour1, contour2):
    """
    Compare length of contours from corner to corner using Euclidean Distance.

    Args:
        contour1: A list of points representing the first contour.
        contour2: A list of points representing the second contour.

    Returns:
        The difference in length between the two contours.
    """
    edge_lengths = []

    for contour in [contour1, contour2]:
        start = 0
        end = len(contour) - 1

        x1, y1 = contour[start]
        x2, y2 = contour[end]

        distance = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
        edge_lengths.append(distance)

    difference = abs(edge_lengths[0] - edge_lengths[1])

    return difference


def adjust_contour_accuracy(contour1, contour2, complexity="high"):
    """
    Adjust the accuracy of contour resampling based on the complexity of the contour.

    Args:
        contour1: A list of points representing the first contour.
        contour2: A list of points representing the second contour.
        complexity: A string indicating the complexity of the contour.
            - "high": Use more points for more complex contours
            - "medium": Use more points for medium complex contours
            - "low": Use more points for simple contours

    Returns:
        A dictionary containing the resampled contours and the number of points used.
    """
    if complexity == "high":
        num_points = 2000
    elif complexity == "medium":
        num_points = 1500
    else:
        num_points = 1000

    return {
        "contour1_resampled": resample_contour(np.array(contour1), num_points),
        "contour2_resampled": resample_contour(np.array(contour2), num_points),
        "num_points": num_points,
    }


def resample_contour(contour, num_points=1000) -> np.ndarray:
    """
    Resample contour to a fixed number of points using linear interpolation.

    Args:
        contour: A numpy array representing the contour to resample.
        num_points: The number of points to resample the contour to.

    Returns:
        A numpy array representing the resampled contour.
    """
    distances = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
    distances = np.concatenate(([0], distances))

    total_length = distances[-1]
    normalized_distances = distances / total_length

    target_distances = np.linspace(0, 1, num_points)

    resampled_contour = np.zeros((num_points, 2))
    for dim in range(2):
        resampled_contour[:, dim] = np.interp(
            target_distances, normalized_distances, contour[:, dim]
        )

    return resampled_contour


def match_pieces(
    pieces_path: str,
    piece1: int,
    piece2: int,
    side1: Literal["A", "B", "C", "D"],
    side2: Literal["A", "B", "C", "D"],
    complexity: Literal["high", "medium", "low"] = "high",
    display_steps: bool = True,
) -> dict:
    """
    Calculate the match score between two pieces using various shape comparison metrics.
    Returns a dictionary of labeled similarity values.

    Args:
        pieces_path (str): Path to the pieces data JSON file.
        piece1 (int): Index of the first piece.
        piece2 (int): Index of the second piece.
        side1 (Literal["A", "B", "C", "D"]): Side of the first piece.
        side2 (Literal["A", "B", "C", "D"]): Side of the second piece.
        complexity (Literal["high", "medium", "low"], optional): Complexity of the
            contour. Defaults to "high".
        display_steps (bool, optional): Whether to display intermediate steps. Defaults
            to True.

    Returns:
        dict: A dictionary of labeled similarity values.
    """
    piece1_oriented_contours = orient_piece_contours(pieces_path, piece1, side1, "Top")
    piece2_oriented_contours = orient_piece_contours(
        pieces_path, piece2, side2, "Bottom"
    )

    # Adjust resampling accuracy based on complexity
    adjusted_contours = adjust_contour_accuracy(
        piece1_oriented_contours, piece2_oriented_contours, complexity
    )
    contour1_resampled = adjusted_contours["contour1_resampled"]
    contour2_resampled = adjusted_contours["contour2_resampled"]

    # Compute various distance measures and similarities
    hausdorff_dist = directed_hausdorff_distance(contour1_resampled, contour2_resampled)
    hu_moments_dist = hu_moments_distance(contour1_resampled, contour2_resampled)
    corners_dist = corners_similarity(contour1_resampled, contour2_resampled)

    # Extract color data for color matching score
    color_data1 = get_data_for_piece(pieces_path, piece1, "piece_side_data")[side1][
        "color_data"
    ]
    color_data2 = get_data_for_piece(pieces_path, piece2, "piece_side_data")[side2][
        "color_data"
    ]

    # Calculate color Difference
    color_difference = hausdorff_color(color_data1, color_data2)

    # Return results in a dictionary
    results = {
        "Hausdorff Distance": hausdorff_dist,
        "Hu Moments Distance": hu_moments_dist,
        "Corners Distance": corners_dist,
        "Color Difference": color_difference,
    }

    # Visualize comparison results using resampled contours
    if display_steps:
        return visualize_comparison_results(
            contour1_resampled,
            contour2_resampled,
            results,
            str(piece1) + side1,
            str(piece2) + side2,
        )

    return results


def visualize_comparison_results(
    piece1_contour,
    piece2_contour,
    results,
    piece1_name="Piece 1",
    piece2_name="Piece 2",
):
    """
    Visualize contours and comparison results in a single figure.

    Args:
        piece1_contour: The contour of the first piece.
        piece2_contour: The contour of the second piece.
        results: A dictionary of comparison results.
        piece1_name: The name of the first piece (default is "Piece 1").
        piece2_name: The name of the second piece (default is "Piece 2").

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Contour Comparison between {piece1_name} & {piece2_name}",
        fontsize=16,
    )
    axes = axes.flatten()

    # Plot original piece 1 contour
    plot_piece(
        axes[0],
        f"{piece1_name} Contour: ({len(piece1_contour)} points)",
        [0, 0],
        piece1_contour,
        "b",
    )

    # Plot original piece 2 contour
    plot_piece(
        axes[1],
        f"{piece2_name} Contour: ({len(piece2_contour)} points)",
        [0, 0],
        piece2_contour,
        "g",
    )

    axes[2].plot(*zip(*piece1_contour), "b-o", label=piece1_name)
    axes[2].plot(*zip(*piece2_contour), "g-o", label=piece2_name)
    axes[2].set_title("Overlayed Contours")
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")
    axes[2].legend()
    axes[2].grid(True)
    axes[2].axis("equal")
    axes[2].invert_yaxis()

    table_data = [
        ["Metric", "Value", "Interpretation"],
        [
            "Hausdorff Distance",
            f"{results['Hausdorff Distance']:.4f}",
            "Lower is better",
        ],
        ["Corners Distance", f"{results['Corners Distance']:.4f}", "Lower is better"],
        ["Color Difference", f"{results['Color Difference']:.4f}", "Lower is better"],
    ]

    axes[3].axis("off")
    table = axes[3].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[3].set_title("Comparison Metrics", fontsize=12)

    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    return results


def plot_piece(ax, title, center_coords, contours, label_color):
    """
    Utility to plot a piece on a given axis.

    Args:
        ax: The axis to plot on.
        title: The title of the plot.
        center_coords: The coordinates of the center of the piece.
        contours: The contours of the piece.
        label_color: The color to use for the labels.

    Returns:
        None
    """
    ax.plot(center_coords[0], center_coords[1], "ro", label="Center")
    contour_x, contour_y = zip(*contours)
    ax.plot(contour_x, contour_y, f"{label_color}o-", label="Contours")
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    # Flip y-axis for top-left origin
    ax.invert_yaxis()
