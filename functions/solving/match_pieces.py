from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from functions.utils.orient_piece_contours import orient_piece_contours
from functions.utils.save_data import get_data_for_piece


def hausdorff_color(color1, color2):
    pixels1 = color1["color_strip"]
    pixels2 = color2["color_strip"]

    pixels1 = np.array(pixels1)
    pixels2 = np.array(pixels2)

    # Compute Hausdorff distances
    fHC = directed_hausdorff(pixels1, pixels2)[0]
    bHC = directed_hausdorff(pixels2, pixels1)[0]

    # Return the max
    return max(fHC, bHC)


def directed_hausdorff_distance(contour1, contour2):
    """
    Compute the directed Hausdorff distance between two contours.
    """
    forward_dist = directed_hausdorff(contour1, contour2)[0]
    reverse_dist = directed_hausdorff(contour2, contour1)[0]

    return max(forward_dist, reverse_dist)


def hu_moments_distance(contour1, contour2):
    """
    Compute the directed Hausdorff distance between two contours.
    """
    similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)

    return similarity


def corners_similarity(contour1, contour2):
    """
    Compare length of contours from corner to corner using Euclidean Distance.
    """
    edge_lengths = []

    # Calculate the length of the contour
    for contour in [contour1, contour2]:
        start = 0
        end = len(contour) - 1

        # Start and end points of the contour
        x1, y1 = contour[start]
        x2, y2 = contour[end]

        # Euclidean distance of the contour
        distance = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
        edge_lengths.append(distance)

    # Difference between the two contour lengths
    difference = abs(edge_lengths[0] - edge_lengths[1])

    return difference


def adjust_contour_accuracy(contour1, contour2, complexity="high"):
    """
    Adjust the accuracy of contour resampling based on the complexity of the contour.
    """
    if complexity == "high":
        # Use more points for more complex contours
        num_points = 2000
    elif complexity == "medium":
        num_points = 1500
    else:
        # Default for simple contours
        num_points = 1000

    return {
        "contour1_resampled": resample_contour(np.array(contour1), num_points),
        "contour2_resampled": resample_contour(np.array(contour2), num_points),
        "num_points": num_points,
    }


def resample_contour(contour, num_points=1000) -> np.ndarray:
    """
    Resample contour to a fixed number of points using linear interpolation.
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
    """
    # Create a figure with 4 subplots: 3 for contours and 1 for the table
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Contour Comparison between {piece1_name} & {piece2_name}",
        fontsize=16,
    )

    # Flatten the axes for easier indexing
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

    # Plot contours together
    axes[2].plot(*zip(*piece1_contour), "b-o", label=piece1_name)
    axes[2].plot(*zip(*piece2_contour), "g-o", label=piece2_name)
    axes[2].set_title("Overlayed Contours")
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")
    axes[2].legend()
    axes[2].grid(True)
    axes[2].axis("equal")
    axes[2].invert_yaxis()

    # Display error metrics as a table in the last subplot
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

    # Turn off axis for the table subplot
    axes[3].axis("off")

    # Create the table
    table = axes[3].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[3].set_title("Comparison Metrics", fontsize=12)

    # Adjust layout and display
    plt.tight_layout()
    plt.show(block=True)

    # Close the figure
    plt.close()

    return results


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
