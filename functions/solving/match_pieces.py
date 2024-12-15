from typing import Literal

import numpy as np
from numpy.linalg import norm
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff

from functions.utils.orient_piece_contours import orient_piece_contours


def fourier_descriptor_similarity(contour1, contour2):
    """
    Compare contours using Fourier Descriptors.
    """

    def compute_fourier_descriptors(contour):
        contour = contour.astype(np.float32)
        contour_complex = contour[:, 0] + 1j * contour[:, 1]

        descriptors = np.fft.fft(contour_complex)

        return np.abs(descriptors[:10])

    desc1 = compute_fourier_descriptors(contour1)
    desc2 = compute_fourier_descriptors(contour2)

    return np.linalg.norm(desc1 - desc2)


def cosine_similarity(contour1, contour2, num_points=1000):
    """
    Compute Cosine Similarity between two resampled contours.
    """
    contour1_resampled = resample_contour(np.array(contour1), num_points)
    contour2_resampled = resample_contour(np.array(contour2), num_points)

    # Flatten the contours into 1D arrays
    contour1_flat = contour1_resampled.flatten()
    contour2_flat = contour2_resampled.flatten()

    # Compute the cosine similarity
    cosine_sim = np.dot(contour1_flat, contour2_flat) / (
        norm(contour1_flat) * norm(contour2_flat)
    )

    return cosine_sim


def procrustes_shape_similarity(contour1, contour2):
    """
    Use Procrustes analysis to compare shapes after alignment.
    """
    contour1 = resample_contour(np.array(contour1))
    contour2 = resample_contour(np.array(contour2))

    _, new_contour1, new_contour2 = procrustes(contour1, contour2)

    return np.mean(np.sqrt(np.sum((new_contour1 - new_contour2) ** 2, axis=1)))


def directed_hausdorff_distance(contour1, contour2):
    """
    Compute the directed Hausdorff distance between two contours.
    """
    forward_dist = directed_hausdorff(contour1, contour2)[0]
    reverse_dist = directed_hausdorff(contour2, contour1)[0]

    return max(forward_dist, reverse_dist)


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


def resample_contour(contour, num_points=1000):
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
) -> dict:
    """
    Calculate the match score between two pieces using various shape comparison metrics.
    Returns a dictionary of labeled similarity values.
    """
    piece1_oriented_contours = orient_piece_contours(pieces_path, piece1, side1, "Top")
    piece2_oriented_contours = orient_piece_contours(pieces_path, piece2, side2, "Top")

    # Adjust resampling accuracy based on complexity
    adjusted_contours = adjust_contour_accuracy(
        piece1_oriented_contours, piece2_oriented_contours, complexity
    )
    contour1_resampled = adjusted_contours["contour1_resampled"]
    contour2_resampled = adjusted_contours["contour2_resampled"]

    # Compute various distance measures and similarities
    hausdorff_dist = directed_hausdorff_distance(
        piece1_oriented_contours, piece2_oriented_contours
    )
    fourier_similarity = fourier_descriptor_similarity(
        contour1_resampled, contour2_resampled
    )
    procrustes_sim = procrustes_shape_similarity(contour1_resampled, contour2_resampled)
    cosine_sim = cosine_similarity(contour1_resampled, contour2_resampled)

    # Return results in a dictionary
    return {
        "Hausdorff Distance": hausdorff_dist,
        "Fourier Descriptor Similarity": fourier_similarity,
        "Procrustes Shape Similarity": procrustes_sim,
        "Cosine Similarity": cosine_sim,
    }
