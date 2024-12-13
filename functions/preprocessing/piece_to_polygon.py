"""
piece_to_polygon.py - Extract a polygonal approximation from a puzzle piece image.
"""

import math
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from functions.utils.normalize_list import normalize_list


def piece_to_polygon(
    image_path: str,
    kernel_size: Tuple[int, int] = (5, 5),
    epsilon_ratio: float = 0.02,
    corner_distance_weight: float = 0.5,
    corner_angle_weight: float = 0.3,
    center_angle_weight: float = 0.2,
    display_steps: bool = True,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int], str]:
    """
    Extracts piece data from a puzzle image.

    Args:
        image_path (str): Path to the input puzzle image.

    Returns:
        List[np.ndarray]: A list of extracted piece data as image arrays.
    """
    # Reading image and converting to rgb and grayscale.
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a heavy Gaussian blur on grayscale image to reduce noise.
    kernel_size = kernel_size
    kernel_sigma = 5
    img_blurred = cv2.GaussianBlur(img_gray, kernel_size, kernel_sigma)

    # Create binary image (threshold to separate pieces from background).
    # Pixels with intensity > 250 are set to 0 (black), rest to 255 (white).
    # We use cv2.THRESH_BINARY_INV to invert the image for piece detection in the next
    # steps.
    threshold = 250
    max_value = 255
    _, img_binary = cv2.threshold(
        img_blurred, threshold, max_value, cv2.THRESH_BINARY_INV
    )

    # Finding contours of the puzzle piece using the binary image.
    # The setting cv2.RETR_EXTERNAL ensures that only the outermost contours
    # are returned (outline of the puzzle piece).
    # The setting cv2.CHAIN_APPROX_NONE ensures that all contour points are returned
    # (in other words, no compression is performed).
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Using cv2.moments to calculate the center of the largest contour.
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    piece_center = (cx, cy)

    # Draw contours and lines from center to each contour point on the original image.
    if display_steps:
        img_contours = img_rgb.copy()
        cv2.drawContours(
            img_contours, contours, -1, (255, 0, 0), 2
        )  # Draw contours in blue

        # Draw red lines from the center to each contour point
        for contour in contours:
            for point in contour:
                point_coords = tuple(point[0])  # Extract (x, y) coordinates
                cv2.line(img_contours, piece_center, point_coords, (255, 0, 0), 1)

        # Display the image with contours and lines.
        plt.figure(figsize=(10, 6))
        plt.imshow(img_contours)
        plt.title("Piece Contour with Lines to Center")
        plt.axis("off")
        plt.show()

    # We perform polygonal approximation on the contours to reduce the number of points
    # in each contour. This helps simplify the contour shape while preserving its
    # overall structure.
    # Initialize an empty list to store the approximated contours.
    approx_contours = []

    # Iterate over each contour detected in the binary image.
    for contour in contours:
        # Calculate the arc length (perimeter) of the current contour.
        # The second parameter, `True`, indicates that the contour is closed.
        arc_length = cv2.arcLength(contour, True)

        # Calculate the approximation accuracy using the `epsilon_ratio` parameter.
        # `epsilon` is a fraction of the arc length, determining how closely the
        # approximated contour should match the original contour.
        epsilon = epsilon_ratio * arc_length

        # Apply the Douglas-Peucker algorithm to approximate the contour.
        # This simplifies the contour by reducing the number of points while
        # ensuring the approximated contour stays within `epsilon` distance
        # of the original contour.
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Append the approximated contour to the list.
        approx_contours.append(approx)

    # Convert the list of approximated contours to a NumPy array for further processing.
    approx_contours = np.array(approx_contours)

    # Draw approximated contours and lines on the original image
    if display_steps:
        # Create a copy of the original RGB image to draw on
        img_contours = img_rgb.copy()

        # Loop through the approximated contours
        for approx in approx_contours:
            # Draw the contour with blue color
            cv2.drawContours(img_contours, [approx], -1, (255, 0, 0), 2)

            # Draw red lines from the center to each contour point
            for point in approx:
                point_coords = tuple(
                    point[0]
                )  # Extract the (x, y) coordinates from the contour point
                cv2.line(img_contours, piece_center, point_coords, (255, 0, 0), 4)

        # Display the image with approximated contours and lines
        plt.figure(figsize=(10, 6))
        plt.imshow(img_contours)
        plt.title("Approximated Contours with Lines to Center")
        plt.axis("off")
        plt.show()

    # Combine all slices into one 2D array.
    approx_contours = approx_contours.reshape(-1, 2)

    # Space to store coordinates, distances, and angles for each point on piece contour.
    contour_data = []

    # Looping through all points on piece contour.
    for _, point in enumerate(approx_contours):
        # Center & Contour 'Border' Points:
        center_X = piece_center[0]
        center_Y = piece_center[1]
        contour_X = point[0]
        contour_Y = point[1]

        # Delta Values:
        dX = contour_X - center_X
        dY = -(contour_Y - center_Y)

        # Distance Equation:
        dX2 = np.square(dX)
        dY2 = np.square(dY)
        distance_from_center = np.sqrt(dX2 + dY2)

        # Calculating Angle and Adjusting for Negative Values:
        angle = np.arctan2(dY, dX) * 180 / np.pi
        if angle < 0:
            angle += 360

        # Saving data.
        contour_data.append(
            {
                "coordinates": point,
                "distance": distance_from_center,
                "angle": angle,
            }
        )

    def points_are_equal(p1, p2):
        """Check if two points have the same coordinates."""
        return np.array_equal(p1["coordinates"], p2["coordinates"])

    def angle_between_points(p1, p2):
        """Calculate the angle between two points."""
        angle1 = p1["angle"]
        angle2 = p2["angle"]

        return (angle2 - angle1) % 360

    def angle_between_surrounding_points(contour_data, index):
        """Given the countour data and an index, find the angle directly between the
        two points surrounding it."""
        total_points = len(contour_data)
        previous_index = (index - 1 + total_points) % total_points
        next_index = (index + 1) % total_points

        # Get the center, previous, and next points
        center_point = contour_data[index]
        prev_point = contour_data[previous_index]
        next_point = contour_data[next_index]

        # Calculate vectors from center point to surrounding points
        prev_vector = (
            prev_point["coordinates"][0] - center_point["coordinates"][0],
            prev_point["coordinates"][1] - center_point["coordinates"][1],
        )
        next_vector = (
            next_point["coordinates"][0] - center_point["coordinates"][0],
            next_point["coordinates"][1] - center_point["coordinates"][1],
        )

        # Calculate the angle between these vectors:
        # Dot product
        dot_product = prev_vector[0] * next_vector[0] + prev_vector[1] * next_vector[1]

        # Magnitudes
        prev_magnitude = math.sqrt(prev_vector[0] ** 2 + prev_vector[1] ** 2)
        next_magnitude = math.sqrt(next_vector[0] ** 2 + next_vector[1] ** 2)

        # Cosine of the angle
        cos_angle = dot_product / (prev_magnitude * next_magnitude)

        # Calculate Angle
        angle_radians = math.acos(max(min(cos_angle, 1), -1))
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    # Save all corner pairs
    corner_pairs = []

    # Loop through filtered contour data.
    for target_index, target_point in enumerate(contour_data):
        # Find the point closest to 90 degrees away, excluding target_point.
        point_90_index, point_90 = min(
            enumerate(contour_data),
            key=lambda enum_point: (
                abs((enum_point[1]["angle"] - (target_point["angle"] + 90)) % 360)
                if not points_are_equal(enum_point[1], target_point)
                else float("inf")
            ),
        )

        # Find the point closest to 180 degrees away, excluding target_point and
        # point_90.
        point_180_index, point_180 = min(
            enumerate(contour_data),
            key=lambda enum_point: (
                abs((enum_point[1]["angle"] - (target_point["angle"] + 180)) % 360)
                if (
                    not points_are_equal(enum_point[1], target_point)
                    and not points_are_equal(enum_point[1], point_90)
                )
                else float("inf")
            ),
        )

        # Find the point closest to 270 degrees away, excluding target_point, point_90,
        # and point_180.
        point_270_index, point_270 = min(
            enumerate(contour_data),
            key=lambda enum_point: (
                abs((enum_point[1]["angle"] - (target_point["angle"] + 270)) % 360)
                if (
                    not points_are_equal(enum_point[1], target_point)
                    and not points_are_equal(enum_point[1], point_90)
                    and not points_are_equal(enum_point[1], point_180)
                )
                else float("inf")
            ),
        )

        # Using these four points, draw red dots on the original image for
        # visualization.
        selected_corners = [target_point, point_90, point_180, point_270]

        # Calculate total distance of selected corners from center.
        total_distance = sum(point["distance"] for point in selected_corners)

        # Calculate total angle error of selected corners.
        angle1 = angle_between_points(target_point, point_90)
        angle2 = angle_between_points(point_90, point_180)
        angle3 = angle_between_points(point_180, point_270)
        angle4 = angle_between_points(point_270, target_point)
        angle_error = (
            abs(angle1 - 90) + abs(angle2 - 90) + abs(angle3 - 90) + abs(angle4 - 90)
        )

        # Check surrounding points to each corner.
        corner_angle1 = angle_between_surrounding_points(contour_data, target_index)
        corner_angle2 = angle_between_surrounding_points(contour_data, point_90_index)
        corner_angle3 = angle_between_surrounding_points(contour_data, point_180_index)
        corner_angle4 = angle_between_surrounding_points(contour_data, point_270_index)
        corner_angle_error = (
            abs(corner_angle1 - 90)
            + abs(corner_angle2 - 90)
            + abs(corner_angle3 - 90)
            + abs(corner_angle4 - 90)
        )

        # Save corner pairs
        corner_pairs.append(
            {
                "total_distance": total_distance,
                "angle_error": angle_error,
                "corner_angle_error": corner_angle_error,
                "points": selected_corners,
            }
        )

    # Select the best corner pairs from total distance and angle error data
    # Assign weights to balance importance of distance and angle error
    distance_weight = corner_distance_weight
    corner_angle_error_weight = corner_angle_weight
    center_angle_error_weight = center_angle_weight

    # Extract distance and angle_error from corner_pairs
    distances = [pair["total_distance"] for pair in corner_pairs]
    angle_errors = [pair["angle_error"] for pair in corner_pairs]
    corner_angle_errors = [pair["corner_angle_error"] for pair in corner_pairs]

    # Normalize values
    normalized_distances = normalize_list(
        distances, reverse=True
    )  # Higher distance is better
    normalized_angle_errors = normalize_list(
        angle_errors
    )  # Lower angle error is better
    normalized_corner_angle_errors = normalize_list(
        corner_angle_errors
    )  # Lower corner angle error is better

    # Calculate composite scores
    for i, pair in enumerate(corner_pairs):
        pair["composite_score"] = (
            distance_weight * normalized_distances[i]
            + center_angle_error_weight * normalized_angle_errors[i]
            + corner_angle_error_weight * normalized_corner_angle_errors[i]
        )

    # Sort by composite score (higher is better in this example)
    sorted_corner_pairs = sorted(corner_pairs, key=lambda x: -x["composite_score"])

    # Select the best corner pair
    best_corner_estimates = sorted_corner_pairs[0]["points"]

    # Visualization of detected corners
    if display_steps:
        img_copy = img_rgb.copy()

        for point in best_corner_estimates:
            point_coords = point["coordinates"]

            # Draw a circle at the corner point
            cv2.circle(img_copy, tuple(point_coords), 5, (0, 0, 255), -1)

            # Draw a red line from the center to the corner point
            cv2.line(img_copy, piece_center, tuple(point_coords), (255, 0, 0), 4)

        plt.figure(figsize=(10, 6))
        plt.imshow(img_copy)
        plt.title("Image with Best Corners and Lines to Center")
        plt.axis("off")
        plt.show()

    # Initialize a counter for flat edges in the detected corners
    num_flat_edges = 0

    # Iterate through the best corner estimates to check for flat edges
    for i in range(len(best_corner_estimates)):
        # Set the current corner and the next corner (circularly)
        item1 = best_corner_estimates[i]
        item2 = best_corner_estimates[
            (i + 1) % len(best_corner_estimates)
        ]  # Wrap around at the end

        # Extract the angles for the current and next corners
        angle1, angle2 = item1["angle"], item2["angle"]

        # Find the index of the angles in the contour data
        index1 = next(
            (i for i, item in enumerate(contour_data) if item["angle"] == angle1), None
        )
        index2 = next(
            (i for i, item in enumerate(contour_data) if item["angle"] == angle2), None
        )

        # Calculate the distance between the two indices in the contour data
        distance = index2 - index1  # type: ignore

        # Special handling if we're at the end of the list (wraps to the start)
        if index1 == (len(contour_data) - 1) and index2 == 0:
            num_flat_edges += 1
            continue

        # If the distance is 1, this indicates consecutive points (flat edge)
        if distance == 1:
            num_flat_edges += 1

    # Classify the piece based on the number of flat edges detected
    classification = "NONE"
    if num_flat_edges == 2:
        classification = "C"  # Two flat edges = Corner
    elif num_flat_edges == 1:
        classification = "E"  # One flat edge = Edge
    elif num_flat_edges == 0:
        classification = "M"  # No flat edges = Middle
    else:
        classification = "ERROR"  # More than two flat edges is an error

    # Convert corner coordinates to integers
    corner_coords = [
        (int(point["coordinates"][0]), int(point["coordinates"][1]))
        for point in best_corner_estimates
    ]

    return corner_coords, piece_center, classification
