"""
piece_to_center.py - This script processes an image of a puzzle piece, thresholds it to
isolate non-white pixels, and computes the center of these pixels (weighted center of
mass).
"""

import cv2
import matplotlib.pyplot as plt


def piece_to_center(image_path: str) -> None:
    """
    Computes the weighted center of non-white pixels in a puzzle piece image and
    displays it.

    Args:
        image_path (str): Path to the input puzzle piece image.

    Returns:
        None
    """
    # Read the image in grayscale (simplifies processing by reducing to 1 channel).
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to create a binary mask:
    # Pixels > 254 are set to 0 (white background); others set to 255 (piece area).
    _, binary = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)

    # Compute image moments from the binary mask to calculate the weighted center.
    moments = cv2.moments(binary)

    # Moments explanation:
    # 'm00': Area of the white region (sum of pixel values).
    # 'm10', 'm01': Spatial moments used to calculate centroid.

    # If the area (m00) is non-zero, calculate center of mass:
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])  # x-coordinate of the center
        center_y = int(moments["m01"] / moments["m00"])  # y-coordinate of the center
    else:
        center_x, center_y = 0, 0  # Avoid division by zero if m00 is zero

    # Overlay a red dot on the original image at the computed center.
    img_center = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
    cv2.circle(img_center, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

    # Plot all results in one figure using subplots
    _, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Thresholded binary image
    axs[0].imshow(binary, cmap="gray")
    axs[0].set_title("Thresholded Binary Image")
    axs[0].axis("off")

    # Original image with the weighted center marked
    axs[1].imshow(cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Weighted Center (Red Dot)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
image_path = "images/pieces/aurora30/piece_11.png"  # Replace with your image path
piece_to_center(image_path)
