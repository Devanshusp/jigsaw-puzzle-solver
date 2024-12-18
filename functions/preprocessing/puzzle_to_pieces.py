"""
puzzle_to_pieces.py -  This script processes an image of a puzzle, extracts individual
pieces as separate components.
"""

from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from functions.utils.rotate_image import rotate_image


def puzzle_to_pieces(
    image_path: str,
    kernel_size: Tuple[int, int] = (5, 5),
    display_steps: bool = True,
    add_random_rotation: bool = False,
) -> List[np.ndarray]:
    """
    Processes an image of a puzzle to extract individual pieces as separate images.

    Args:
        image_path (str): Path to the puzzle image.
        kernel_size (Tuple[int, int], optional): Kernel size for Gaussian blurring.
            Defaults to (5, 5).
        display_steps (bool, optional): Flag to display intermediate steps.
        add_random_rotation (bool, optional): Flag to apply random rotation to
            puzzle pieces.

    Returns:
        List[np.ndarray]: List of extracted piece images.
    """
    # Read the image in BGR using OpenCV and convert it to RGB for Matplotlib.
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the source image.
    if display_steps:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title("Source Image")
        plt.show()

    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a heavy Gaussian blur on grayscale image to reduce noise.
    kernel_size = kernel_size
    kernel_sigma = 5
    img_blurred = cv2.GaussianBlur(img_gray, kernel_size, kernel_sigma)

    # Display the preprocessed blurred grayscale image.
    if display_steps:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_blurred, cmap="gray")
        plt.title("Blurred Grayscale Image")
        plt.show()

    # Create binary image (threshold to separate pieces from background).
    # Pixels with intensity > 250 are set to 0 (black), rest to 255 (white).
    # We use cv2.THRESH_BINARY_INV to invert the image for piece detection in the next
    # steps.
    threshold = 250
    max_value = 255
    _, img_binary = cv2.threshold(
        img_blurred, threshold, max_value, cv2.THRESH_BINARY_INV
    )

    # Display the binary image
    if display_steps:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_binary, cmap="gray")
        plt.title("Binary Image")
        plt.show()

    # Find connected components (the puzzle pieces).
    # Use connected component analysis to label distinct regions in the binary image.
    # `connvectivity` and only be 4 or 8. For 4, only the adjacent pixels (up, down,
    # left, right) are connected, while for 8, the diagonal pixels are also considered.
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_binary, connectivity=connectivity
    )

    # 'num_labels':
    #   int - Total number of connected components (including background at index 0).
    # 'labels':
    #   np.ndarray - A 2D array (same dimensions as input image) where each pixel is
    #   labeled with its component index.
    # 'stats':
    #   np.ndarray - A 2D array of shape (num_labels, 5) where each row corresponds to
    #   a component with statistics: [x, y, width, height, area].
    # 'centroids':
    #   np.ndarray - A 2D array of shape (num_labels, 2) where each row contains the
    #   (x, y) centroid of a component.

    print("Number of connected components:", num_labels)

    # Initialize a list to store the extracted puzzle pieces and loop through each
    # label, skipping 0 (background label).
    pieces = []
    for label_index in range(1, num_labels):
        # Create a binary mask for the current piece label from `labels`.
        img_mask = labels == label_index

        # Use the mask to isolate the current piece from the original image by setting
        # all pixels outside the current piece to white.
        img_piece = img_rgb.copy()
        img_piece[img_mask == 0] = [255, 255, 255]

        # Extract the bounding box for the current piece from `stats`.
        x, y, w, h = stats[label_index][:4]  # x (left), y (top), w (width), h (height)

        # Crop the isolated piece to its bounding box.
        horizontal_indices = slice(y, y + h)
        vertical_indices = slice(x, x + w)
        img_piece_cropped = img_piece[horizontal_indices, vertical_indices]

        if add_random_rotation:
            # Add random rotation to the cropped image around its center.
            # np.random.seed(100)
            angle = np.random.randint(0, 360)
            img_piece_cropped = rotate_image(img_piece_cropped, angle)

        # Save the current piece to the list of extracted pieces.
        pieces.append(img_piece_cropped)

    return pieces
