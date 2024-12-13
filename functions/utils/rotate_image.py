"""
rotate_image.py - Utility function to rotate an image by a specified angle around
specified center point.
"""

import cv2


def rotate_image(image, angle):
    """
    Rotate an image around its center without cropping or scaling,
    expanding the canvas as needed.

    Args:
        image (np.ndarray): The image to be rotated.
        angle (float): The angle by which the image will be rotated.

    Returns:
        np.ndarray: The rotated image with an expanded canvas.
    """
    h, w = image.shape[:2]
    # Compute the center of the image
    cx, cy = w // 2, h // 2

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Calculate the new bounding box size
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    # New width and height of the rotated image
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # Create a white background and rotate the image
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

    return rotated
