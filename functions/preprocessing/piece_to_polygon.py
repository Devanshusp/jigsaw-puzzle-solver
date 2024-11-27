"""
piece_to_polygon.py - Extract a polygonal approximation from a puzzle piece image.
"""

from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def piece_to_polygon(image_path: str) -> None:
    """
    Extracts piece data from a puzzle image.

    Args:
        image_path (str): Path to the input puzzle image.

    Returns:
        List[np.ndarray]: A list of extracted piece data as image arrays.
    """
    # 1. takes in a puzzle .png
    # 2. approximates its shape with a poly approximation
    # 3. this data must be saved lol
    # 3. detects corners
    # 4. labels each piece as "corner", "edge", or "center"
