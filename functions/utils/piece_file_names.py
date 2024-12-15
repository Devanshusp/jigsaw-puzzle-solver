"""
piece_file_names.py - Utility functions to get the names of puzzle pieces.
"""

import os
import re


def get_piece_file_names(PIECES_SAVE_PATH, piece_pattern: re.Pattern) -> list:
    """
    Returns a list of file names for puzzle pieces in a given directory.

    Args:
        PIECES_SAVE_PATH (str): The path to the directory containing the puzzle pieces.

    Returns:
        list: A list of file names for puzzle pieces.
    """
    # List to store matching file paths
    piece_files = []

    # Loop through all files in the directory
    for filename in os.listdir(PIECES_SAVE_PATH):
        if piece_pattern.match(filename):
            piece_files.append(os.path.join(PIECES_SAVE_PATH, filename))

    # Sort files by the piece number
    piece_files.sort(
        key=lambda x: int(
            piece_pattern.search(os.path.basename(x)).group(1)  # type: ignore
        )
    )

    return piece_files
