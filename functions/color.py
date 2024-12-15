import json
import os
import re
from typing import List

import cv2
from matplotlib import pyplot as plt
import numpy as np
from functions.solving.solve_border import solve_border
from functions.utils.display_pieces import display_pieces
from functions.utils.piece_file_names import get_piece_file_names
from functions.utils.save_data import get_data_for_piece

DISPLAY_STEPS = True
SKIP_DISPLAY_PREPROCESS_PUZZLE = False
SKIP_DISPLAY_PREPROCESS_PIECES = False
SKIP_DISPLAY_SOLVE = False
SKIP_PREPROCESS = False
SAVE_PREPROCESS = True
SAVE_PREPROCESS_FOLDER = ".images2"
PUZZLES_TO_PROCESS = ["aurora12",]


def display_pieces_side(
    pieces: List[np.ndarray],
    save_name: str,
    figsize: tuple = (10, 10),
    save: bool = True,
    display_steps: bool = True,
    coordinates: List[tuple] = None  # List of (x, y) coordinates to plot
) -> None:
    """
    Display a list of puzzle piece images in a grid layout, with optional points drawn on them.

    Args:
        pieces: A list of image pieces (numpy arrays) to be displayed.
        coordinates: A list of (x, y) coordinates to plot on each piece.
        save_name: The name of the file to save the grid image.
        figsize: The size of the figure.
        save: Whether to save the image.
        display_steps: Whether to display the plot.
    """
    # Calculate number of pieces.
    num_pieces = len(pieces)

    # Calculate grid dimensions (rows and columns).
    rows = int(np.ceil(np.sqrt(num_pieces)))
    cols = int(np.ceil(num_pieces / rows))

    # Create a new figure for displaying all pieces.
    plt.figure(figsize=figsize)

    # Add each piece to the grid.
    for i, piece in enumerate(pieces):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(piece)
        
        # Check if coordinates are provided and are not empty
        if coordinates is not None and len(coordinates) > 0:
            for (x, y) in coordinates:
                plt.scatter(x, y, color='red', s=50)  # Plot red points
        
        plt.title(f"Piece {i}")
        plt.axis("off")

    # Adjust layout to avoid overlap.
    plt.tight_layout()

    # Save the figure as an image file if needed.
    if save:
        plt.savefig(f"{save_name}.png", dpi=300)

    if display_steps:
        # Show the plot.
        plt.show()

    # Clear the figure after saving and showing to avoid issues.
    plt.clf()
    plt.close()

def display_pieces_color(
    pieces: List[np.ndarray],
    save_name: str,
    n_pixels: int = 10,  # Number of pixels to move towards the centroid
    figsize: tuple = (10, 10),
    save: bool = True,
    display_steps: bool = True,
) -> None:
    """
    Display a list of puzzle piece images in a grid layout and draw a red line on each piece
    showing the movement of contour points towards the centroid.

    Args:
        pieces: A list of image pieces (numpy arrays) to be displayed.
        contours_list: A list of contour points for each piece.
        centroids_list: A list of centroids for each piece.
        n_pixels: Number of pixels to move towards the centroid.
        figsize: Size of the figure for display.
        save_name: Name of the file to save the grid of pieces.
        save: Whether to save the image.
        display_steps: Whether to display the grid of pieces.

    Returns:
        None: It shows the pieces in a grid layout using Matplotlib.
    """
    # Calculate number of pieces.
    num_pieces = len(pieces)

    # Calculate grid dimensions (rows and columns).
    rows = int(np.ceil(np.sqrt(num_pieces)))
    cols = int(np.ceil(num_pieces / rows))

    # Create a new figure for displaying all pieces.
    plt.figure(figsize=figsize)

    # Add each piece to the grid.
    for i, piece in enumerate(pieces):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(piece)
        plt.title(f"Piece {i}")
        plt.axis("off")
        
        # Apply a heavy Gaussian blur on grayscale image to reduce noise.
        kernel_size = (5, 5)
        kernel_sigma = 5
        img_rgb = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(img_rgb, kernel_size, kernel_sigma)

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
        centroid = (cx, cy)

        # Combine all slices into one 2D array.
        contours = np.array(contours)
        contours = contours.reshape(-1, 2)
        
        # Compute direction vectors from each contour point to the centroid
        direction_vectors = centroid - contours
        
        # Normalize the direction vectors (convert to unit vectors)
        magnitudes = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
        normalized_vectors = direction_vectors / magnitudes
        
        # Move each contour point `n_pixels` towards the centroid
        moved_contours = contours + normalized_vectors * n_pixels
        
        # Draw a red line connecting original contour points to moved points
        for (x1, y1), (x2, y2) in zip(contours, moved_contours):
            # Plot the line from original to moved contour point
            plt.plot([x1, x2], [y1, y2], color='red', linewidth=0.1)
            #print("Colored")

    # Adjust layout to avoid overlap.
    plt.tight_layout()

    # Save the figure as an image file if needed.
    if save:
        plt.savefig(f"{save_name}.png", dpi=300)

    if display_steps:
        # Show the plot.
        plt.show()

    # Clear the figure after saving and showing to avoid issues.
    plt.clf()
    plt.close()

def view_pieces(
    PUZZLE_IMAGE_NAME: str,
    n_pixels: int = 10, 
    save: bool = False,
    save_folder: str = "images",
    display_steps: bool = True,
):
    # Define the folder path
    PIECES_SAVE_PATH = f"{save_folder}/pieces/{PUZZLE_IMAGE_NAME}"
    PIECES_DATA_GRID_SAVE_PATH = f"{save_folder}/pieces_grid"

    # Regex pattern to match files named piece_{i}.png
    piece_pattern = re.compile(r"^piece_(\d+)\.png$")

    # Store matching file paths
    piece_files = get_piece_file_names(PIECES_SAVE_PATH, piece_pattern)

    # List to store piece data
    piece_indices = []
    pieces = []

    # Process each piece file
    for piece_file in piece_files:
        # Extract piece index value
        match = piece_pattern.search(os.path.basename(piece_file))
        piece_index = int(match.group(1))  # type: ignore

        img = cv2.imread(PIECES_SAVE_PATH + f"/piece_{piece_index}.png")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        piece_indices.append(piece_index)
        pieces.append(img_rgb)
    
    # Display corner pieces in a grid
    display_pieces_color(
        pieces,
        save_name=PIECES_DATA_GRID_SAVE_PATH + f"/{PUZZLE_IMAGE_NAME}_2_corner",
        n_pixels = n_pixels,  # Number of pixels to move towards the centroid
        figsize = (10, 10),
        save=save,
        display_steps=display_steps,
    )
    
def color(puzzle, piece, side, pixels): 
    
    img = ".images2/pieces/" + puzzle + "/piece_" + str(piece) + ".png"

    pieceNum = "piece_" + str(piece)

    # Load the image
    image = cv2.imread(img, cv2.IMREAD_COLOR)

    # Reading JSON file 
    json1 = ".images2/pieces/" + puzzle + "/data.json" 
    with open(json1, 'r') as file: 
        data = json.load(file) 

    contours = data[pieceNum]["piece_side_data"][side]["contour_points"]
    
    centroid = data[pieceNum]["center_coords"]
    
    # Convert lists to numpy arrays for easier vector math
    contours = np.array(contours)
    centroid = np.array(centroid)
    
    # Compute direction vector from each contour point to the centroid
    direction_vectors = centroid - contours
    
    # Normalize the direction vectors (convert to unit vectors)
    magnitudes = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    normalized_vectors = direction_vectors / magnitudes
    
    # Move each contour point n_pixels towards the centroid
    contours = contours + normalized_vectors * pixels
    
    # Extract color data along the contour
    colors = []
    for (x, y) in contours:
        x = int(x)
        y = int(y)
        # Check bounds to avoid index errors
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            color = image[y, x]  # OpenCV uses (row, col) indexing
            colors.append(color)  # Append the RGB color values

    # Print the RGB values for each pixel location
    # for idx, color in enumerate(colors):
    #     print(f"Pixel {idx} -> RGB: {color}")

    # Visualize the extracted colors as a color strip
    color_strip = np.array(colors)  # Convert colors list to NumPy array
    color_strip = color_strip.reshape((1, -1, 3))  # Reshape into a horizontal strip
    color_strip = np.repeat(color_strip, 20, axis=0)  # Repeat rows for better visibility
    
    # Display the color strip
    plt.imshow(color_strip)
    plt.title(f"Piece {piece}, Side {side}")
    plt.axis('off')  # Turn off axis labels for a clean view
    plt.show()
    
    pieces = []
    
    imgPath = cv2.imread(img)
    img_rgb = cv2.cvtColor(imgPath, cv2.COLOR_BGR2RGB)
    pieces.append(img_rgb)
    
    PIECES_DATA_GRID_SAVE_PATH = f".images2/pieces_grid_colors"

    display_pieces_side(
        pieces,
        save_name=PIECES_DATA_GRID_SAVE_PATH + f"/{pieceNum}_color",
        save=False,
        display_steps=True,
        coordinates=contours
    )
    
pixels = 25 

for puzzle in PUZZLES_TO_PROCESS:
    view_pieces(
        puzzle,
        n_pixels=pixels,
        save=SAVE_PREPROCESS,
        save_folder=SAVE_PREPROCESS_FOLDER,
        display_steps=DISPLAY_STEPS and not SKIP_DISPLAY_SOLVE,
    )

color("aurora12", 0, "A", pixels)
color("aurora12", 0, "B", pixels)
color("aurora12", 0, "C", pixels)
color("aurora12", 0, "D", pixels)
