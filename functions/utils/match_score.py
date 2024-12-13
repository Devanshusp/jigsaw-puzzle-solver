import json
from typing import Tuple
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff


def contours(
    image_path: str, 
    kernel_size: Tuple[int, int] = (5, 5), 
    epsilon_ratio: float = 0.001
):
    """
    (Desc)

    Args:
    
    Returns:
    
    """
    # Reading image and converting to rgb and grayscale.
    img = cv2.imread(image_path)
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
    approx_contours = np.array(approx_contours).reshape(-1, 2) 
    
    return approx_contours 
    
# Takes in data for two pieces, as well as the edge side to compare 
def calc_match_score(p1, p2, side): 
    
    # Reading JSON file 
    json1 = "./images/pieces/aurora12/data.json" 
    with open(json1, 'r') as file: 
        data = json.load(file) 

    # Access the corners 
    corners1 = data["piece_8"]["corners"] 
    corners2 = data["piece_2"]["corners"] 
    
    centroid2 = data["piece_2"]["center_coords"] 
    
    p1a = contours(p1) 
    p2a = contours(p2) 
    
    # Sorting by x coordinate 
    sorted_x1 = sorted(corners1, key=lambda point: point[0]) 
    #sorted_x2 = sorted(corners2, key=lambda point: point[0]) 
    
    # Separate left and right points 
    left_points1 = sorted_x1[:2] 
    right_points1 = sorted_x1[2:] 
    
    # left_points2 = sorted_x2[:2] 
    # right_points2 = sorted_x2[2:] 
    
    # Sort left points by y-coordinate to determine top-left and bottom-left
    left_points1.sort(key=lambda point: point[1])
    top_left1, bottom_left1 = left_points1
    
    # left_points2.sort(key=lambda point: point[1])
    # top_left2, bottom_left2 = left_points2
    
    # Sort right points by y-coordinate to determine top-right and bottom-right
    right_points1.sort(key=lambda point: point[1])
    top_right1, bottom_right1 = right_points1
    
    # right_points2.sort(key=lambda point: point[1])
    # top_right2, bottom_right2 = right_points2
    
    print("TL", top_left1)
    print("TR", top_right1)
    print("BL", bottom_left1)
    print("BR", bottom_right1) 
    
    if side == "top": 
        cornerA1 = top_left1
        cornerB1 = top_right1
    if side == "left": 
        cornerA1 = top_left1
        cornerB1 = bottom_left1
    if side == "bottom": 
        cornerA1 = bottom_left1
        cornerB1 = bottom_right1
    if side == "right": 
        cornerA1 = bottom_right1
        cornerB1 = top_right1
    
    # Get the indices 
    indexA1 = next((i for i, item in enumerate(p1a) if item[0] == cornerA1[0] and item[1] == cornerA1[1]), None)
    indexB1 = next((i for i, item in enumerate(p1a) if item[0] == cornerB1[0] and item[1] == cornerB1[1]), None)
    
    # Matching left 
    indexAl = next((i for i, item in enumerate(p2a) if item[0] == top_left2[0] and item[1] == top_left2[1]), None)
    indexBl = next((i for i, item in enumerate(p2a) if item[0] == bottom_left2[0] and item[1] == bottom_left2[1]), None)
    
    # Matching top 
    indexAt = next((i for i, item in enumerate(p2a) if item[0] == top_left2[0] and item[1] == top_left2[1]), None)
    indexBt = next((i for i, item in enumerate(p2a) if item[0] == top_right2[0] and item[1] == top_right2[1]), None)
    
    # Matching right 
    indexAr = next((i for i, item in enumerate(p2a) if item[0] == bottom_right2[0] and item[1] == bottom_right2[1]), None)
    indexBr = next((i for i, item in enumerate(p2a) if item[0] == top_right2[0] and item[1] == top_right2[1]), None)
    
    # Matching bottom 
    indexAb = next((i for i, item in enumerate(p2a) if item[0] == bottom_left2[0] and item[1] == bottom_left2[1]), None)
    indexBb = next((i for i, item in enumerate(p2a) if item[0] == bottom_right2[0] and item[1] == bottom_right2[1]), None)
    
    # Input Side 
    minVal1 = min(indexA1, indexB1) 
    maxVal1 = max(indexA1, indexB1) 
    
    # Left 
    minVal2 = min(indexAl, indexBl)
    maxVal2 = max(indexAl, indexBl)
    
    # Top 
    minVal3 = min(indexAt, indexBt)
    maxVal3 = max(indexAt, indexBt)
    
    # Right 
    minVal4 = min(indexAr, indexBr)
    maxVal4 = max(indexAr, indexBr)
    
    # Bottom 
    minVal5 = min(indexAb, indexBb)
    maxVal5 = max(indexAb, indexBb)
    
    # Input 
    if (indexA1 == (len(p1a) - 1) & indexB1 == 0) | (indexB1 == (len(p1a) - 1) & indexA1 == 0): 
        coords_list1 = [cornerA1, cornerB1] 
    else: 
        coords_list1 = p1a[minVal1:maxVal1+1] 
    
    # Left 
    if (indexAl == (len(p2a) - 1) & indexBl == 0) | (indexBl == (len(p2a) - 1) & indexAl == 0): 
        coords_list2 = [top_left2, bottom_left2] 
    else: 
        coords_list2 = p2a[minVal2:maxVal2+1] 
    
    # Top 
    if (indexAt == (len(p2a) - 1) & indexBt == 0) | (indexBt == (len(p2a) - 1) & indexAt == 0): 
        coords_list3 = [top_left2, top_right2] 
    else: 
        coords_list3 = p2a[minVal3:maxVal3+1] 
        
    # Right   
    if (indexAr == (len(p2a) - 1) & indexBr == 0) | (indexBr == (len(p2a) - 1) & indexAr == 0): 
        coords_list4 = [top_right2, bottom_right2] 
    else: 
        coords_list4 = p2a[minVal4:maxVal4+1] 
      
    # Bottom   
    if (indexAb == (len(p2a) - 1) & indexBb == 0) | (indexBb == (len(p2a) - 1) & indexAb == 0): 
        coords_list5 = [bottom_left2, bottom_right2] 
    else: 
        coords_list5 = p2a[minVal5:maxVal5+1] 

    # Change (corner_x, corner_y) to the coords of the corner you rotate around 
    transform1 = coords_list3 - (centroid2[0], centroid2[1]) # Top 
    transform2 = coords_list4 - (centroid2[0], centroid2[1]) # Right 
    transform3 = coords_list5 - (centroid2[0], centroid2[1]) # Bottom 

    # Calculate inverse tangent 
    angle1 = np.pi / 2 
    angle2 = np.pi 
    angle3 = np.pi * (3 / 2)

    # Rotation matrix
    rm1 = np.array([
        [np.cos(angle1), -np.sin(angle1)],
        [np.sin(angle1), np.cos(angle1)]
    ])
    rm2 = np.array([
        [np.cos(angle2), -np.sin(angle2)],
        [np.sin(angle2), np.cos(angle2)]
    ])
    rm3 = np.array([
        [np.cos(angle3), -np.sin(angle3)],
        [np.sin(angle3), np.cos(angle3)]
    ])

    # Apply rotation matrix 
    r1 = np.dot(transform1, rm1.T).astype(int)
    r2 = np.dot(transform2, rm2.T).astype(int)
    r3 = np.dot(transform3, rm3.T).astype(int)
    
    r1 += (centroid2[0], centroid2[1])
    r2 += (centroid2[0], centroid2[1])
    r3 += (centroid2[0], centroid2[1])
    
    coords_list2 = coords_list2[::-1] 
    r1 = r1[::-1] 
    r2 = r2[::-1] 
    
    dif1 = coords_list1[0] - coords_list2[0] 
    left_side = coords_list2 + dif1 
    
    dif2 = coords_list1[0] - r1[0] 
    top_side = r1 + dif2 
    
    dif3 = coords_list1[0] - r2[0] 
    right_side = r2 + dif3 
    
    dif4 = coords_list1[0] - r3[0] 
    bottom_side = r3 + dif4 
    
    # coords_list1 vs. left/top/bottom/right 
    
    sides = [left_side, top_side, right_side, bottom_side] 
    
    lowest_hd = float("inf")
    
    count = 0
    
    for side in sides: 
        
        # Compute directed Hausdorff distances
        d_AB = directed_hausdorff(coords_list1, side)[0]
        d_BA = directed_hausdorff(side, coords_list1)[0]

        # Hausdorff distance is the max of the two directed distances
        hausdorff_distance = max(d_AB, d_BA)
        
        if hausdorff_distance < lowest_hd: 
            lowest_hd = hausdorff_distance 
            best = count 

        # print(f"Hausdorff Distance: {hausdorff_distance}")
        
        count += 1 
    
    if best == 0: 
        best_side = "Left"
        orientation = 0
    if best == 1: 
        best_side = "Top"
        orientation = 1
    if best == 2: 
        best_side = "Right"
        orientation = 2
    if best == 3: 
        best_side = "Bottom"
        orientation = 3
    
    print("Best Side:", best_side)
    print("# Turns:", orientation)
    
    return lowest_hd
    
# Run the approximation
image_path = "images/pieces/aurora12/piece_8.png"  # Replace with your image path
image_path2 = "images/pieces/aurora12/piece_2.png"

calc_match_score(image_path, image_path2, "right")

