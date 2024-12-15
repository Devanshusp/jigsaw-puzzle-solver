import json
from typing import Tuple
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff

from match_score import contours 

def calc_ms3_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, S2): 
    
    CN2 = data[piece2]["piece_side_data"][S2]["points"]
    
    # Piece 1 
    index1a = next((i for i, item in enumerate(CT1) if item[0] == CN1[0][0] and item[1] == CN1[0][1]), None)
    index1b = next((i for i, item in enumerate(CT1) if item[0] == CN1[1][0] and item[1] == CN1[1][1]), None)
    # Piece 2 
    index2a = next((i for i, item in enumerate(CT2) if item[0] == CN2[0][0] and item[1] == CN2[0][1]), None)
    index2b = next((i for i, item in enumerate(CT2) if item[0] == CN2[1][0] and item[1] == CN2[1][1]), None)
    
    # # Determining lower/higher index 
    # Piece 1 
    minVal1 = min(index1a, index1b) 
    maxVal1 = max(index1a, index1b) 
    # Piece 2 
    minVal2 = min(index2a, index2b) 
    maxVal2 = max(index2a, index2b) 
    
    # If the two indices are the beginning and end of the list, 
    # they are the two corners of a flat edge and we wish to skip the 
    # indices between them. Otherwise, take all indices between the min 
    # and max index value. 
    # Piece 1 
    piece1corners = data[piece1]["corners"] 
    piece2corners = data[piece2]["corners"]

    # Initialize values to store the lowest and highest x and y points
    lowest_y_points = [[float('inf'), float('inf')], [float('inf'), float('inf')]]
    highest_y_points = [[-float('inf'), -float('inf')], [-float('inf'), -float('inf')]]
    lowest_x_points = [[float('inf'), float('inf')], [float('inf'), float('inf')]]
    highest_x_points = [[-float('inf'), -float('inf')], [-float('inf'), -float('inf')]]

    # Iterate through the points and find the lowest and highest x and y points
    for corner in piece1corners: 
        
        # For the lowest y-values
        if corner[1] < lowest_y_points[0][1]:
            lowest_y_points[1] = lowest_y_points[0]
            lowest_y_points[0] = corner
        elif corner[1] < lowest_y_points[1][1]:
            lowest_y_points[1] = corner
            
        # For the highest y-values
        if corner[1] > highest_y_points[0][1]:
            highest_y_points[1] = highest_y_points[0]
            highest_y_points[0] = corner
        elif corner[1] > highest_y_points[1][1]:
            highest_y_points[1] = corner
            
        # For the lowest x-values
        if corner[0] < lowest_x_points[0][0]:
            lowest_x_points[1] = lowest_x_points[0]
            lowest_x_points[0] = corner
        elif corner[0] < lowest_x_points[1][0]:
            lowest_x_points[1] = corner
            
        # For the highest x-values
        if corner[0] > highest_x_points[0][0]:
            highest_x_points[1] = highest_x_points[0]
            highest_x_points[0] = corner
        elif corner[0] > highest_x_points[1][0]:
            highest_x_points[1] = corner

    # Step 2: Check if the given points match the lowest y-value points
    if sorted(CN1) == sorted(lowest_y_points):
        #print("(top) The given points are the corners with the smallest y-values.")
        side1 = "T"
    if sorted(CN1) == sorted(highest_y_points):
        #print("(bottom) The given points are the corners with the highest y-values.")
        side1 = "B"
    if sorted(CN1) == sorted(lowest_x_points):
        #print("(left) The given points are the corners with the smallest x-values.")
        side1 = "L"
    if sorted(CN1) == sorted(highest_x_points):
        #print("(right) The given points are the corners with the highest x-values.")
        side1 = "R"
        
    # Initialize values to store the lowest and highest x and y points
    lowest_y_points2 = [[float('inf'), float('inf')], [float('inf'), float('inf')]]
    highest_y_points2 = [[-float('inf'), -float('inf')], [-float('inf'), -float('inf')]]
    lowest_x_points2 = [[float('inf'), float('inf')], [float('inf'), float('inf')]]
    highest_x_points2 = [[-float('inf'), -float('inf')], [-float('inf'), -float('inf')]]

    # Iterate through the points and find the lowest and highest x and y points
    for corner in piece2corners: 
        
        # For the lowest y-values
        if corner[1] < lowest_y_points2[0][1]:
            lowest_y_points2[1] = lowest_y_points2[0]
            lowest_y_points2[0] = corner
        elif corner[1] < lowest_y_points2[1][1]:
            lowest_y_points2[1] = corner
            
        # For the highest y-values
        if corner[1] > highest_y_points2[0][1]:
            highest_y_points2[1] = highest_y_points2[0]
            highest_y_points2[0] = corner
        elif corner[1] > highest_y_points2[1][1]:
            highest_y_points2[1] = corner
            
        # For the lowest x-values
        if corner[0] < lowest_x_points2[0][0]:
            lowest_x_points2[1] = lowest_x_points2[0]
            lowest_x_points2[0] = corner
        elif corner[0] < lowest_x_points2[1][0]:
            lowest_x_points2[1] = corner
            
        # For the highest x-values
        if corner[0] > highest_x_points2[0][0]:
            highest_x_points2[1] = highest_x_points2[0]
            highest_x_points2[0] = corner
        elif corner[0] > highest_x_points2[1][0]:
            highest_x_points2[1] = corner

    # Step 2: Check if the given points match the lowest y-value points
    if sorted(CN2) == sorted(lowest_y_points2):
        #print("(top) The given points are the corners with the smallest y-values.")
        side2 = "T"
    elif sorted(CN2) == sorted(highest_y_points2):
        #print("(bottom) The given points are the corners with the highest y-values.")
        side2 = "B"
    elif sorted(CN2) == sorted(lowest_x_points2):
        #print("(left) The given points are the corners with the smallest x-values.")
        side2 = "L"
    elif sorted(CN2) == sorted(highest_x_points2):
        #print("(right) The given points are the corners with the highest x-values.")
        side2 = "R"
    
    # Piece 1 
    if side1 == "T": 
        coords_list1 = CT1[maxVal1:len(CT1)] 
        coords_list1 = np.append(coords_list1, [CT1[0]], axis=0) 
    elif (index1a == (len(CT1) - 1) & index1b == 0) | (index1b == (len(CT1) - 1) & index1a == 0): 
        coords_list1 = [CN1[0], CN1[1]] 
    else: 
        coords_list1 = CT1[minVal1:maxVal1+1] 
    # Piece 2 
    if side2 == "T": 
        coords_list2 = CT2[maxVal2:len(CT2)] 
        coords_list2 = np.append(coords_list2, [CT2[0]], axis=0) 
    elif (index2a == (len(CT2) - 1) & index2b == 0) | (index2b == (len(CT2) - 1) & index2a == 0): 
        coords_list2 = [CN2[0], CN2[1]] 
    else: 
        coords_list2 = CT2[minVal2:maxVal2+1] 
    
    # Rotate around the centroid 
    transform = coords_list2 - (CEN[0], CEN[1]) 
    
    # Side cases 
    LBRT = {
        ("L", "L"): 2, 
        ("L", "T"): 3, 
        ("L", "R"): 0, 
        ("L", "B"): 1, 
        ("T", "L"): 1, 
        ("T", "T"): 2, 
        ("T", "R"): 3, 
        ("T", "B"): 0, 
        ("R", "L"): 0, 
        ("R", "T"): 1, 
        ("R", "R"): 2, 
        ("R", "B"): 3, 
        ("B", "L"): 3, 
        ("B", "T"): 0, 
        ("B", "R"): 1, 
        ("B", "B"): 2, 
    } 

    # Get the logic based on input 
    result = LBRT.get((side1, side2), -1) 
    
    # print("Rotations:", result)
    
    angle = result * (np.pi / 2) 

    # Rotation matrix 
    rm = np.array([ 
        [np.cos(angle), -np.sin(angle)], 
        [np.sin(angle), np.cos(angle)] 
    ]) 

    # Apply rotation matrix 
    r = np.dot(transform, rm.T).astype(int) 
    r += (CEN[0], CEN[1]) 

    # Must invert even # of rotations 
    if result % 2 == 0: 
        r = r[::-1] 
    
    # Translation 
    dif = coords_list1[0] - r[0] 
    side = r + dif 
    
    # Compute Hausdorff distance 
    dh1 = directed_hausdorff(coords_list1, side)[0] 
    dh2 = directed_hausdorff(side, coords_list1)[0] 
    hausdorff_distance = max(dh1, dh2) 
    
    # print(hausdorff_distance)
    
    return hausdorff_distance 

# Takes in data for two pieces, as well as the edge side to compare 
def calc_ms3(PZ, P1, P2, S): 
    
    piece1 = "piece_" + str(P1)
    piece2 = "piece_" + str(P2)
    
    # Image paths 
    image_path = ".images/pieces/" + PZ + "/" + piece1 + ".png"  
    image_path2 = ".images/pieces/" + PZ + "/" + piece2 + ".png"
    
    CT1 = contours(image_path) 
    CT2 = contours(image_path2) 
    
    # Reading JSON file 
    json1 = ".images/pieces/" + PZ + "/data.json" 
    with open(json1, 'r') as file: 
        data = json.load(file) 
    
    CEN = data[piece2]["center_coords"] 
    
    CN1 = data[piece1]["piece_side_data"][S]["points"]
    
    hausdorff_distances = [] 
    
    hausdorff_distances.append(
        calc_ms3_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, "A"))
    
    hausdorff_distances.append(
        calc_ms3_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, "B"))
    hausdorff_distances.append(
        calc_ms3_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, "C")) 
    hausdorff_distances.append(
        calc_ms3_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, "D")) 
    
    print(min(hausdorff_distances))

    return min(hausdorff_distances)


calc_ms3("aurora12", 8, 2, "A") 
calc_ms3("aurora12", 8, 2, "B") 
calc_ms3("aurora12", 8, 2, "C") 
calc_ms3("aurora12", 8, 2, "D") 