import json
from typing import Tuple
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff

from match_score import contours

# Takes in data for two pieces, as well as the edge side to compare 

# CT1 - piece one contours 
# CT2 - piece two contours 
# CN1 - piece one corners [[x1,y1],[x2,y2]]
# CN2 - piece two corners 
# Cen - piece two centroid 
# S1 - side of piece one to compare 
# S2 - side of piece two to compare 
def calc_ms2(CT1, CT2, CN1, CN2, CEN, S1, S2): 
    
    # Piece 1 
    index1a = next((i for i, item in enumerate(CT1) if item[0] == CN1[0][0] and item[1] == CN1[0][1]), None)
    index1b = next((i for i, item in enumerate(CT1) if item[0] == CN1[1][0] and item[1] == CN1[1][1]), None)
    # Piece 2 
    index2a = next((i for i, item in enumerate(CT2) if item[0] == CN2[0][0] and item[1] == CN2[0][1]), None)
    index2b = next((i for i, item in enumerate(CT2) if item[0] == CN2[1][0] and item[1] == CN2[1][1]), None)
    
    # Determining lower/higher index 
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
    if S1 == "T": 
        coords_list1 = CT1[maxVal1:len(CT1)]
        coords_list1 = np.append(coords_list1, [CT1[0]], axis=0)
    elif (index1a == (len(CT1) - 1) & index1b == 0) | (index1b == (len(CT1) - 1) & index1a == 0): 
        coords_list1 = [CN1[0], CN1[1]] 
    else: 
        coords_list1 = CT1[minVal1:maxVal1+1] 
    # Piece 2 
    if S2 == "T": 
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
    result = LBRT.get((S1, S2), -1) 
    
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
    
    print(hausdorff_distance)
    
    return hausdorff_distance 


# Image paths 
image_path = ".images/pieces/aurora12/piece_8.png"  
image_path2 = ".images/pieces/aurora12/piece_2.png"

p1a = contours(image_path)
p2a = contours(image_path2) 

# Reading JSON file 
json1 = ".images/pieces/aurora12/data.json" 
with open(json1, 'r') as file: 
    data = json.load(file) 

cornersPiece1 = data["piece_8"]["corners"]
cornersPiece2 = data["piece_2"]["corners"]

tCorn1 = [[2, 7], [286, 4]] # low y's 
lCorn1 = [[2, 7], [0, 347]] # low x's 
bCorn1 = [[0, 347], [283, 348]] # high y's 
rCorn1 = [[286, 4], [283, 348]] # high x's 

tCorn2 = [[64, 0], [374, 18]] # low y's 
lCorn2 = [[64, 0], [59, 346]] # low x's 
bCorn2 = [[59, 346], [403, 346]] # high y's 
rCorn2 = [[374, 18], [403, 346]] # high x's 

centroid = data["piece_2"]["center_coords"] 

calc_ms2(p1a, p2a, tCorn1, tCorn2, centroid, "T", "T") # Check  
calc_ms2(p1a, p2a, tCorn1, lCorn2, centroid, "T", "L") # Check 
calc_ms2(p1a, p2a, tCorn1, bCorn2, centroid, "T", "B") # Check 
calc_ms2(p1a, p2a, tCorn1, rCorn2, centroid, "T", "R") # Check 
calc_ms2(p1a, p2a, lCorn1, tCorn2, centroid, "L", "T") # Check 
calc_ms2(p1a, p2a, lCorn1, lCorn2, centroid, "L", "L") # Inv
calc_ms2(p1a, p2a, lCorn1, bCorn2, centroid, "L", "B") # No
calc_ms2(p1a, p2a, lCorn1, rCorn2, centroid, "L", "R") # Inv
calc_ms2(p1a, p2a, bCorn1, tCorn2, centroid, "B", "T") # Check 
calc_ms2(p1a, p2a, bCorn1, lCorn2, centroid, "B", "L") # No 
calc_ms2(p1a, p2a, bCorn1, bCorn2, centroid, "B", "B") # Inv
calc_ms2(p1a, p2a, bCorn1, rCorn2, centroid, "B", "R") # No 
calc_ms2(p1a, p2a, rCorn1, tCorn2, centroid, "R", "T") # Check 
calc_ms2(p1a, p2a, rCorn1, lCorn2, centroid, "R", "L") # Inv
calc_ms2(p1a, p2a, rCorn1, bCorn2, centroid, "R", "B") # No
calc_ms2(p1a, p2a, rCorn1, rCorn2, centroid, "R", "R") # Inv

