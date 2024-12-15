import json
from typing import Tuple
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff

from match_score import contours 

def calc_ms4_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, S2): 
    
    CN2 = data[piece2]["piece_side_data"][S2]["points"]
    
    piece1corners = data[piece1]["corners"] 
    piece2corners = data[piece2]["corners"]
    
    
    
    # Rotate around the centroid 
    transform = coords_list2 - (CEN[0], CEN[1]) 
    
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
def calc_ms4(PZ, P1, P2, S): 
    
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
        calc_ms4_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, "A"))
    
    hausdorff_distances.append(
        calc_ms4_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, "B"))
    hausdorff_distances.append(
        calc_ms4_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, "C")) 
    hausdorff_distances.append(
        calc_ms4_helper(CT1, CN1, CT2, data, piece1, piece2, CEN, "D")) 
    
    print(min(hausdorff_distances))

    return min(hausdorff_distances)


calc_ms4("aurora12", 8, 2, "A") 
calc_ms4("aurora12", 8, 2, "B") 
calc_ms4("aurora12", 8, 2, "C") 
calc_ms4("aurora12", 8, 2, "D") 