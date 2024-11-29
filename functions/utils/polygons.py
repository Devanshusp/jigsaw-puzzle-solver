import csv
import pprint
import cv2
import matplotlib.pyplot as plt
import numpy as np


def polygonal_approximation(image_path, epsilon_ratio=0.02):
    # Reading image and converting to grayscale 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # plt.figure(figsize=(12, 6))
    # plt.title("Original Img")
    # plt.imshow(img, cmap="gray")
    # plt.tight_layout()
    # plt.show()
    
    # Threshold the image to create a binary mask:
    # Pixels > 254 are set to 0 (white background); others set to 255 (piece area).
    _, binary = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)

    # Compute image moments from the binary mask to calculate the weighted center.
    moments = cv2.moments(binary)

    # If the area (m00) is non-zero, calculate center of mass:
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])  # x-coordinate of the center
        center_y = int(moments["m01"] / moments["m00"])  # y-coordinate of the center
    else:
        center_x, center_y = 0, 0  # Avoid division by zero if m00 is zero

    # Detecting edges using canny 
    edges = cv2.Canny(img, 100, 200)

    # Finding contours: takes in edges, returns only external contours, no approx 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Polygonal approximation  
    approx_contours = []
    for contour in contours:
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_contours.append(approx)
        
    approx_contours = np.array(approx_contours)
    
    # Combine all slices into one 2D array
    reshaped = approx_contours.reshape(-1, 2) 
    
    # Combine into a single mean x-y pair
    mean_xy = np.round(np.mean(reshaped, axis=0)).astype(int)
    
    # Testing Coordinates + Mean 
    print("coordinates:", reshaped)
    print("mean:", mean_xy)
    
    # Space to store coordinates, distances, and angles 
    dT = []

    for i in range(0, len(reshaped)):
        
        # Center and Border Points: 
        cX = mean_xy[0] 
        cY = mean_xy[1]
        bX = reshaped[i,0]
        bY = reshaped[i,1]
        
        # Delta Values: 
        dX = bX - cX 
        dY = -(bY - cY)
        
        # Distance Eqn: 
        dX2 = np.square(dX)
        dY2 = np.square(dY)
        dR = np.sqrt(dX2 + dY2) 
        
        # Angle 
        theta = np.arctan2(dY,dX) * 180 / np.pi
        
        if theta < 0: 
            theta += 360 
        
        # Test Prints 
        
        if 25 < theta < 65: 
            print("----------------------45------------------------")
            print("Theta:", theta)
            print("bX,bY:", bX, bY)
            print("distance:", dR)
            
        if 115 < theta < 155: 
            print("----------------------135------------------------")
            print("Theta:", theta)
            print("bX,bY:", bX, bY)
            print("distance:", dR)
            
        if 205 < theta < 245: 
            print("----------------------225------------------------")
            print("Theta:", theta)
            print("bX,bY:", bX, bY)
            print("distance:", dR)
            
        if 295 < theta < 315: 
            print("----------------------315------------------------")
            print("Theta:", theta)
            print("bX,bY:", bX, bY)
            print("distance:", dR)
        
        # Saving data 
        dT.append({ 
                "coords": (bX, bY), 
                "dist": dR, 
                "angle": theta
        })
    
    # Testing print 
    pprint.pprint(dT)
    
    # Converting types and saving in csv 
    for item in dT:
        item['angle'] = float(item['angle'])
        item['coords'] = tuple(map(int, item['coords']))
        item['dist'] = float(item['dist'])

    with open("output.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["angle", "coords", "dist", image_path])
        writer.writeheader()
        for item in dT:
            item['coords'] = str(item['coords']) 
            writer.writerow(item)
    
    # Initiazing empty images to draw contours on 
    img_contours = np.zeros_like(img) 
    img_approx = np.zeros_like(img) 

    # Draw original & polygonal approx contours 
    cv2.drawContours(img_contours, contours, -1, (255), 1)
    cv2.drawContours(img_approx, approx_contours, -1, (255), 1)
    
    # Overlay a red dot on the original image at the computed center.
    img_center = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
    cv2.circle(img_center, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

    # Show results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 4, 2)
    plt.title("Original Contours")
    plt.imshow(img_contours, cmap="gray")

    plt.subplot(1, 4, 3)
    
    # Original image with the weighted center marked
    plt.imshow(cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB))
    plt.title("Weighted Center (Red Dot)")

    # Creating and drawing lines 
    color = (255, 0, 0) 
    thickness = 1
    point1 = (center_x, center_y)
    for i in range(0, len(reshaped)): 
        point2 = (reshaped[i,0],reshaped[i,1])
        cv2.line(img_approx, point1, point2, color, thickness)

    plt.subplot(1, 4, 4)
    plt.title("Polygonal w/ Lines") 
    plt.imshow(img_approx, cmap="gray")
    
    plt.tight_layout()
    plt.show()
    
    # Corner angles 
    corner_angles = [45, 135, 225, 315]
  
    # Initialize list to store best corner matches 
    filtered_items = []

    for target in corner_angles:
        
        # Choosing the closest to corner 
        closest_items = sorted(
            dT, 
            key=lambda item: abs(float(item['angle']) - target) 
        )[:3] # Top 3
        
        # Test printing 
        print("Closest Angles")
        pprint.pprint(closest_items)

        # Choose the largest distance 
        closest_item = max(
            closest_items, 
            key=lambda item: float(item['dist'])  # Maximize based on distance
        )
        
        # Add the closest item 
        filtered_items.append({
            'angle': closest_item['angle'], 
            'dist': closest_item['dist'], 
            'coords': closest_item['coords']
        })
    
    # Test print 
    print(len(filtered_items))
    
    # Initiazing empty images to draw contours on 
    corners = np.zeros_like(img) 

    # Draw original & polygonal approx contours 
    cv2.drawContours(corners, approx_contours, -1, (255), 1)
    
    import ast

    # Drawing lines between the closest matches 
    for i in range(len(filtered_items)):
        
        item = filtered_items[i]
        coords_str = item['coords']
        
        # Convert coords to tuple 
        coords = ast.literal_eval(coords_str)
        if isinstance(coords, tuple) and len(coords) == 2:
            bX, bY = coords  
        else:
            print(f"Improper format: {coords}")
            continue  
            
        point2 = (bX, bY) 
        
        # Draw a line from point1 to point2
        cv2.line(corners, point1, point2, color, thickness) 
        
    plt.figure(figsize=(12, 6))
    plt.title("Corners w/ Lines") 
    plt.imshow(corners, cmap="gray")
    
    plt.tight_layout()
    plt.show()

# Run the approximation
image_path = "images/pieces/aurora30/piece_0.png"  # Replace with your image path
polygonal_approximation(image_path, epsilon_ratio=0.002) 