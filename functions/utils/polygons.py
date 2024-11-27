import pprint
import cv2
import matplotlib.pyplot as plt
import numpy as np


def polygonal_approximation(image_path, epsilon_ratio=0.02):
    # Reading image and converting to grayscale 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(12, 6))
    plt.title("Original Img")
    plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.show()

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
    
    
    dT = []

    for i in range(0, len(reshaped)):
        
        # Center and Border Points: 
        cX = mean_xy[0] 
        #print("Center X:", cX)
        cY = mean_xy[1]
        #print("Center Y:", cY)
        bX = reshaped[i,0]
        print("Border X:", bX)
        bY = reshaped[i,1]
        print("Border Y:", bY)
        
        # Delta Values: 
        dX = bX - cX 
        print("Delta X:", dX)
        dY = -(bY - cY)
        print("Delta Y:", dY)
        
        # Distance Eqn: 
        dX2 = np.square(dX)
        dY2 = np.square(dY)
        dR = np.sqrt(dX2 + dY2) 
        #print("Distance:", dR)
        
        # Angle 
        theta = np.arctan2(dY,dX) * 180 / np.pi
        
        if theta < 0: 
            theta += 360 
            
        print("Theta:", theta)
        
        dT.append({ 
                "coords": (bX, bY), 
                "dist": dR, 
                "angle": theta
        })
    
    pprint.pprint(dT)
    
    # Initiazing empty images to draw contours on 
    img_contours = np.zeros_like(img) 
    img_approx = np.zeros_like(img) 

    # Draw original & polygonal approx contours 
    cv2.drawContours(img_contours, contours, -1, (255), 1)
    cv2.drawContours(img_approx, approx_contours, -1, (255), 1)

    # Show results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Original Contours")
    plt.imshow(img_contours, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Polygonal Approximation")
    plt.imshow(img_approx, cmap="gray")
    
    # Add the red point on top of the image
    plt.scatter(mean_xy[0], mean_xy[1], color='red', s=100, label='Mean Point')  # s=100 for size of the point

    plt.tight_layout()
    plt.show()

# Run the approximation
image_path = "images/pieces/aurora30/piece_0.png"  # Replace with your image path
polygonal_approximation(image_path, epsilon_ratio=0.002) 