import cv2
import numpy as np


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated


def match_score(contour1, contour2):
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)


def join_pieces(p1, p2):
    contours1, _ = cv2.findContours(p1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(p2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = float("inf")
    best_combined = None

    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(p2, angle)
        for c1 in contours1:
            for c2 in contours2:
                score = match_score(c1, c2)
                if score < best_score:
                    best_score = score
                    best_combined = np.hstack((p1, rotated))  # Example merge

    return best_score, best_combined


# Example usage
p1 = cv2.imread(
    "/Users/devanshu/Documents/Code/1_code/jigsaw-puzzle-solver/images/pieces/aurora12/piece_0.png",
    cv2.IMREAD_GRAYSCALE,
)
p2 = cv2.imread(
    "/Users/devanshu/Documents/Code/1_code/jigsaw-puzzle-solver/images/pieces/aurora12/piece_1.png",
    cv2.IMREAD_GRAYSCALE,
)

score, combined = join_pieces(p1, p2)
print("Best match score:", score)
