import cv2
import numpy as np

def detect_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, _ = orb.detectAndCompute(gray, None)

    # Draw keypoints on the image
    keypoint_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    return keypoint_image, keypoints

# Example usage
cat_image_path = './cat_image.jpg'
keypoint_image, keypoints = detect_features(cat_image_path)
cv2.imshow('Cat Features', keypoint_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
