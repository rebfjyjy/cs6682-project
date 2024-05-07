import cv2
import numpy as np

# Load the image
image_path = './data/cat.png'
image = cv2.imread(image_path)

# Check if image is loaded
if image is None:
    print("Error: Image could not be read.")
    exit()

# Define points (you would replace these with your actual points)
# Example: Four points defining a rectangle
points = np.array([
    [620.6183668235853,106.16862051911403],
    [636.4445716483799,86.38586448812111],
    [743.2714542157419,216.9520542926748],
    [727.4452493909475,239.70222372831677]
])

# Compute the bounding box from the points
x_min = int(np.min(points[:, 0]))
y_min = int(np.min(points[:, 1]))
x_max = int(np.max(points[:, 0]))
y_max = int(np.max(points[:, 1]))

# Crop the image
cropped_image = image[y_min:y_max, x_min:x_max]

# Save or display the cropped image
cv2.imwrite('cropped_image.png', cropped_image)
# cv2.imshow('Cropped Image', cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("Image has been cropped and saved.")
