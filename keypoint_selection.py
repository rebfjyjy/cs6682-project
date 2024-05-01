import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

class ImageProcess:
    def __init__(self, image_path, csv_path='dance_landmarks.csv'):
        self.image_path = image_path
        self.csv_path = csv_path
        self.points_dict = {}

    def draw_points(self, ax, points, color):
        """ Helper function to draw points on the axis """
        for point in points:
            ax.scatter(point[0], point[1], color=color, s=100)  # s is the size of the point

    def select_features(self):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.axis('off')  # Turn off axis

        body_parts = [
            "head", "upper_left_arm", "lower_left_arm",
            "upper_right_arm", "lower_right_arm", "left_leg",
            "right_leg", "body"
        ]

        for part in body_parts:
            plt.title(f'Select the {part}, then press Enter')
            points = plt.ginput(n=4, timeout=0)
            self.draw_points(ax, points, 'red')
            plt.draw()
            self.points_dict[part] = points
            print(f'{part} points:', points)

        plt.close(fig)
        return self.points_dict

    def save_points(self):
        # Write the points to a CSV file
        with open(csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write the header
            header = [''] 
            for i in range(4):  # Assuming each part has exactly 4 points
                header.append(f'x_{i+1}')
                header.append(f'y_{i+1}')
            csv_writer.writerow(header)
            
            # Write the body parts and their points
            for part, points in self.points_dict.items():
                row = [part]
                for point in points:
                    row.extend(point)  # Append the x and y for each point
                csv_writer.writerow(row)
        return self.points_dict


cat_image_path = './cat.png'
csv_path = './cat_features.csv'
cat_process = ImageProcess(image_path=cat_image_path, csv_path=csv_path)
cat_process.select_features()

# def detect_features(image_path):
#     image = cv2.imread(image_path)
#     cv2.imshow('cat', image)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Initialize ORB detector
#     orb = cv2.ORB_create(nfeatures=500)
#     keypoints, _ = orb.detectAndCompute(gray, None)

#     # Draw keypoints on the image
#     keypoint_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
#     return keypoint_image, keypoints

# def check_permission(image_path):
#     print(os.path.exists('./cat.png'))
 
#     if os.access(image_path, os.R_OK):
#         print(f"Read permission is granted for {image_path}")
#     else:
#         print(f"Read permission is not granted for {image_path}")

#     # Get the current file permissions
#     current_permissions = os.stat(image_path).st_mode

#     # Calculate the new permissions by adding read permission for owner
#     new_permissions = current_permissions | 0o400  # 0o400 is octal representation for read permission for owner

#     # Set the new permissions
#     os.chmod(image_path, new_permissions)

