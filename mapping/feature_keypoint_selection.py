import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

class FeaturesImageProcess:
    def __init__(self, part, image_path, csv_path):
        self.part = part
        self.image_path = image_path
        self.csv_path = csv_path
        self.src_points = []

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

        plt.title(f'Select the {self.part} keypoints')
        points = plt.ginput(n=4, timeout=0)
        self.draw_points(ax, points, 'red')
        plt.draw()
        self.src_points = points
        print(f'{self.part} points:', points)

        plt.close(fig)
        return self.src_points

    def save_points(self):
        # Write the points to a CSV file
        with open(self.csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write the header
            header = [''] 
            for i in range(4):  # each part has exactly 4 points
                header.append(f'x_{i+1}')
                header.append(f'y_{i+1}')
            csv_writer.writerow(header)
            
            # Write the body parts and their points
            row = [self.part]
            for point in self.src_points:
                row.extend(point)  # Append the x and y for each point
            csv_writer.writerow(row)
        return self.src_points



if __name__ == "__main__":

    # body_parts = [
    #     "head", "body", 
    #     "upper_left_arm", "lower_left_arm",
    #     "upper_right_arm", "lower_right_arm", 
    #     "left_leg", "right_leg"
    # ]

    body_parts = ["lower_left_arm", "lower_right_arm"]  # select keypoint for single part

    for part in body_parts:
        # cat keypoint selection
        image_path = f'./data/{part}.png'
        csv_path = f'./result/source_points/{part}_src_pts.csv'

        cat_process = FeaturesImageProcess(part=part, image_path=image_path, csv_path=csv_path)
        cat_process.select_features()
        cat_process.save_points()