import cv2
import numpy as np
import pandas as pd
from create_cat_image import CreateCatImage
import os


class CreateCatVideo:

    def __init__(self):
        # frame size: cat image
        cat_path = './data/cat.png'
        cat_image = cv2.imread(cat_path)
        self.height = cat_image.shape[0]
        self.width = cat_image.shape[1]

        # frame size: frame in human dancing video
        human_path = './data/human.png'
        human_image = cv2.imread(human_path)
        self.height = human_image.shape[0]
        self.width = human_image.shape[1]

        # print(self.height)
        # print(self.width)

        self.body_parts = [
            "head", "body", 
            "upper_left_arm", "lower_left_arm",
            "upper_right_arm", "lower_right_arm", 
            "left_leg", "right_leg"
        ]
        # self.body_parts = [
        #     "head", "body"
        # ]
        self.all_frames_dst_pts = {}     # {"head": [[frame_1], frame_2, ...], "body": [[frame_1], frame_2, ...], "upper_left_arm": [...]}

    # # dst pnts base on square little man 
    def load_features_dst_points_all_frames(self):
        for part in self.body_parts:
            csv_file_path = f'./result/dst_points/{part}.csv' 
            df = pd.read_csv(csv_file_path)

            # DataFrame `df` has columns named 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'x_4', 'y_4'
            # Convert these columns to a list of lists, each containing x and y coordinates of a point
            self.all_frames_dst_pts[part] = []
            for i, row in df.iterrows():
                # if i > 20:  # select frame size
                #     break
                offset = (max(self.height, self.width) - min(self.height, self.width)) / 2  # move cat to the center
                dst_points = [
                    [row[f'{part}_1_x'] * self.height + offset, self.height - row[f'{part}_1_y'] * self.height],  # Top-left corner of the image
                    [row[f'{part}_2_x'] * self.height + offset, self.height - row[f'{part}_2_y'] * self.height],  # Top-right corner
                    [row[f'{part}_3_x'] * self.height + offset, self.height - row[f'{part}_3_y'] * self.height],  # Bottom-right corner
                    [row[f'{part}_4_x'] * self.height + offset, self.height - row[f'{part}_4_y'] * self.height]   # Bottom-left corner
                ]
                # dst_points = [
                #     [row[f'{part}_1_x'] * self.width, row[f'{part}_1_y'] * self.height],
                #     [row[f'{part}_2_x'] * self.width, row[f'{part}_2_y'] * self.height],  
                #     [row[f'{part}_3_x'] * self.width, row[f'{part}_3_y'] * self.height],
                #     [row[f'{part}_4_x'] * self.width, row[f'{part}_4_y'] * self.height],  
                # ]
                dst_points = np.array(dst_points, dtype=np.float32)
                self.all_frames_dst_pts[part].append(dst_points)

    def create_dst_points_dict_square_man(self, frame_num):
        dst_points_dict = {}
        for part in self.body_parts:
            dst_points_dict[part] = self.all_frames_dst_pts[part][frame_num]
        return dst_points_dict
    
    def create_dst_points_dict_original_man(self, frame_num):
        file_path = f'./result/human_features_diff_frames/human_features_{frame_num}.csv'  # Update this to the path of your CSV file
        df = pd.read_csv(file_path)

        body_parts = {}

        for index, row in df.iterrows():
            # print(row)
            part_name = row['Unnamed: 0']  # Assumes the part names are in the first column which is unnamed
            # Collect the coordinates into a NumPy array and reshape to match the required shape (4, 2)
            coordinates = np.array([
                [row['x_1'], row['y_1']],
                [row['x_2'], row['y_2']],
                [row['x_3'], row['y_3']],
                [row['x_4'], row['y_4']]
            ])
            body_parts[part_name] = coordinates
        # print(body_parts)
        return body_parts


    def one_frame_cat(self, frame_num, dst_type):
        if dst_type == 'square_man':
            dst_points_dict = self.create_dst_points_dict_square_man(frame_num=frame_num)   # dst pnts from square little man
        elif dst_type == 'original_man':
            dst_points_dict = self.create_dst_points_dict_original_man(frame_num=frame_num)    # dst pnts from original man in the video
        # print(dst_points_dict)
        catImage = CreateCatImage(dst_points_dict)
        output_image = catImage.get_cat_image()
        return output_image
    

    def create_video_from_frames_original_man(self, output_video_path, fps=10, frame_size=(1710, 1080)):   # frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        selected_frames = [1, 4, 10, 16]
        hashmap = {}
        for frame_num in selected_frames:  # Loop through each frame file
            frame = self.one_frame_cat(frame_num, 'original_man')
            # print('1', frame.shape)
            # print(type(frame))
            # print(frame)
            if frame is not None:
                # Ensure the frame is in the correct format and size
                if frame.shape[2] == 4:  # Check if the frame includes an alpha channel
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
                    # print('2', frame.shape)
                # if frame.shape[:2] != frame_size:
                #     frame = cv2.resize(frame, frame_size)  # Resize the frame to match the VideoWriter's size
                #     print('3', frame.shape)
                out.write(frame)  # Write the frame into the video
                hashmap[frame_num] = frame
            else:
                print(f"Warning: Could not create frame")

        for _ in range(30):  # video repeat cycle
            for frame_num in selected_frames:
                frame = hashmap[frame_num]
                out.write(frame)
        
        out.release()
        print("The video was successfully created.")


    def create_video_from_frames_square_man(self, output_video_path, fps=10, frame_size=(1710, 1080)):   # frame_size = (width, height)
        self.load_features_dst_points_all_frames()  # dst pnts base on square little man only
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        video_total_frame_num = len(self.all_frames_dst_pts['head'])
        # print(video_total_frame_num)  # 2509
        video_total_frame_num = 60     # comment out this line if want to create the full-length video

        for frame_num in range(video_total_frame_num):  # Loop through each frame
            frame = self.one_frame_cat(frame_num, 'square_man')
            print(f'Frame {frame_num}st finished')

            if frame is not None:
                if frame.shape[2] == 4:  # Check if the frame includes an alpha channel
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
                out.write(frame)  # Write the frame into the video
            else:
                print(f"Warning: Could not create frame")
        
        out.release()
        print("The video was successfully created.")



if __name__ == "__main__":
    dst_type = 'original_man'
    # dst_type = 'square_man'

    catVideo = CreateCatVideo()
    frame_size = (catVideo.width, catVideo.height)

    if dst_type == 'original_man':
        output_video_path = './result/cat_video_original.mp4'
        catVideo.create_video_from_frames_original_man(output_video_path, fps=6, frame_size=frame_size)
    elif dst_type == 'square_man':
        output_video_path = './result/cat_video_square.mp4'
        catVideo.create_video_from_frames_square_man(output_video_path, fps=6, frame_size=frame_size)
    
    # # Display the final output
    # cv2.imshow('Warped Image on Canvas', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




























# ------------------------
# import cv2
# import numpy as np

# # Example human and cat points. You need to ensure these are correctly mapped and extracted from your datasets.
# human_points = np.array([
#     [620.6183668235853,106.16862051911403],
#     [636.4445716483799,86.38586448812111],
#     [743.2714542157419,216.9520542926748],
#     [727.4452493909475,239.70222372831677]  # Example points for the lower_left_arm
# ], dtype=np.float32)

# cat_points = np.array([
#     [602.8295454545455,1094.6022727272734],
#     [933.7386363636365,686.1363636363644],
#     [933.7386363636365,1523.750000000001],
#     [706.2386363636365,1725.397727272728]  # Example points for the lower_left_arm
# ], dtype=np.float32)

# # Calculate the homography matrix
# H, status = cv2.findHomography(human_points, cat_points, cv2.RANSAC, 5.0)

# # Example use of the homography matrix to transform a point
# # Transforming the first point of the human head to cat coordinate space
# human_point = np.array([[[620.6183668235853,106.16862051911403]]], dtype='float32')  # Double brackets to match expected 3D array
# transformed_point = cv2.perspectiveTransform(human_point, H)

# print("Homography Matrix:\n", H)
# print("Transformed Point:", transformed_point)


# -------------------------------------------

# import cv2
# import numpy as np

# # Sample points extracted manually from your description, replace these with actual values from your data
# # Human features points [x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4]
# human_points = np.array([
#     [835.261, 179.376, 902.522, 181.343, 898.566, 265.419, 845.152, 272.343],  # head
#     [928.240, 201.637, 981.653, 208.049, 1011.327, 200.137, 967.805, 209.812]   # upper_right_arm
#     # Add more corresponding points for better accuracy
# ])

# # Cat features points [point1_x, 1_y, point2_x, 2_y, point3_x, 3_y, point4_x, 4_y]
# cat_points = np.array([
#     [845.840, 8.806, 1797.204, 65.681, 938.909, 691.306, 1750.670, 763.693],  # head
#     [1916.125, 1203.181, 2474.534, 779.204, 2521.068, 1094.602, 1973.0, 1539.261]  # upper_right_arm
#     # Add more corresponding points for better accuracy
# ])

# # Calculate the homography matrix
# H, status = cv2.findHomography(human_points, cat_points, cv2.RANSAC, 5.0)

# # Example use of the homography matrix to transform a point
# # Transforming the first point of human head to cat coordinate space
# human_point = np.array([[human_points[0, :2]]], dtype='float32')
# transformed_point = cv2.perspectiveTransform(human_point, H)

# print("Homography Matrix:\n", H)
# print("Transformed Point:", transformed_point)

