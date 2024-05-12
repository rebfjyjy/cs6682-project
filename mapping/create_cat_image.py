import cv2
import numpy as np
import pandas as pd


class CreateCatImage:

    def __init__(self, dst_points_dict):
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

        self.src_points_dict = {}
        self.dst_points_dict = dst_points_dict
        self.body_parts = [
            "head", "body",
            "upper_left_arm", "lower_left_arm",
            "upper_right_arm", "lower_right_arm",
            "left_leg", "right_leg"
        ]
        # self.body_parts = [
        #     "head", "body"
        # ]

    def load_features_src_points(self):
        for part in self.body_parts:
            csv_file_path = f'./result/source_points/{part}_src_pts.csv'
            df = pd.read_csv(csv_file_path)

            # DataFrame `df` has columns named 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'x_4', 'y_4'
            # Convert these columns to a list of lists, each containing x and y coordinates of a point
            src_points = [
                [df['x_1'].values[0], df['y_1'].values[0]],  # Top-left corner of the image
                [df['x_2'].values[0], df['y_2'].values[0]],  # Top-right corner
                [df['x_3'].values[0], df['y_3'].values[0]],  # Bottom-right corner
                [df['x_4'].values[0], df['y_4'].values[0]]   # Bottom-left corner
            ]
            src_points = np.array(src_points, dtype=np.float32)
            self.src_points_dict[part] = src_points

    def create_white_paper(self):
        # Create a blank canvas (white paper)
        canvas = np.ones((self.height, self.width, 4), dtype=np.uint8) * 255
        return canvas

    # def warp_and_blend(self, part_image, src_points, dst_points, canvas, width, height):
    #     # Compute the homography matrix
    #     H, _ = cv2.findHomography(src_points, dst_points)
    #     # H, _ = cv2.getAffineTransform(src_points, dst_points)
    #
    #     # Warp the original image to fit the new perspective
    #     # warped_image = cv2.warpPerspective(part_image, H, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    #     warped_image = cv2.warpPerspective(part_image, H, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE, borderValue=(0, 0, 0, 0))
    #     # warped_image = cv2.warpAffine(part_image, H, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    #
    #     # Create a mask from the alpha channel of the warped image
    #     mask = warped_image[:, :, 3]  # Alpha channel
    #     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3 channel
    #     mask = mask / 255.0  # Normalize mask to range [0,1]
    #
    #     # Prepare the canvas (convert to float and normalize to range [0,1] for multiplication)
    #     canvas_float = canvas[:, :, :3].astype(float) / 255
    #
    #     # Convert warped image to float and normalize
    #     warped_float = warped_image[:, :, :3].astype(float) / 255
    #
    #     # Blend images using the mask
    #     combined = (canvas_float * (1 - mask)) + (warped_float * mask)
    #
    #     # Convert back to 8-bit
    #     combined = (combined * 255).astype(np.uint8)
    #
    #     # Put back the combined image on the canvas
    #     canvas[:combined.shape[0], :combined.shape[1], :3] = combined
    #
    #     return canvas

    def warp_and_blend(self, part_image, src_points, dst_points, canvas, width, height):
        # Compute the homography matrix
        src_points1 = src_points[[0, 1, 2]]
        # src_points2 = src_points[0, 2, 3]
        dst_points1 = dst_points[[0, 1, 2]]
        # dst_points2 = dst_points[0, 2, 3]
        # H, _ = cv2.findHomography(src_points1, dst_points1)
        # H2, _ = cv2.findHomography(src_points2, dst_points2)
        H = cv2.getAffineTransform(src_points1, dst_points1)


        # Warp the original image to fit the new perspective
        # warped_image = cv2.warpPerspective(part_image, H, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        warped_image = cv2.warpAffine(part_image, H, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        # warped_image2 = cv2.warpAffine(part_image, H2, (width, height), borderMode=cv2.BORDER_CONSTANT,
        #                               borderValue=(0, 0, 0, 0))

        # Create a mask from the alpha channel of the warped image
        mask = warped_image[:, :, 3]  # Alpha channel
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3 channel
        mask = mask / 255.0  # Normalize mask to range [0,1]

        # Prepare the canvas (convert to float and normalize to range [0,1] for multiplication)
        canvas_float = canvas[:, :, :3].astype(float) / 255

        # Convert warped image to float and normalize
        warped_float = warped_image[:, :, :3].astype(float) / 255

        # Blend images using the mask
        combined = (canvas_float * (1 - mask)) + (warped_float * mask)

        # Convert back to 8-bit
        combined = (combined * 255).astype(np.uint8)

        # Put back the combined image on the canvas
        canvas[:combined.shape[0], :combined.shape[1], :3] = combined

        return canvas


    def group_features(self):

        canvas = self.create_white_paper()

        for part in self.body_parts:
            image_path = f'./data/{part}.png'
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # Check if image is loaded
            if image is None:
                print("Error: Image could not be read.")
                exit()

            # Define the destination points on the canvas
            # These points define where the image corners should map to
            # dst_points = np.array([
            #     [845.840909090909,8.806818181818926],
            #     [1797.2045454545455,65.68181818181893],
            #     [1750.6704545454545,763.6931818181824],
            #     [938.909090909091,691.3068181818189]
            # ], dtype=np.float32)
            dst_points = self.dst_points_dict[part]

            # Define the source points from the image
            # src_points = np.array([
            #     [0, 0],  # Top-left corner of the image
            #     [image.shape[1], 0],  # Top-right corner
            #     [image.shape[1], image.shape[0]//3],  # Bottom-right corner
            #     [0, image.shape[0]//3]  # Bottom-left corner
            # ], dtype=np.float32)
            # src_points = np.array([
            #     [34.29193722943728, 20.13095238095184],  # Top-left corner of the image
            #     [838.430194805195, 24.819805194804758],  # Top-right corner
            #     [824.3636363636365, 718.7700216450214],  # Bottom-right corner
            #     [71.80275974025972, 711.7367424242423]  # Bottom-left corner
            # ], dtype=np.float32)
            src_points = self.src_points_dict[part]

            canvas = self.warp_and_blend(image, src_points, dst_points, canvas, self.width, self.height)

        return canvas


    def get_cat_image(self):
        self.load_features_src_points()
        output_image = self.group_features()
        return output_image


# if __name__ == "__main__":

#     # dst_points = np.array([
#     #             [845.840909090909,8.806818181818926],
#     #             [1797.2045454545455,65.68181818181893],
#     #             [1750.6704545454545,763.6931818181824],
#     #             [938.909090909091,691.3068181818189]
#     #         ], dtype=np.float32)
#     # dst_points_dict = {}
#     # dst_points_dict['head'] = dst_points
#     dst_points_dict = {........}
#     create_cat = CreateCatImage(dst_points_dict)
#     output_image = create_cat.get_cat_image()

#     # Display the final output
#     cv2.imshow('Warped Image on Canvas', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



    # # ---------------------------
    # # head part
    # image_path = './data/head.png'
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # # Check if image is loaded
    # if image is None:
    #     print("Error: Image could not be read.")
    #     exit()

    # # Define the destination points on the canvas
    # # These points define where the image corners should map to
    # dst_points = np.array([
    #     [845.840909090909,8.806818181818926],
    #     [1797.2045454545455,65.68181818181893],
    #     [1750.6704545454545,763.6931818181824],
    #     [938.909090909091,691.3068181818189]
    #     # [100, 200],  # Top-left corner
    #     # [1100, 200],  # Top-right corner
    #     # [1100, 600],  # Bottom-right corner
    #     # [100, 600]   # Bottom-left corner
    # ], dtype=np.float32)

    # # Define the source points from the image
    # # src_points = np.array([
    # #     [0, 0],  # Top-left corner of the image
    # #     [image.shape[1], 0],  # Top-right corner
    # #     [image.shape[1], image.shape[0]//3],  # Bottom-right corner
    # #     [0, image.shape[0]//3]  # Bottom-left corner
    # # ], dtype=np.float32)
    # src_points = np.array([
    #     [34.29193722943728, 20.13095238095184],  # Top-left corner of the image
    #     [838.430194805195, 24.819805194804758],  # Top-right corner
    #     [824.3636363636365, 718.7700216450214],  # Bottom-right corner
    #     [71.80275974025972, 711.7367424242423]  # Bottom-left corner
    # ], dtype=np.float32)

    # canvas = warp_and_blend(image, src_points, dst_points, canvas, width, height)



    # # --------------
    # # body part
    # image_path = './data/body.png'
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # # Check if image is loaded
    # if image is None:
    #     print("Error: Image could not be read.")
    #     exit()

    # # Define the destination points on the canvas
    # # These points define where the image corners should map to
    # dst_points = np.array([
    #     [918.227272727273,722.329545454546],
    #     [1755.840909090909,748.1818181818189],
    #     [1952.318181818182,3493.693181818183],
    #     [1119.875,3679.8295454545464]
    # ], dtype=np.float32)

    # # Define the source points from the image
    # # src_points = np.array([
    # #     [0, 0],  # Top-left corner of the image
    # #     [image.shape[1], 0],  # Top-right corner
    # #     [image.shape[1], image.shape[0]],  # Bottom-right corner
    # #     [0, image.shape[0]]  # Bottom-left corner
    # # ], dtype=np.float32)
    # src_points = np.array([
    #     [443.9220779220775, 175.12662337662323],  # Top-left corner of the image
    #     [1658.873376623376, 191.1655844155839],  # Top-right corner
    #     [1839.311688311688, 2925.8084415584417],  # Bottom-right corner
    #     [560.2045454545448, 2909.7694805194806]  # Bottom-left corner
    # ], dtype=np.float32)

    # canvas = warp_and_blend(image, src_points, dst_points, canvas, width, height)
    # # ---------------

    # # Merge the warped image with the canvas
    # # This step isn't strictly necessary as warpPerspective already fills the area, but if needed:
    # # canvas[dst_points[:,1].min():dst_points[:,1].max(), dst_points[:,0].min():dst_points[:,0].max()] = warped_image[dst_points[:,1].min():dst_points[:,1].max(), dst_points[:,0].min():dst_points[:,0].max()]

    # # cv2.imwrite('test_image.png', warped_image)

    # # Display the final output
    # cv2.imshow('Warped Image on Canvas', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
