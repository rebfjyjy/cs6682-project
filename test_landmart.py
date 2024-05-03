import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt


def body_test():
    df_body = pd.read_csv('body.csv')
    df_head = pd.read_csv('head.csv')
    df_left_leg = pd.read_csv('left_leg.csv')
    df_right_leg = pd.read_csv('right_leg.csv')
    df_upper_left_arm = pd.read_csv('upper_left_arm.csv')
    df_lower_left_arm = pd.read_csv('lower_left_arm.csv')
    df_upper_right_arm = pd.read_csv('upper_right_arm.csv')
    df_lower_right_arm = pd.read_csv('lower_right_arm.csv')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('triangle_video.avi', fourcc, 20.0, (640, 480))

    # for index in range(100):
    for index in range(len(df_head)):
        # Create a plot
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='datalim')

        # Plot head rectangle
        head_x = [df_head.loc[index, f'head_{i}_x'] for i in range(1, 5)]
        head_y = [df_head.loc[index, f'head_{i}_y'] for i in range(1, 5)]
        head_x.append(head_x[0])  # Close the rectangle by repeating the first point
        head_y.append(head_y[0])
        ax.plot(head_x, head_y, 'ro-')  # Red rectangle for head

        # Plot body rectangle
        body_x = [df_body.loc[index, f'body_{i}_x'] for i in range(1, 5)]
        body_y = [df_body.loc[index, f'body_{i}_y'] for i in range(1, 5)]
        body_x.append(body_x[0])  # Close the rectangle by repeating the first point
        body_y.append(body_y[0])
        ax.plot(body_x, body_y, 'bo-')  # Blue rectangle for body

        # Plot left_leg rectangle
        left_leg_x = [df_left_leg.loc[index, f'left_leg_{i}_x'] for i in range(1, 5)]
        left_leg_y = [df_left_leg.loc[index, f'left_leg_{i}_y'] for i in range(1, 5)]
        left_leg_x.append(left_leg_x[0])  # Close the rectangle by repeating the first point
        left_leg_y.append(left_leg_y[0])
        ax.plot(left_leg_x, left_leg_y, 'go-')  # Green rectangle for left_leg

        # Plot right_leg rectangle
        right_leg_x = [df_right_leg.loc[index, f'right_leg_{i}_x'] for i in range(1, 5)]
        right_leg_y = [df_right_leg.loc[index, f'right_leg_{i}_y'] for i in range(1, 5)]
        right_leg_x.append(right_leg_x[0])  # Close the rectangle by repeating the first point
        right_leg_y.append(right_leg_y[0])
        ax.plot(right_leg_x, right_leg_y, 'go-')  # Green rectangle for right_leg

        # Plot upper_left_arm rectangle
        upper_left_arm_x = [df_upper_left_arm.loc[index, f'upper_left_arm_{i}_x'] for i in range(1, 5)]
        upper_left_arm_y = [df_upper_left_arm.loc[index, f'upper_left_arm_{i}_y'] for i in range(1, 5)]
        upper_left_arm_x.append(upper_left_arm_x[0])  # Close the rectangle by repeating the first point
        upper_left_arm_y.append(upper_left_arm_y[0])
        ax.plot(upper_left_arm_x, upper_left_arm_y, 'yo-')  # Green rectangle for right_leg

        # Plot lower_left_arm rectangle
        lower_left_arm_x = [df_lower_left_arm.loc[index, f'lower_left_arm_{i}_x'] for i in range(1, 5)]
        lower_left_arm_y = [df_lower_left_arm.loc[index, f'lower_left_arm_{i}_y'] for i in range(1, 5)]
        lower_left_arm_x.append(lower_left_arm_x[0])  # Close the rectangle by repeating the first point
        lower_left_arm_y.append(lower_left_arm_y[0])
        ax.plot(lower_left_arm_x, lower_left_arm_y, 'yo-')  # Green rectangle for right_leg

        # Plot upper_right_arm rectangle
        upper_right_arm_x = [df_upper_right_arm.loc[index, f'upper_right_arm_{i}_x'] for i in range(1, 5)]
        upper_right_arm_y = [df_upper_right_arm.loc[index, f'upper_right_arm_{i}_y'] for i in range(1, 5)]
        upper_right_arm_x.append(upper_right_arm_x[0])  # Close the rectangle by repeating the first point
        upper_right_arm_y.append(upper_right_arm_y[0])
        ax.plot(upper_right_arm_x, upper_right_arm_y, 'yo-')  # Green rectangle for right_leg

        # Plot lower_right_arm rectangle
        lower_right_arm_x = [df_lower_right_arm.loc[index, f'lower_right_arm_{i}_x'] for i in range(1, 5)]
        lower_right_arm_y = [df_lower_right_arm.loc[index, f'lower_right_arm_{i}_y'] for i in range(1, 5)]
        lower_right_arm_x.append(lower_right_arm_x[0])  # Close the rectangle by repeating the first point
        lower_right_arm_y.append(lower_right_arm_y[0])
        ax.plot(lower_right_arm_x, lower_right_arm_y, 'yo-')  # Green rectangle for right_leg

        # Draw triangle (assuming points are in order 1->2->3->4->1)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert colors from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write frame to video
        video.write(img)

        # Close the plot figure to free memory
        plt.close(fig)

    # Release the video writer
    video.release()


if __name__ == "__main__":
    body_test()
