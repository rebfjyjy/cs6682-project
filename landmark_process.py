import pandas as pd


def generate_transfer():
    file_path = "./dance_landmarks.csv"
    try:
        body = {
            "body_1_x": [],
            "body_1_y": [],
            "body_2_x": [],
            "body_2_y": [],
            "body_3_x": [],
            "body_3_y": [],
            "body_4_x": [],
            "body_4_y": [],
        }
        head = {
            "head_1_x": [],
            "head_1_y": [],
            "head_2_x": [],
            "head_2_y": [],
            "head_3_x": [],
            "head_3_y": [],
            "head_4_x": [],
            "head_4_y": [],
        }

        left_leg = {
            "left_leg_1_x": [],
            "left_leg_1_y": [],
            "left_leg_2_x": [],
            "left_leg_2_y": [],
            "left_leg_3_x": [],
            "left_leg_3_y": [],
            "left_leg_4_x": [],
            "left_leg_4_y": [],
        }

        right_leg = {
            "right_leg_1_x": [],
            "right_leg_1_y": [],
            "right_leg_2_x": [],
            "right_leg_2_y": [],
            "right_leg_3_x": [],
            "right_leg_3_y": [],
            "right_leg_4_x": [],
            "right_leg_4_y": [],
        }

        right_leg = {
            "right_leg_1_x": [],
            "right_leg_1_y": [],
            "right_leg_2_x": [],
            "right_leg_2_y": [],
            "right_leg_3_x": [],
            "right_leg_3_y": [],
            "right_leg_4_x": [],
            "right_leg_4_y": [],
        }

        lower_left_arm = {
            "lower_left_arm_1_x": [],
            "lower_left_arm_1_y": [],
            "lower_left_arm_2_x": [],
            "lower_left_arm_2_y": [],
            "lower_left_arm_3_x": [],
            "lower_left_arm_3_y": [],
            "lower_left_arm_4_x": [],
            "lower_left_arm_4_y": [],
        }
        upper_left_arm = {
            "upper_left_arm_1_x": [],
            "upper_left_arm_1_y": [],
            "upper_left_arm_2_x": [],
            "upper_left_arm_2_y": [],
            "upper_left_arm_3_x": [],
            "upper_left_arm_3_y": [],
            "upper_left_arm_4_x": [],
            "upper_left_arm_4_y": [],
        }

        lower_right_arm = {
            "lower_right_arm_1_x": [],
            "lower_right_arm_1_y": [],
            "lower_right_arm_2_x": [],
            "lower_right_arm_2_y": [],
            "lower_right_arm_3_x": [],
            "lower_right_arm_3_y": [],
            "lower_right_arm_4_x": [],
            "lower_right_arm_4_y": [],
        }
        upper_right_arm = {
            "upper_right_arm_1_x": [],
            "upper_right_arm_1_y": [],
            "upper_right_arm_2_x": [],
            "upper_right_arm_2_y": [],
            "upper_right_arm_3_x": [],
            "upper_right_arm_3_y": [],
            "upper_right_arm_4_x": [],
            "upper_right_arm_4_y": [],
        }

        df = pd.read_csv(file_path)
        row_num = len(df)
        for row in range(row_num):
            # extract body
            body_1_x = df.loc[row, 'landmark_12_x']
            body_1_y = 1-df.loc[row, 'landmark_12_y']
            body_2_x = df.loc[row, 'landmark_11_x']
            body_2_y = 1-df.loc[row, 'landmark_11_y']
            body_3_x = (df.loc[row, 'landmark_25_x'] + df.loc[row, 'landmark_23_x']) / 2
            body_3_y = 1-df.loc[row, 'landmark_25_y']
            body_4_x = (df.loc[row, 'landmark_26_x'] + df.loc[row, 'landmark_24_x']) / 2
            body_4_y = 1-df.loc[row, 'landmark_26_y']
            body['body_1_x'].append(body_1_x)
            body['body_1_y'].append(body_1_y)
            body['body_2_x'].append(body_2_x)
            body['body_2_y'].append(body_2_y)
            body['body_3_x'].append(body_3_x)
            body['body_3_y'].append(body_3_y)
            body['body_4_x'].append(body_4_x)
            body['body_4_y'].append(body_4_y)

            # extract head
            head_1_x = df.loc[row, 'landmark_8_x']
            head_1_y = 1-df.loc[row, 'landmark_5_y']
            head_2_x = df.loc[row, 'landmark_7_x']
            head_2_y = 1-df.loc[row, 'landmark_2_y']
            head_3_x = df.loc[row, 'landmark_7_x']
            head_3_y = body_2_y
            head_4_x = df.loc[row, 'landmark_8_x']
            head_4_y = body_1_y
            head['head_1_x'].append(head_1_x)
            head['head_1_y'].append(head_1_y)
            head['head_2_x'].append(head_2_x)
            head['head_2_y'].append(head_2_y)
            head['head_3_x'].append(head_3_x)
            head['head_3_y'].append(head_3_y)
            head['head_4_x'].append(head_4_x)
            head['head_4_y'].append(head_4_y)

            # extract left leg
            left_leg_1_x = body_4_x
            left_leg_1_y = body_4_y
            left_leg_2_x = body_4_x - (body_4_x - body_3_x)/3
            left_leg_2_y = body_4_y - (body_4_y - body_3_y)/3
            left_leg_3_x = df.loc[row, 'landmark_30_x']
            left_leg_3_y = 1 - df.loc[row, 'landmark_30_y']
            left_leg_4_x = df.loc[row, 'landmark_32_x']
            left_leg_4_y = 1 - df.loc[row, 'landmark_32_y']
            left_leg['left_leg_1_x'].append(left_leg_1_x)
            left_leg['left_leg_1_y'].append(left_leg_1_y)
            left_leg['left_leg_2_x'].append(left_leg_2_x)
            left_leg['left_leg_2_y'].append(left_leg_2_y)
            left_leg['left_leg_3_x'].append(left_leg_3_x)
            left_leg['left_leg_3_y'].append(left_leg_3_y)
            left_leg['left_leg_4_x'].append(left_leg_4_x)
            left_leg['left_leg_4_y'].append(left_leg_4_y)

            # extract left leg
            right_leg_1_x = body_3_x - (body_3_x - body_4_x) / 3
            right_leg_1_y = body_3_y - (body_3_y - body_4_y) / 3
            right_leg_2_x = body_3_x
            right_leg_2_y = body_3_y
            right_leg_3_x = df.loc[row, 'landmark_31_x']
            right_leg_3_y = 1 - df.loc[row, 'landmark_31_y']
            right_leg_4_x = df.loc[row, 'landmark_29_x']
            right_leg_4_y = 1 - df.loc[row, 'landmark_29_y']
            right_leg['right_leg_1_x'].append(right_leg_1_x)
            right_leg['right_leg_1_y'].append(right_leg_1_y)
            right_leg['right_leg_2_x'].append(right_leg_2_x)
            right_leg['right_leg_2_y'].append(right_leg_2_y)
            right_leg['right_leg_3_x'].append(right_leg_3_x)
            right_leg['right_leg_3_y'].append(right_leg_3_y)
            right_leg['right_leg_4_x'].append(right_leg_4_x)
            right_leg['right_leg_4_y'].append(right_leg_4_y)

            # extract upper left arm
            upper_left_arm_1_x = df.loc[row, 'landmark_14_x']
            upper_left_arm_1_y = 1 - df.loc[row, 'landmark_14_y'] - 0.01
            upper_left_arm_2_x = df.loc[row, 'landmark_14_x']
            upper_left_arm_2_y = upper_left_arm_1_y + 0.02
            upper_left_arm_3_x = body_1_x
            upper_left_arm_3_y = body_1_y
            m = (body_4_y - body_1_y) / (body_4_x - body_1_x)
            upper_left_arm_4_y = upper_left_arm_3_y - 0.02
            upper_left_arm_4_x = body_1_x + (upper_left_arm_4_y - body_1_y) / m
            upper_left_arm['upper_left_arm_1_x'].append(upper_left_arm_1_x)
            upper_left_arm['upper_left_arm_1_y'].append(upper_left_arm_1_y)
            upper_left_arm['upper_left_arm_2_x'].append(upper_left_arm_2_x)
            upper_left_arm['upper_left_arm_2_y'].append(upper_left_arm_2_y)
            upper_left_arm['upper_left_arm_3_x'].append(upper_left_arm_3_x)
            upper_left_arm['upper_left_arm_3_y'].append(upper_left_arm_3_y)
            upper_left_arm['upper_left_arm_4_x'].append(upper_left_arm_4_x)
            upper_left_arm['upper_left_arm_4_y'].append(upper_left_arm_4_y)

            # extract lower left arm
            lower_left_arm_1_x = df.loc[row, 'landmark_18_x']
            lower_left_arm_1_y = 1 - df.loc[row, 'landmark_18_y']
            lower_left_arm_2_x = df.loc[row, 'landmark_20_x']
            lower_left_arm_2_y = 1 - df.loc[row, 'landmark_20_y']
            lower_left_arm_3_x = upper_left_arm_2_x
            lower_left_arm_3_y = upper_left_arm_2_y
            lower_left_arm_4_x = upper_left_arm_1_x
            lower_left_arm_4_y = upper_left_arm_1_y
            lower_left_arm['lower_left_arm_1_x'].append(lower_left_arm_1_x)
            lower_left_arm['lower_left_arm_1_y'].append(lower_left_arm_1_y)
            lower_left_arm['lower_left_arm_2_x'].append(lower_left_arm_2_x)
            lower_left_arm['lower_left_arm_2_y'].append(lower_left_arm_2_y)
            lower_left_arm['lower_left_arm_3_x'].append(lower_left_arm_3_x)
            lower_left_arm['lower_left_arm_3_y'].append(lower_left_arm_3_y)
            lower_left_arm['lower_left_arm_4_x'].append(lower_left_arm_4_x)
            lower_left_arm['lower_left_arm_4_y'].append(lower_left_arm_4_y)

            # extract upper right arm
            upper_right_arm_1_x = df.loc[row, 'landmark_13_x']
            upper_right_arm_1_y = 1 - df.loc[row, 'landmark_13_y'] + 0.01
            upper_right_arm_2_x = df.loc[row, 'landmark_13_x']
            upper_right_arm_2_y = upper_right_arm_1_y - 0.02
            upper_right_arm_4_x = body_2_x
            upper_right_arm_4_y = body_2_y
            m = (body_3_y - body_2_y) / (body_3_x - body_2_x)
            upper_right_arm_3_y = upper_right_arm_4_y - 0.02
            upper_right_arm_3_x = body_2_x + (upper_right_arm_3_y - body_2_y) / m
            upper_right_arm['upper_right_arm_1_x'].append(upper_right_arm_1_x)
            upper_right_arm['upper_right_arm_1_y'].append(upper_right_arm_1_y)
            upper_right_arm['upper_right_arm_2_x'].append(upper_right_arm_2_x)
            upper_right_arm['upper_right_arm_2_y'].append(upper_right_arm_2_y)
            upper_right_arm['upper_right_arm_3_x'].append(upper_right_arm_3_x)
            upper_right_arm['upper_right_arm_3_y'].append(upper_right_arm_3_y)
            upper_right_arm['upper_right_arm_4_x'].append(upper_right_arm_4_x)
            upper_right_arm['upper_right_arm_4_y'].append(upper_right_arm_4_y)

            # extract lower right arm
            lower_right_arm_1_x = df.loc[row, 'landmark_19_x']
            lower_right_arm_1_y = 1 - df.loc[row, 'landmark_19_y']
            lower_right_arm_2_x = df.loc[row, 'landmark_17_x']
            lower_right_arm_2_y = 1 - df.loc[row, 'landmark_17_y']
            lower_right_arm_3_x = upper_right_arm_2_x
            lower_right_arm_3_y = upper_right_arm_2_y
            lower_right_arm_4_x = upper_right_arm_1_x
            lower_right_arm_4_y = upper_right_arm_1_y
            lower_right_arm['lower_right_arm_1_x'].append(lower_right_arm_1_x)
            lower_right_arm['lower_right_arm_1_y'].append(lower_right_arm_1_y)
            lower_right_arm['lower_right_arm_2_x'].append(lower_right_arm_2_x)
            lower_right_arm['lower_right_arm_2_y'].append(lower_right_arm_2_y)
            lower_right_arm['lower_right_arm_3_x'].append(lower_right_arm_3_x)
            lower_right_arm['lower_right_arm_3_y'].append(lower_right_arm_3_y)
            lower_right_arm['lower_right_arm_4_x'].append(lower_right_arm_4_x)
            lower_right_arm['lower_right_arm_4_y'].append(lower_right_arm_4_y)

        body_df = pd.DataFrame(body)
        body_df.to_csv('body.csv', index=False)
        head_df = pd.DataFrame(head)
        head_df.to_csv('head.csv', index=False)
        left_leg_df = pd.DataFrame(left_leg)
        left_leg_df.to_csv('left_leg.csv', index=False)
        right_leg_df = pd.DataFrame(right_leg)
        right_leg_df.to_csv('right_leg.csv', index=False)
        upper_left_arm_df = pd.DataFrame(upper_left_arm)
        upper_left_arm_df.to_csv('upper_left_arm.csv', index=False)
        lower_left_arm_df = pd.DataFrame(lower_left_arm)
        lower_left_arm_df.to_csv('lower_left_arm.csv', index=False)
        upper_right_arm_df = pd.DataFrame(upper_right_arm)
        upper_right_arm_df.to_csv('upper_right_arm.csv', index=False)
        lower_right_arm_df = pd.DataFrame(lower_right_arm)
        lower_right_arm_df.to_csv('lower_right_arm.csv', index=False)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    generate_transfer()
