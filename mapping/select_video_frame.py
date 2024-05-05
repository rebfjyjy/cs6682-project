import cv2

# Open the video file
video_path = './data/croped_video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

selected_frames = [1, 4, 10, 16]

for frame_number in selected_frames:
    # Move to the 1st frame.
    # frame_number = 1  # 1st frame, 1, 4, 10, 16, 1, 4, 10, 16, ...
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the specific frame
    ret, frame = cap.read()

    # # Display the final output
    # cv2.imshow('Image', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Check if the frame has been captured
    if ret:
        cv2.imwrite(f'./data/human_frames/human_{frame_number}.png', frame)
        print(f"The {frame_number}st frame extracted and saved.")
    else:
        print("Error: Could not read the {frame_number}th frame from video.")

cap.release()
