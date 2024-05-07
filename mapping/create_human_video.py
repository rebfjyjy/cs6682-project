import cv2
import os

def create_video_from_frames(frame_folder, output_video_path, fps=10, frame_size=(1710, 1080)):  # frame_size = (width, height)
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs like 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    # Sort frame files to ensure they are in the correct order
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    for _ in range(30):   # repeat cycle
        # Loop through each frame file
        for frame_file in frame_files:
            frame_path = os.path.join(frame_folder, frame_file)
            frame = cv2.imread(frame_path)  # Read the frame image
            # print(frame.shape)
            if frame is not None:
                # frame = cv2.resize(frame, frame_size)  # Resize the frame if necessary
                # print(frame.shape)
                out.write(frame)  # Write the frame into the video
            else:
                print(f"Warning: Could not read frame {frame_file}")
    
    # Release everything when job is finished
    out.release()
    print("The video was successfully created.")

# Path to the folder containing frame images
frame_folder = './data/human_frames'
# Path to the output video file
output_video_path = './result/human_video.mp4'

# Create the video
width = 1710
height = 1080
create_video_from_frames(frame_folder, output_video_path, fps=6, frame_size=(width, height))




# import cv2
# import numpy as np

# # Set video width and height
# width = 1710
# height = 1080

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('./result/human_video.mp4', fourcc, 30, (width, height))

# # Generate frames (Here, I'm just creating a blank black frame for demonstration)
# for i in range(100):
#     frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # Create a blank black frame
#     out.write(frame)  # Write the frame to the video file

# # Release everything when done
# out.release()
# cv2.destroyAllWindows()

