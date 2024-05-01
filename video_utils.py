import mediapipe as mp
import cv2
import csv
import os
import subprocess
import numpy as np

class ProcessVideo:
    def __init__(self, video_path, csv_path='dance_landmarks.csv'):
        self.video_path = video_path
        self.csv_path = csv_path
        self.landmarks = []

    def detect_landmarks(self):
        # Initialize MediaPipe Pose.
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        # Setup video capture.
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract landmarks.
                self.landmarks.append([landmark for landmark in results.pose_landmarks.landmark])

                # Optionally visualize the landmarks.
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow('Pose Estimation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        
        # Close pose detection.
        pose.close()
        return self.landmarks

    def save_to_csv(self):
        # Check if landmarks were detected.
        if not self.landmarks:
            print("No landmarks to save.")
            return
        
        # Setup CSV file and writer.
        with open(self.csv_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # Write headers to CSV.
            headers = ['landmark_{}_{}'.format(i, axis) for i in range(33) for axis in ['x', 'y', 'z', 'v']]
            csv_writer.writerow(headers)

            # Write landmarks to CSV.
            for landmark_frame in self.landmarks:
                row = [coord for landmark in landmark_frame for coord in (landmark.x, landmark.y, landmark.z, landmark.visibility)]
                csv_writer.writerow(row)

        print(f"Landmarks saved to {self.csv_path}")
    
    def get_audio(self, audio_path):
        # Command to extract audio using ffmpeg
        command = [
            'ffmpeg', '-i', self.video_path,     # Input video file path
            '-q:a', '0', '-map', 'a',       # Quality level for audio (0 is highest)
            audio_path,                     # Output audio file path
            '-y'                            # Overwrite output file if it exists
        ]

        # Execute the command
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"Audio extracted and saved to {audio_path}")
        return audio_path
    
    def attach_audio(self, audio_path, output_path):
        # Command to add/replace audio using ffmpeg
        command = [
            'ffmpeg', '-i', self.video_path,  # Input video file
            '-i', audio_path,            # Input audio file
            '-c:v', 'copy',              # Copy video stream directly without re-encoding
            '-map', '0:v:0',             # Map video stream from the first input to output
            '-map', '1:a:0',             # Map audio stream from the second input to output
            '-shortest',                 # If audio and video lengths differ, cut at the shortest
            output_path,                 # Output file path
            '-y'                         # Overwrite output file if it exists
        ]

        # Execute the command
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"Video with new audio saved to {output_path}")
        return output_path
    
    def remove_video_background(self):

        mp_selfie_segmentation = mp.solutions.selfie_segmentation

        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.video_path)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Instantiate the selfie segmentation model
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert the BGR image to RGB.
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image and generate the mask
                results = selfie_segmentation.process(image_rgb)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5

                # Set the background you want (e.g., plain white)
                bg_image = np.ones(frame.shape, dtype=np.uint8) * 255
                output_image = np.where(condition, frame, bg_image)

                # Write the frame into the file 'output_video.mp4'
                out.write(output_image)
                # Display the resulting frame
                cv2.imshow('Frame', output_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

video_path = './cropped_video.mp4'
csv_path = './dance_landmarks.csv'
audio_path = './audio.mp3'
output_path = './result.mp4'

landmark_saver = ProcessVideo(video_path, csv_path)
landmark_saver.detect_landmarks()  # Detect landmarks.
landmark_saver.save_to_csv()       # Save landmarks to CSV.
landmark_saver.get_audio(audio_path=audio_path)
