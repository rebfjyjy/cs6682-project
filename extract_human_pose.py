import mediapipe as mp
import cv2
import csv
import os

path = '/Users/hanyibei/Desktop/SP2024/6682/cs6682-project/croped_video.mp4'

def save_landmarks(path):
    # Initialize the pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Setup video capture
    cap = cv2.VideoCapture(path)

    # Setup CSV file and writer
    csv_path = os.path.join(os.getcwd(), 'dance_landmarks.csv')
    print(csv_path)
    with open(csv_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Write headers to CSV, 33 landmarks each with x, y, z and visibility (4 * 33 = 132 columns)
        headers = []
        for i in range(33):  # MediaPipe pose has 33 landmarks
            headers.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z', f'landmark_{i}_v'])
        csv_writer.writerow(headers)

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                # Prepare a row for the CSV file
                row = []
                for landmark in landmarks:
                    # Append x, y, z coordinates and visibility of each landmark to the row
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                # Write row to CSV
                csv_writer.writerow(row)

                # Optionally, visualize the landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow('Pose Estimation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
save_landmarks(path)
