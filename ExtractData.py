import cv2
import mediapipe as mp
import pandas as pd
import os
import re  # Import regex library for filename processing

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


def process_video(video_file):
    print(f"Processing video: {video_file}")
    cap = cv2.VideoCapture(video_file)
    # Extract gesture class from filename using regex
    class_name = re.match(r"([a-z ]+)[0-9]+", os.path.splitext(os.path.basename(video_file))[0], re.I).group(1)
    all_keypoints = []  # Store keypoints for all frames

    frame_count = 0  # Add a frame counter
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        frame_count += 1  # Update frame count
        if frame_count % 100 == 0:  # Log every 100 frames
            print(f"    Processed {frame_count} frames...")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y])
                keypoints.append(class_name.strip())  # Append class name as the last element, stripping any spaces
                all_keypoints.append(keypoints)

    cap.release()
    print(f"Finished processing {frame_count} frames for video: {video_file}")
    return pd.DataFrame(all_keypoints)


# Iterate over all .mov video files in the directory and process them
video_folder = 'video'  # Replace with the actual path to the video folder
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mov')]

all_df = pd.DataFrame()

for video_file in video_files:
    df_keypoints = process_video(video_file)
    all_df = pd.concat([all_df, df_keypoints], ignore_index=True)

print("All videos processed. Saving to CSV...")
all_df.to_csv('keypoints.csv', index=False)  # Save all video keypoints data to a CSV file
print("Data saved to keypoints.csv.")
