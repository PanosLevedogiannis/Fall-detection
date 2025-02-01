import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv
import os

# Load MoveNet Model (Load Once to Improve Performance)
def load_movenet_model():
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    print("Loading MoveNet model...")
    model = tf.saved_model.load(hub.resolve(model_url))
    print("MoveNet model loaded successfully!")
    return model

# Preprocess the image for MoveNet
def preprocess_image(frame, input_size=192):
    img = cv2.resize(frame, (input_size, input_size))
    img = img.astype(np.int32)
    img = np.expand_dims(img, axis=0)
    return img

# Save keypoints and velocities to CSV
def save_to_csv(filename, keypoints, velocities, label):
    row = [coord for point in keypoints for coord in point[:2]]  # Flatten [x, y]
    row.extend(velocities)  # Add velocity features
    row.append(label)  # Append the label (1 for FALL, 0 for NO FALL)
    
    if len(row) == 52:  # Ensure 17 keypoints (x, y) + 17 velocities + 1 label
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    else:
        print(f"Skipping frame with incomplete data: {len(row)} fields")

# Calculate velocity for keypoints
def calculate_velocity(current_keypoints, previous_keypoints, time_interval):
    velocities = []
    for curr, prev in zip(current_keypoints, previous_keypoints):
        if curr[2] > 0.5 and prev[2] > 0.5:  # Confidence threshold for keypoints
            velocity = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2) / time_interval
            velocities.append(velocity)
        else:
            velocities.append(0.0)  # Default velocity if confidence is low
    return velocities

# Process a video and extract keypoints with velocities
def process_video(video_path, label, output_csv, model, input_size=192):
    # Check if the video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found. Skipping...")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'. Skipping...")
        return

    previous_keypoints = None
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if not available
    time_interval = 1 / fps  # Time interval between frames

    print(f"Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Preprocess the frame
        input_image = preprocess_image(frame, input_size)
        input_image_tensor = tf.convert_to_tensor(input_image, dtype=tf.int32)

        # Get keypoints
        outputs = model.signatures["serving_default"](input_image_tensor)
        current_keypoints = outputs["output_0"].numpy().reshape(-1, 3)  # Shape: [17, 3]

        # Calculate velocities
        if previous_keypoints is not None:
            velocities = calculate_velocity(current_keypoints, previous_keypoints, time_interval)
        else:
            velocities = [0.0] * 17  # Placeholder velocities for the first frame

        # Save keypoints and velocities to CSV
        save_to_csv(output_csv, current_keypoints, velocities, label)

        previous_keypoints = current_keypoints
        frame_count += 1

    cap.release()
    print(f"Finished processing {video_path}. Frames processed: {frame_count}")

# Main function
if __name__ == "__main__":
    # Define paths for 5 FALL and 5 NO FALL videos
    videos = [
        {"path": "falls/fall_1.mp4", "label": 1},  # FALL
        {"path": "falls/fall_2.mp4", "label": 1},
        {"path": "falls/fall_3.mp4", "label": 1},
        {"path": "falls/fall_4.mp4", "label": 1},
        {"path": "falls/fall_5.mp4", "label": 1},

        {"path": "no_falls/no_fall_1.mp4", "label": 0},  # NO FALL
        {"path": "no_falls/no_fall_2.mp4", "label": 0},
        {"path": "no_falls/no_fall_3.mp4", "label": 0},
        {"path": "no_falls/no_fall_4.mp4", "label": 0},
        {"path": "no_falls/no_fall_5.mp4", "label": 0},
    ]

    output_csv = "lstm1_v1.csv"  # Output CSV file

    # Load MoveNet once to avoid reloading it for each video
    movenet_model = load_movenet_model()

    # Process each video
    for video in videos:
        process_video(video["path"], video["label"], output_csv, movenet_model)
