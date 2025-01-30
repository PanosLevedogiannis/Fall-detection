import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv

# Load the MoveNet model
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
        if curr[2] > 0.5 and prev[2] > 0.5:  # Confidence threshold for both keypoints
            velocity = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2) / time_interval
            velocities.append(velocity)
        else:
            velocities.append(0.0)  # No reliable velocity if confidence is low
    return velocities

# Process video and extract keypoints with velocities
def process_video(video_path, label, output_csv, input_size=192):
    model = load_movenet_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    previous_keypoints = None
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if not available
    time_interval = 1 / fps  # Time interval between frames

    print(f"Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
    # Define video paths and labels
    videos = [
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-01-cam0.mp4", "label": 1},  # 1 for FALL
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-02-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-03-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-04-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-05-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-06-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-07-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-08-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-09-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-10-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-11-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-12-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-13-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-14-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-15-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-16-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-17-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-18-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-19-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-20-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-21-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-22-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-23-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-24-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-25-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-26-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-27-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-28-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-29-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_falling_files/fall-30-cam0.mp4", "label": 1},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-01-cam0.mp4", "label": 0},  # 0 for NO FALL
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-02-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-03-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-04-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-05-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-06-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-07-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-08-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-09-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-10-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-11-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-12-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-13-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-14-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-15-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-16-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-17-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-18-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-19-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-20-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-21-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-22-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-23-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-24-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-25-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-26-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-27-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-28-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-29-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-30-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-31-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-32-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-33-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-34-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-35-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-36-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-37-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-38-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-39-cam0.mp4", "label": 0},
        {"path": "/Users/panoslevedogiannis/Documents/fall-detection-code/daidalos_ADL_files/adl-40-cam0.mp4", "label": 0},
        
    ]
    output_csv = "daidalos.csv"

    for video in videos:
        process_video(video["path"], video["label"], output_csv)
