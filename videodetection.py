import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Constants
confidence_threshold = 0.5
input_size = 192
required_features = 340

def draw_keypoints_and_edges(frame, keypoints, fall_detected, confidence_threshold=0.5):
    edges = [
        (0, 1), (1, 3), (0, 2), (2, 4),  # Arms
        (0, 5), (5, 7), (7, 9),          # Left side
        (0, 6), (6, 8), (8, 10),         # Right side
        (5, 6), (11, 5), (12, 6),        # Shoulders
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    h, w, _ = frame.shape

    # Draw keypoints
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > confidence_threshold:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Draw edges
    for edge in edges:
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display fall detection alert
    if fall_detected:
        cv2.putText(frame, "FALL DETECTED!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def predict_fall(trained_model, keypoints):
    # Extract coordinates from keypoints
    sample = [coord for point in keypoints for coord in point[:2]]
    
    # Ensure correct input size
    if len(sample) < required_features:
        sample = np.pad(sample, (0, required_features - len(sample)), constant_values=0)
    elif len(sample) > required_features:
        sample = sample[:required_features]

    # Reshape and predict
    sample = np.array(sample).reshape(1, -1)
    prediction = trained_model.predict(sample)[0][0]
    
    return prediction > 0.5

def process_video(video_path, output_path=None):
    # Load models
    movenet_model = load_movenet_model()
    trained_model = load_trained_model()

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        input_image = preprocess_image(frame, input_size)
        input_image_tensor = tf.convert_to_tensor(input_image, dtype=tf.int32)
        outputs = movenet_model.signatures["serving_default"](input_image_tensor)
        keypoints = outputs["output_0"].numpy().reshape(-1, 3)

        # Detect fall
        fall_detected = predict_fall(trained_model, keypoints)

        # Draw visualization
        draw_keypoints_and_edges(frame, keypoints, fall_detected)

        # Save frame if output path is provided
        if output_path:
            out.write(frame)

        # Display frame
        cv2.imshow("Video Fall Detection", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def load_movenet_model():
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    print("Loading MoveNet model...")
    model = hub.load(model_url)
    print("MoveNet model loaded successfully!")
    return model

def load_trained_model():
    print("Loading trained model...")
    model = tf.keras.models.load_model("lstm_v1.h5")
    print("Trained model loaded successfully!")
    return model

def preprocess_image(frame, input_size=192):
    img = cv2.resize(frame, (input_size, input_size))
    img = img.astype(np.int32)
    img = np.expand_dims(img, axis=0)
    return img

if __name__ == "__main__":
    video_path = "falls/fall_1.mp4"
    output_path = "output_video.mp4"
    process_video(video_path, output_path)