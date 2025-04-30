import cv2
import mediapipe as mp
import pickle
import numpy as np
import time

# Load the model with preprocessing pipeline
print("Loading models for ensemble...")
model_dict1 = pickle.load(open('model_gpt.pickle', 'rb'))
model1 = model_dict1['model']
# Load second model
model_dict2 = pickle.load(open('model.pickle', 'rb'))
model2 = model_dict2['model']

# Ensemble prediction: average probabilities
from sklearn.utils.validation import check_array

def ensemble_predict_proba(features_gpt, features_old):
    # Predict with GPT-trained model (expects 64 features)
    p1 = model1.predict_proba([features_gpt])
    # Predict with original model (expects 42 features)
    p2 = model2.predict_proba([features_old])
    # Average probabilities
    return (p1 + p2) / 2

# Use the ensemble for predictions
# Get metadata
feature_count = model_dict1.get('features', 63)  # Default to 63 (21 landmarks × 3 coordinates)
model_type = model_dict1.get('model_type', 'Unknown')
classes = model_dict1.get('classes', None)
print(f"Loaded {model_type} model with {feature_count} features")

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up webcam with improved settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Optimize MediaPipe hands for real-time performance
hands = mp_hands.Hands(
    static_image_mode=False,  # Better for video
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0  # Use simplest model for speed
)

# Create label dictionary from classes or default to alphabet
if classes is not None:
    # Use class list from model
    labels_dict = {i: label for i, label in enumerate(classes)}
else:
    # Default alphabet dictionary as fallback
    labels_dict = {i: chr(65+i) for i in range(26)}  # A-Z

# Add string keys as values for direct string predictions
for key, value in list(labels_dict.items()):
    labels_dict[value] = value

# Function to normalize landmarks (same as in training)
def normalize_landmarks(landmarks):
    # Extract all coordinates
    xs = [landmark.x for landmark in landmarks.landmark]
    ys = [landmark.y for landmark in landmarks.landmark]
    zs = [landmark.z for landmark in landmarks.landmark]
    
    # Find bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Calculate bounding box size
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    # Calculate bounding box center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Calculate normalization scale (larger dimension)
    scale = max(bbox_width, bbox_height)
    if scale == 0:  # Avoid division by zero
        scale = 1.0
        
    # Normalize landmarks
    normalized_landmarks = []
    for i in range(len(landmarks.landmark)):
        # Center and scale the coordinates
        norm_x = (landmarks.landmark[i].x - center_x) / scale
        norm_y = (landmarks.landmark[i].y - center_y) / scale
        norm_z = landmarks.landmark[i].z / scale  # Just scale Z
        
        # Add to list
        normalized_landmarks.extend([norm_x, norm_y, norm_z])
    
    return normalized_landmarks

# Function to calculate angle-based features (same as in training)
def calculate_angle_features(landmarks):
    angle_features = []
    
    # Get key points (example: using thumb, index and middle finger tips)
    thumb_tip = np.array([landmarks.landmark[4].x, landmarks.landmark[4].y, landmarks.landmark[4].z])
    index_tip = np.array([landmarks.landmark[8].x, landmarks.landmark[8].y, landmarks.landmark[8].z])
    middle_tip = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y, landmarks.landmark[12].z])
    
    # Calculate vectors
    v1 = thumb_tip - index_tip
    v2 = middle_tip - index_tip
    
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
    v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
    
    # Calculate angle (cosine of the angle between vectors)
    cos_angle = np.dot(v1, v2)
    angle_features.append(cos_angle)
    
    # Add more angle features as needed
    # ...
    
    return angle_features

# Variables for tracking predictions
last_predictions = []  # For temporal smoothing
prediction_smooth_window = 5
confidence_threshold = 0.3  # Minimum confidence to display a prediction
last_timestamp = 0
fps = 0

# Variables for fingerspelling/sentence building
current_text = ""
last_letter = None
letter_hold_frames = 0
required_hold_frames = 10  # Number of frames a letter must be detected consistently
cooldown_frames = 0
required_cooldown = 15  # Frames to wait after adding a letter

print("Ready! Press ESC to exit.")

while True:
    # Measure FPS
    current_time = time.time()
    if last_timestamp:
        fps = 1 / (current_time - last_timestamp)
    last_timestamp = current_time
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame from camera. Exiting...")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Process image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Status display area
    status_bg = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)
    
    # Show FPS
    cv2.putText(
        status_bg, f"FPS: {int(fps)}", 
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
    )
    
    # Show current text
    cv2.putText(
        status_bg, f"Text: {current_text}", 
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
    )
    
    # Combined frame and status
    display_frame = np.vstack([frame, status_bg])
    
    # No hand detected
    if not results.multi_hand_landmarks:
        cv2.putText(
            display_frame, "No hand detected", 
            (10, frame.shape[0] + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.imshow('ASL Recognition', display_frame)
        
        # Clear prediction history when no hand is visible
        last_predictions = []
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        continue
    
    # Draw hand landmarks
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    # Extract features using the same methods as training
    hand_landmarks = results.multi_hand_landmarks[0]  # Use first hand
    
    # Get normalized landmarks (64 features)
    normalized_data = normalize_landmarks(hand_landmarks)
    angle_features = calculate_angle_features(hand_landmarks)
    all_features = normalized_data + angle_features
    
    # Ensure the right feature length (pad or truncate if needed)
    if len(all_features) > feature_count:
        all_features = all_features[:feature_count]
    elif len(all_features) < feature_count:
        all_features = all_features + [0] * (feature_count - len(all_features))
    
    # Prepare feature set for original model: raw x,y coords (42 features)
    original_xy = []
    for lm in hand_landmarks.landmark:
        original_xy.extend([lm.x, lm.y])
    # Ensure correct length
    original_xy = original_xy[:42]
    
    # Make prediction using ensemble
    try:
        letter_proba = ensemble_predict_proba(all_features, original_xy)
        best_idx = np.argmax(letter_proba)
        confidence = letter_proba[0][best_idx]
        predicted_label = model1.classes_[best_idx]
        
        # Add to prediction history for smoothing
        last_predictions.append((predicted_label, confidence))
        if len(last_predictions) > prediction_smooth_window:
            last_predictions.pop(0)
        
        # Get the most common prediction in the window
        if len(last_predictions) > 0:
            # Count occurrences of each label
            label_counts = {}
            for lbl, conf in last_predictions:
                if lbl not in label_counts:
                    label_counts[lbl] = {"count": 0, "total_conf": 0}
                label_counts[lbl]["count"] += 1
                label_counts[lbl]["total_conf"] += conf
            
            # Find the most common
            best_label = None
            max_count = 0
            max_conf = 0
            
            for lbl, data in label_counts.items():
                avg_conf = data["total_conf"] / data["count"]
                if data["count"] > max_count or (data["count"] == max_count and avg_conf > max_conf):
                    max_count = data["count"]
                    max_conf = avg_conf
                    best_label = lbl
                    
            predicted_label = best_label
            confidence = max_conf
            
            # Handle letter adding logic for fingerspelling
            if cooldown_frames > 0:
                cooldown_frames -= 1
                
            # Check if this is a consistently detected letter
            if last_letter == predicted_label:
                letter_hold_frames += 1
                
                # If held long enough and confidence is high, add to text
                if letter_hold_frames > required_hold_frames and confidence > confidence_threshold and cooldown_frames == 0:
                    if predicted_label != last_letter or not current_text:
                        current_text += labels_dict.get(predicted_label, predicted_label)
                        cooldown_frames = required_cooldown
            else:
                # Reset the counter if letter changed
                letter_hold_frames = 0
                last_letter = predicted_label
                    
            # Display the prediction with confidence
            if confidence > confidence_threshold:
                letter_text = labels_dict.get(predicted_label, predicted_label)
                cv2.putText(
                    frame, f"{letter_text} ({confidence:.2f})", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                )
                
                # Show hold progress 
                hold_progress = min(letter_hold_frames / required_hold_frames, 1.0)
                if hold_progress > 0:
                    bar_width = int(200 * hold_progress)
                    cv2.rectangle(status_bg, (10, 100), (10 + bar_width, 120), (0, 255, 0), -1)
                    cv2.rectangle(status_bg, (10, 100), (210, 120), (100, 100, 100), 2)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        cv2.putText(
            display_frame, f"Error: {str(e)[:20]}...", 
            (10, frame.shape[0] + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    
    # Update display frame with the newly modified status bar
    display_frame = np.vstack([frame, status_bg])
    
    # Show the frame
    cv2.imshow('ASL Recognition', display_frame)
    
    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Space
        current_text += " "
    elif key == 8:  # Backspace
        current_text = current_text[:-1] if current_text else ""

# Clean up
cap.release()
cv2.destroyAllWindows()
