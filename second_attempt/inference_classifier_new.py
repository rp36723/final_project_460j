import cv2
import mediapipe as mp
import pickle
import numpy as np
import time  # For FPS calculation

cap = cv2.VideoCapture(0)
# Performance optimizations for video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Request higher framerate

model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Optimize MediaPipe hands for real-time performance
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for video (better tracking)
    max_num_hands=1,          # Only detect one hand to save processing
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
    model_complexity=0        # Use simplest model (0=fastest, 1=balanced)
)

# Map both indices and string labels to letter outputs
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
               11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
               21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
               # Also map string keys in case model returns strings
               'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I',
               'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R',
               'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}

# Variables for FPS calculation
prev_frame_time = 0
curr_frame_time = 0
fps = 0

# Variable to control prediction frequency
prediction_skip_frames = 2  # Only predict every 3rd frame
frame_count = 0
last_prediction = None

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to get frame from camera. Exiting...")
        break
    
    # Calculate FPS
    curr_frame_time = time.time()
    fps = 1/(curr_frame_time - prev_frame_time) if (curr_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = curr_frame_time
    
    # Convert BGR to RGB and flip horizontally for more natural interaction
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process image with MediaPipe - this is the expensive part
    results = hands.process(frame_rgb)
    
    # Only do prediction on every few frames to improve responsiveness 
    frame_count = (frame_count + 1) % prediction_skip_frames
    
    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
    
    # If no hand detected, show message and continue
    if not results.multi_hand_landmarks:
        cv2.putText(frame, "No hand detected", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Display last prediction if available
        if last_prediction:
            cv2.putText(frame, f"Last: {last_prediction}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
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
    
    # Only predict every few frames to improve responsiveness
    if frame_count == 0:
        # Extract landmark data
        data_aux = []
        hand_landmarks = results.multi_hand_landmarks[0]
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)
        
        # Ensure we have the expected amount of data (42 values)
        if len(data_aux) > 42:
            data_aux = data_aux[:42]
        elif len(data_aux) < 42:
            data_aux.extend([0] * (42 - len(data_aux)))
        
        # Make prediction
        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_value = prediction[0]
            
            if predicted_value in labels_dict:
                predicted_character = labels_dict[predicted_value]
                last_prediction = predicted_character
                cv2.putText(frame, f"Prediction: {predicted_character}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print(f"Unknown prediction value: {predicted_value}")
                cv2.putText(frame, f"Unknown: {predicted_value}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        except Exception as e:
            print(f"Error during prediction: {e}")
            cv2.putText(frame, f"Error: {str(e)[:20]}...", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Display previous prediction if available
        if last_prediction:
            cv2.putText(frame, f"Prediction: {last_prediction}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame - use a faster wait time (1ms)
    cv2.imshow('ASL Recognition', frame)
    
    # Check for key press with minimal delay 
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
