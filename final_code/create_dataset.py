import os
import pickle
from tqdm import tqdm
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np  # Added for normalization calculations

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Increased detection confidence for better quality data
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = os.path.join('dataset', 'asl_alphabet_train', 'asl_alphabet_train')

# Function to normalize landmarks to be scale, translation and rotation invariant
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

# Function to calculate angle-based features
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

data = []
labels = []
discard_count = 0
total_count = 0

# Get directories with valid class labels
print(f"Looking for directories in {os.path.abspath(DATA_DIR)}")
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Directory '{DATA_DIR}' does not exist!")
    print(f"Current working directory: {os.getcwd()}")
    print("Available directories:")
    for d in os.listdir('.'):
        if os.path.isdir(d):
            print(f"- {d}")
    exit(1)

valid_directories = [dir_ for dir_ in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, dir_))]

if not valid_directories:
    print(f"ERROR: No valid subdirectories found in {DATA_DIR}!")
    print("Contents of directory:")
    for item in os.listdir(DATA_DIR):
        print(f"- {item} ({'directory' if os.path.isdir(os.path.join(DATA_DIR, item)) else 'file'})")
    exit(1)

print(f"Found {len(valid_directories)} valid class directories: {valid_directories}")

for dir_ in tqdm(valid_directories, desc='Processing classes'):
    class_dir = os.path.join(DATA_DIR, dir_)
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Warning: No image files found in directory {dir_}")
        continue
        
    print(f"Processing {len(image_files)} images for class '{dir_}'")
    
    for img_path in tqdm(image_files, desc=f'Processing {dir_}', leave=False):
        total_count += 1
        img = cv2.imread(os.path.join(class_dir, img_path))
        
        if img is None:
            print(f"Warning: Could not read {img_path} in {dir_}")
            discard_count += 1
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply histogram equalization to improve contrast
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        # Try detection with both regular and enhanced images
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            results = hands.process(img_enhanced)
            
        if not results.multi_hand_landmarks:
            discard_count += 1
            continue
            
        for hand_landmarks in results.multi_hand_landmarks:
            # Get normalized landmarks (scale and translation invariant)
            normalized_data = normalize_landmarks(hand_landmarks)
            
            # Add angle-based features
            angle_features = calculate_angle_features(hand_landmarks)
            
            # Combine all features
            all_features = normalized_data + angle_features
            
            data.append(all_features)
            labels.append(dir_)
            
            # For debugging - visualize landmarks on the first few images
            # if total_count < 5:
            #     annotated_img = img_rgb.copy()
            #     mp_drawing.draw_landmarks(
            #         annotated_img,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style()
            #     )
            #     plt.figure(figsize=(8, 8))
            #     plt.imshow(annotated_img)
            #     plt.title(f"Class: {dir_}, File: {img_path}")
            #     plt.savefig(f"debug_{dir_}_{img_path}.png")
            #     plt.close()

# Avoid division by zero
if total_count > 0:
    discard_percentage = discard_count / total_count * 100
else:
    discard_percentage = 0

print(f"Processed {total_count} images, discarded {discard_count} ({discard_percentage:.2f}%) with no hand landmarks")

# Only save if we have data
if len(data) > 0:
    print(f"Final dataset size: {len(data)} samples from {len(set(labels))} classes")
    f = open('data_1.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
    print("Dataset saved to data_1.pickle")
else:
    print("No data was collected! Check your dataset structure and paths.")