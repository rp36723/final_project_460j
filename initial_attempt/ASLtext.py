import os, numpy as np
from collections import deque
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# load class names (from saved .npy) or derive from train/ subfolders
base_dir = os.path.dirname(__file__)
cn_path = os.path.join(base_dir, 'class_names.npy')
if os.path.exists(cn_path):
    class_names = np.load(cn_path).tolist()
else:
    train_dir = os.path.join(base_dir, 'train')
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# load the Keras model
model_path = os.path.join(base_dir, 'asl_cnn_best_tf.h5')
model = load_model(model_path)

# preprocess function to convert frame into model input
def preprocess_frame(frame, img_size=(64, 64)):
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # enhance local contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    # apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    # adaptive thresholding for illumination invariance
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    # morphological opening and closing to clean up small noise
    # resize to model input size
    binary_resized = cv2.resize(binary, img_size)
    # save mask for display
    mask = binary_resized.copy()
    # convert single channel back to 3-channel RGB
    img = cv2.cvtColor(binary_resized, cv2.COLOR_GRAY2RGB)
    # expand dimensions and cast to float32
    arr = np.expand_dims(img, axis=0).astype(np.float32)
    # apply same preprocessing as during training
    proc = tf.keras.applications.vgg16.preprocess_input(arr)
    return proc, mask

#below captures frames from camera, should be the main loop function
def capture_frames():
    # Open the default camera (0). You can change the index if you have multiple cameras.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    print("Press 'control + c' to exit.")
    
    # buffer to smooth out predictions over last N frames
    label_buffer = deque(maxlen=10)

    while True:
        #check if frame exists and save it in variable
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        #frame variable contains captured frame
        #below simply displays frame back to user
        cv2.imshow("Live Camera Feed", frame)
        #TODO: call function to detect ASL signs + display wherever else needed

        # preprocess and predict (also get binary mask for display)
        input_tensor, mask = preprocess_frame(frame)
        # show the binary mask that the model sees
        cv2.imshow("Preprocessed Mask", mask)
        preds = model.predict(input_tensor, verbose=0)
        # print top-3 class probabilities for debugging
        top_idxs = np.argsort(preds[0])[::-1][:3]
        top_probs = preds[0][top_idxs]
        print("Top-3 preds:", [(class_names[i], float(top_probs[j])) for j, i in enumerate(top_idxs)])
        pred_idx = np.argmax(preds, axis=1)[0]
        label = class_names[pred_idx]
        # add to buffer and pick the most common label for stability
        label_buffer.append(label)
        stable_label = max(set(label_buffer), key=label_buffer.count)
        # overlay predicted label
        cv2.putText(frame, f"Predicted: {stable_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # display frame with overlay
        cv2.imshow("Live Camera Feed", frame)
        
        #time interval for capturing frames, right now is 1 frame per second
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #release frame and delete active windows to end
    #TODO: have the release and destroy be based on image recognition
    #ie: stop/disconnect is signed therefore camera closes
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_frames()
