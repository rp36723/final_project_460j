import os
import pickle
from tqdm import tqdm
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = os.path.join('dataset', 'asl_alphabet_train', 'asl_alphabet_train')


data = []
labels = []

for dir_ in tqdm(os.listdir(DATA_DIR), desc='Directories'):
  for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
    data_aux = []
    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    resuts = hands.process(img_rgb)
    if not resuts.multi_hand_landmarks:
      continue
    for hand_landmarks in resuts.multi_hand_landmarks:
      for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x)
        data_aux.append(y)
      

      data.append(data_aux)
      labels.append(dir_)
    # plt.imshow(img_rgb)
    # plt.show()
    
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()