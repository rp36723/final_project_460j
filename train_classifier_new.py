import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

print("Loading data...")
data_dict = pickle.load(open('data_depth.pickle', 'rb'))

# We know from inspection that most samples have length 42
TARGET_LENGTH = 42
print(f"Standardizing all samples to length {TARGET_LENGTH}...")

data_list = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Process data to ensure consistent length
processed_data = []
for item in tqdm(data_list, desc="Processing samples"):
    if len(item) > TARGET_LENGTH:
        # Truncate longer samples (those with length 84)
        processed_data.append(item[:TARGET_LENGTH])
    elif len(item) < TARGET_LENGTH:
        # Pad shorter samples (shouldn't happen based on our inspection)
        processed_data.append(item + [0] * (TARGET_LENGTH - len(item)))
    else:
        # Already the right length
        processed_data.append(item)
            
# Convert to numpy array
data = np.asarray(processed_data)
print(f"Processed data shape: {data.shape}")

# Split the data
print("Splitting data into training and test sets...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
print("Training Random Forest classifier...")
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly'.format(score * 100))

f = open('model1.pickle', 'wb')
pickle.dump({'model': model}, f)
f.close()
print("Model saved to model1.pickle")