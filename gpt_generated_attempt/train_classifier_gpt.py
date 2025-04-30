import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

print("Loading data...")
data_dict = pickle.load(open('../dataset/data_gpt.pickle', 'rb'))

# Get the new data format (should be normalized landmarks with angle features)
data_list = data_dict['data']
labels = np.asarray(data_dict['labels'])
print("Data loaded successfully")

# Check data shape to adapt processing
feature_counts = [len(item) for item in data_list]
unique_counts = set(feature_counts)
print(f"Found feature counts: {unique_counts}")

# Process data to ensure consistent feature length
if len(unique_counts) > 1:
    # We need to standardize
    max_length = max(feature_counts)
    print(f"Standardizing all samples to length {max_length}...")
    
    processed_data = []
    for item in tqdm(data_list, desc="Processing samples"):
        if len(item) > max_length:
            processed_data.append(item[:max_length])
        elif len(item) < max_length:
            processed_data.append(item + [0] * (max_length - len(item)))
        else:
            processed_data.append(item)
    data = np.array(processed_data)
else:
    # All samples already have the same length
    print("All samples have consistent feature length.")
    data = np.array(data_list)

print(f"Data shape: {data.shape}")

# Check for class imbalance
unique_labels, label_counts = np.unique(labels, return_counts=True)
print(f"Classes: {len(unique_labels)}")
print(f"Samples per class (min: {min(label_counts)}, max: {max(label_counts)}, avg: {np.mean(label_counts):.1f})")

# Print the classes with fewest samples
min_samples_per_class = {}
for i, label in enumerate(unique_labels):
    count = label_counts[i]
    min_samples_per_class[label] = count
    
print("\nClasses with fewest samples:")
for label, count in sorted(min_samples_per_class.items(), key=lambda x: x[1])[:5]:
    print(f"  {label}: {count} samples")

# Split the data
print("\nSplitting data into training and test sets...")
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels
)

# Check if we can apply SMOTE (need at least k+1 samples per class)
min_samples_class = np.min(np.bincount(np.where(unique_labels[:,None] == y_train[None,:])[0]))
print(f"\nSmallest class in training set has {min_samples_class} samples")

# Apply SMOTE if possible, otherwise skip
X_train_resampled = x_train
y_train_resampled = y_train

if min_samples_class >= 6:  # Default SMOTE needs at least 6 samples per class
    print("Applying SMOTE to handle class imbalance...")
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
        print(f"Original training set: {x_train.shape}, Resampled: {X_train_resampled.shape}")
    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Proceeding with original unbalanced data")
else:
    print(f"SMOTE requires at least 6 samples per class, but smallest class has {min_samples_class}")
    print("Skipping SMOTE and using original unbalanced data")

# Feature preprocessing and dimensionality reduction
print("Setting up preprocessing pipeline...")
# Start with scaling to normalize the features
preprocessing = Pipeline([
    ('scaler', StandardScaler()),
    # Optional: Uncomment to use PCA if you have many features
    # ('pca', PCA(n_components=0.95))  # Keep 95% of variance
])

# Fit preprocessing on training data
X_train_processed = preprocessing.fit_transform(X_train_resampled)
X_test_processed = preprocessing.transform(x_test)

print("Performing hyperparameter tuning with cross-validation...")

# Define models to try - reduced parameter sets for faster training
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100],
            'max_depth': [None, 10],
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100],
            'learning_rate': [0.1],
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [1],
            'gamma': ['scale'],
        }
    }
}

# Modified progress tracking that works reliably
total_models = len(models)
print(f"\nTraining {total_models} different models:")

# Perform cross-validation for each model
best_models = {}
best_score = 0
best_model_name = ""

# Track overall progress
model_progress = tqdm(models.items(), desc="Training models", total=total_models)

for model_name, mp in model_progress:
    start_time = time.time()
    model_progress.set_description(f"Training {model_name}")
    print(f"\n{'-'*50}")
    print(f"Training {model_name}...")
    print(f"{'-'*50}")
    
    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        mp['model'],
        mp['params'],
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,  # Use sklearn's built-in verbosity for progress
        error_score='raise'
    )
    
    try:
        # Train model
        grid_search.fit(X_train_processed, y_train_resampled)
        
        # Store results
        best_models[model_name] = grid_search.best_estimator_
        
        elapsed_time = time.time() - start_time
        print(f"Best {model_name} parameters: {grid_search.best_params_}")
        print(f"Best {model_name} CV score: {grid_search.best_score_:.4f}")
        print(f"Training time: {elapsed_time:.1f} seconds")
        
        # Check if this is the best model so far
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model_name = model_name
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Get the best model
best_model = best_models.get(best_model_name)
if best_model is None:
    print("Error: No model was successfully trained.")
    exit(1)

print(f"\nBest model: {best_model_name} with CV score: {best_score:.4f}")

# Evaluate on test set
print(f"\nEvaluating best model ({best_model_name}) on test set...")
y_pred = best_model.predict(X_test_processed)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the confusion matrix for later analysis
plt.figure(figsize=(16, 14))
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - {best_model_name} Model')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix visualization to 'confusion_matrix.png'")

# Create a final model with preprocessing pipeline + best classifier
final_pipeline = Pipeline([
    ('preprocessor', preprocessing),
    ('classifier', best_model)
])

# Retrain the pipeline on the full training set
print("\nFinalizing model...")
final_pipeline.fit(X_train_resampled, y_train_resampled)

# Save the model with additional metadata
model_data = {
    'model': final_pipeline,
    'features': data.shape[1],
    'classes': list(unique_labels),
    'preprocessing': preprocessing,
    'model_type': best_model_name,
    'accuracy': test_accuracy
}

print("\nSaving the model and metadata...")
with open('model_gpt.pickle', 'wb') as f:
    pickle.dump(model_data, f)
print("Model saved to model_gpt.pickle")