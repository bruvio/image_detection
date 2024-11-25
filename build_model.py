import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import logging
import random

# Set the random seeds for reproducibility
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Initialize logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# Load and preprocess images
def load_images_from_folder(folder):
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder)
    LOGGER.info(f"Folder path is {folder_path}")
    images = []
    labels = []

    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist.")

    for label in os.listdir(folder_path):
        LOGGER.info(f"Processing label: {label}")
        if label == "images_test":
            continue
        label_folder = os.path.join(folder_path, label)
        LOGGER.info(f"Label '{label}' folder is {label_folder}")
        if not os.path.isdir(label_folder):
            continue
        # Walk through the label folder and its subfolders
        for root, dirs, files in os.walk(label_folder):
            for file in files:
                img_path = os.path.join(root, file)
                # LOGGER.info(f"Processing image {img_path}")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                if img is not None:
                    img = cv2.resize(img, (80, 140))  # Resize to 80x140 (width x height) for the model
                    images.append(img)
                    labels.append(label)
                else:
                    LOGGER.warning(f"Failed to load image {img_path}")
        # Removed the 'break' statement to allow processing all labels

    return np.array(images), np.array(labels)


# Load data
LOGGER.info("Loading images from dataset")
image_data, labels = load_images_from_folder("dataset/")
image_data = image_data.reshape(-1, 140, 80, 1)  # Add channel dimension for grayscale
image_data = image_data / 255.0  # Normalize the pixel values

# Convert labels to integers (e.g., ticked=0, unticked=1, circled_yes=2, circled_no=3)
LOGGER.info("Labeling images")
label_mapping = {"ticked": 0, "unticked": 1, "circled_yes": 2, "circled_no": 3}
labels = np.array([label_mapping[label] for label in labels])

# Count the number of images per label
unique_labels, counts = np.unique(labels, return_counts=True)
label_names = {v: k for k, v in label_mapping.items()}  # Reverse mapping
LOGGER.info("\nNumber of images per label in the entire dataset:")
for label_int, count in zip(unique_labels, counts):
    label_name = label_names[label_int]
    LOGGER.info(f"{label_name} ({label_int}): {count} images")

# Split data into training, validation, and testing
LOGGER.info("Splitting data into training, validation, and testing")
X_train_full, X_test, y_train_full, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)  # 10% of training data for validation


# Function to count labels
def count_labels(y, dataset_name):
    unique_labels, counts = np.unique(y, return_counts=True)
    LOGGER.info(f"\nNumber of images per label in the {dataset_name} set:")
    for label_int, count in zip(unique_labels, counts):
        label_name = label_names[label_int]
        LOGGER.info(f"{label_name} ({label_int}): {count} images")


# Count labels in each dataset
count_labels(y_train, "training")
count_labels(y_val, "validation")
count_labels(y_test, "testing")

# Compute class weights
class_weights_array = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))


# Set default values
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.0005

# Read environment variables with defaults
epochs = int(os.getenv("EPOCHS", DEFAULT_EPOCHS))
learning_rate = float(os.getenv("LEARNING_RATE", DEFAULT_LEARNING_RATE))

LOGGER.info(f"Using hyperparameters: epochs={epochs}, learning_rate={learning_rate}")

# Create the CNN model
LOGGER.info("Creating CNN model based on the original architecture")
model = models.Sequential()

# First convolutional block
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(140, 80, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Second convolutional block
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Third convolutional block
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))  # Added pooling layer
model.add(layers.Dropout(0.25))

# Flatten and dense layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))  # Increased neurons
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation="softmax"))

# Compile the model with dynamic learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train the model without data augmentation
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,  # Use dynamic number of epochs
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weights,
)


# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
LOGGER.info(f"\nTest accuracy\n: {test_acc}")


# Evaluate the model
LOGGER.info("\nEvaluating the model")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
LOGGER.info(f"\nClassification Report\n {classification_report(y_test, y_pred, target_names=label_mapping.keys())}")


# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
LOGGER.info(f"Confusion Matrix:\n {conf_mat}")


model.save("image_classifier.keras")
