import os
import random
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.regularizers import l2

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

# Reverse mapping for label names
label_names = {v: k for k, v in label_mapping.items()}  # Reverse mapping

# Count the number of images per label
unique_labels, counts = np.unique(labels, return_counts=True)
LOGGER.info("\nNumber of images per label in the entire dataset:")
for label_int, count in zip(unique_labels, counts):
    label_name = label_names[label_int]
    LOGGER.info(f"{label_name} ({label_int}): {count} images")

# Set default values
DEFAULT_EPOCHS = 500
DEFAULT_LEARNING_RATE = 0.0005

# Read environment variables with defaults
epochs = int(os.getenv("EPOCHS", DEFAULT_EPOCHS))
learning_rate = float(os.getenv("LEARNING_RATE", DEFAULT_LEARNING_RATE))

LOGGER.info(f"Using hyperparameters: epochs={epochs}, learning_rate={learning_rate}")


# Function to build the CNN model
def build_model():
    model = models.Sequential()

    # First convolutional block
    model.add(layers.Conv2D(64, (3, 3), padding="same", input_shape=(140, 80, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Second convolutional block
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    # Third convolutional block
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation="softmax"))

    return model


# Stratified K-Fold Cross-Validation
k = 5  # Number of folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train_index, val_index in skf.split(image_data, labels):
    LOGGER.info(f"\nStarting fold {fold_no}...")

    # Split data
    X_train_fold, X_val_fold = image_data[train_index], image_data[val_index]
    y_train_fold, y_val_fold = labels[train_index], labels[val_index]

    # Compute class weights for the current fold
    class_weights_fold = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_fold), y=y_train_fold
    )
    class_weights_fold = dict(enumerate(class_weights_fold))

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(X_train_fold)

    # Build and compile the model for each fold
    model = build_model()
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7)
    checkpoint = ModelCheckpoint(f"best_model_fold_{fold_no}.h5", monitor="val_loss", save_best_only=True)
    callbacks = [early_stopping, reduce_lr, checkpoint]

    # Train the model
    history = model.fit(
        datagen.flow(X_train_fold, y_train_fold, batch_size=32),
        epochs=epochs,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=callbacks,
        class_weight=class_weights_fold,
        verbose=1,
    )

    # Evaluate the model
    scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    LOGGER.info(
        f"Score for fold {fold_no}: {model.metrics_names[0]} = {scores[0]:.4f}; {model.metrics_names[1]} = {scores[1]:.4f}"
    )
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

    # Plot training metrics for each fold
    plt.figure(figsize=(12, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy", color="blue")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.title(f"Model Accuracy - Fold {fold_no}")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss", color="blue")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title(f"Model Loss - Fold {fold_no}")

    plt.tight_layout()
    plt.savefig(f"training_metrics_fold_{fold_no}.png")
    plt.close()

    # Confusion Matrix and Classification Report
    y_pred_probs = model.predict(X_val_fold)
    y_pred = np.argmax(y_pred_probs, axis=1)
    LOGGER.info(
        f"\nClassification Report for fold {fold_no}:\n{classification_report(y_val_fold, y_pred, target_names=label_mapping.keys())}"
    )

    conf_mat = confusion_matrix(y_val_fold, y_pred)
    LOGGER.info(f"Confusion Matrix for fold {fold_no}:\n {conf_mat}")

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=label_names.values(), yticklabels=label_names.values()
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Fold {fold_no}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_fold_{fold_no}.png")
    plt.close()

    fold_no += 1

# Displaying the results
LOGGER.info("------------------------------------------------------------------------")
LOGGER.info("Score per fold")
for i in range(0, len(acc_per_fold)):
    LOGGER.info(f"> Fold {i+1} - Loss: {loss_per_fold[i]:.4f} - Accuracy: {acc_per_fold[i]*100:.2f}%")
LOGGER.info("------------------------------------------------------------------------")
LOGGER.info("Average scores for all folds:")
LOGGER.info(f"> Accuracy: {np.mean(acc_per_fold)*100:.2f}% (+- {np.std(acc_per_fold)*100:.2f}%)")
LOGGER.info(f"> Loss: {np.mean(loss_per_fold):.4f}")
LOGGER.info("------------------------------------------------------------------------")

# Saving the final model (Optional: You might want to save the model with the best validation accuracy)
model.save("image_classifier_final.keras")
