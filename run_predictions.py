import os
import cv2
import logging
from collections import defaultdict
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from main import build_consent, annotate_image
import numpy as np
import sys
import json

# Initialize logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# **New Section: Read Environment Variables for Configurable Parameters**
# Set default values
DEFAULT_THRESHOLD = 0.8

# Read environment variables with defaults
threshold = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))
LOGGER.info(f"Using threshold: {threshold}")

# Define paths and settings
folder_list = ["dataset/images_test"]
widths = []
heights = []
aspect_ratios = []
USE_MODEL = os.getenv("USE_MODEL", "simple")
# **Load the Model**
# model_path = "image_classifier.keras"
model_path = f"image_classifier_final_{USE_MODEL}_model.keras"
if not os.path.exists(model_path):
    LOGGER.error(f"Model file {model_path} does not exist.")
    sys.exit(1)

try:
    model = load_model(model_path)
    LOGGER.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    LOGGER.exception(f"Failed to load model from {model_path}: {e}")
    sys.exit(1)

# **Log Model Summary**
LOGGER.info("Model Summary:")
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
LOGGER.info("\n".join(model_summary))

# **Optional: Plot and Save Model Architecture Diagram**
# Uncomment the following lines if you want to save a visual representation of the model
# try:
#     plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
#     LOGGER.info("Model architecture diagram saved as model_architecture.png")
# except Exception as e:
#     LOGGER.warning(f"Failed to plot model architecture: {e}")

# **Evaluate Model Accuracy on Test Set Before Running Predictions**
# Assuming 'dataset/images_test' contains labeled images structured in subfolders by label


# Function to load and preprocess images for evaluation
def load_evaluation_data(folder):
    images = []
    labels = []
    label_mapping = {"ticked": 0, "unticked": 1, "circled_yes": 2, "circled_no": 3}
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue
        if label not in label_mapping:
            LOGGER.warning(f"Unknown label '{label}' found in {folder}, skipping.")
            continue
        for file in os.listdir(label_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                img_path = os.path.join(label_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (80, 140))
                    images.append(img)
                    labels.append(label_mapping[label])
                else:
                    LOGGER.warning(f"Failed to load image {img_path}")
    if not images:
        LOGGER.error("No images found for evaluation.")
        sys.exit(1)
    images = np.array(images).reshape(-1, 140, 80, 1) / 255.0  # Normalize
    labels = np.array(labels)
    return images, labels, reverse_label_mapping


# Load evaluation data
LOGGER.info("Loading evaluation data for computing overall accuracy.")
eval_images, eval_labels, reverse_label_mapping = load_evaluation_data("dataset/images_test")

# **Evaluate the Model**
LOGGER.info("Evaluating the model on the evaluation dataset.")
try:
    eval_loss, eval_accuracy = model.evaluate(eval_images, eval_labels, verbose=0)
    LOGGER.info(f"Evaluation Accuracy: {eval_accuracy * 100:.2f}%")
except Exception as e:
    LOGGER.exception(f"Failed to evaluate the model: {e}")
    sys.exit(1)

# **Initialize metrics storage for detailed predictions**
results = defaultdict(lambda: {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0})
all_predictions = []
all_true_labels = []
false_positives = []
false_negatives = []


# Function to list files in a folder
def list_files_in_folder(folder, exts=("PNG", "JPG", "JPEG", "BMP", "GIF")):
    return [
        file
        for file in os.listdir(folder)
        if file.upper().endswith(exts) and os.path.isfile(os.path.join(folder, file))
    ]


# Process each folder for detailed predictions
for root_folder in folder_list:
    for label_folder in os.listdir(root_folder):
        label_path = os.path.join(root_folder, label_folder)
        if not os.path.isdir(label_path):
            continue

        # Real label is the folder name
        real_label = label_folder
        if real_label not in reverse_label_mapping.values():
            LOGGER.warning(f"Unknown label '{real_label}' found, skipping.")
            continue

        LOGGER.info(f"Processing folder: {label_path} with label: {real_label}")
        image_paths = list_files_in_folder(label_path)

        for file_name in image_paths:
            file_path = os.path.join(label_path, file_name)
            LOGGER.info("\n\nProcessing image: %s", file_path)

            try:
                # Load and process the image
                image = cv2.imread(file_path)
                if image is None:
                    LOGGER.error(f"Failed to read image: {file_path}")
                    continue

                h, w = image.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)

                # Preprocess image for prediction
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_resized = cv2.resize(image_gray, (80, 140)).reshape(1, 140, 80, 1) / 255.0

                # Predict using the model
                y_pred_probs = model.predict(image_resized)
                result_payload = {k: float(v) for k, v in zip(reverse_label_mapping.values(), y_pred_probs[0])}
                LOGGER.info(f"Raw predictions: {result_payload}")

                # Determine the label with the highest confidence
                predicted_label = max(result_payload, key=result_payload.get)
                confidence = result_payload[predicted_label]

                LOGGER.info(f"Processed response: {result_payload}")

                # Store predictions and labels for the report
                all_predictions.append(predicted_label)
                all_true_labels.append(real_label)

                # Calculate metrics
                for label, prob in result_payload.items():
                    if label == real_label:
                        if label == predicted_label and prob >= threshold:
                            results[label]["true_positive"] += 1
                        else:
                            results[label]["false_negative"] += 1
                            false_negatives.append(
                                {
                                    "file": file_path,
                                    "predicted_label": predicted_label,
                                    "confidence": confidence,
                                    "real_label": real_label,
                                }
                            )
                    else:
                        if label == predicted_label and prob >= threshold:
                            results[label]["false_positive"] += 1
                            false_positives.append(
                                {
                                    "file": file_path,
                                    "predicted_label": predicted_label,
                                    "confidence": confidence,
                                    "real_label": real_label,
                                }
                            )
                        else:
                            results[label]["true_negative"] += 1

                # Log the prediction result
                if predicted_label == real_label and confidence >= threshold:
                    LOGGER.info(f"Correct Prediction: {predicted_label} (Confidence: {confidence:.2f})")
                else:
                    LOGGER.warning(
                        f"Incorrect Prediction: {predicted_label} (Confidence: {confidence:.2f}), Real Label: {real_label}"
                    )

                # Build consent dictionary
                consent = build_consent(result_payload, threshold)

                # Annotate the image
                output_directory = "results_model"
                os.makedirs(output_directory, exist_ok=True)
                file_name_prefix = os.path.splitext(os.path.basename(file_path))[0]
                annotated_image_path = annotate_image(
                    image=image, consent=consent, output_path=output_directory, file_name_prefix=file_name_prefix
                )
                LOGGER.info(f"Annotated image saved at: {annotated_image_path}")

            except Exception as e:
                LOGGER.exception(f"An unexpected error occurred while processing image {file_path}: {e}")

# **Generate Report after All Predictions**
LOGGER.info("\n=== Classification Report ===")
report = classification_report(all_true_labels, all_predictions, target_names=list(reverse_label_mapping.values()))
LOGGER.info("\n%s", report)

LOGGER.info("\n=== Confusion Matrix ===")
conf_matrix = confusion_matrix(all_true_labels, all_predictions, labels=list(reverse_label_mapping.values()))
LOGGER.info("\n%s", conf_matrix)

# **Compute and Log Overall Accuracy**
overall_accuracy = accuracy_score(all_true_labels, all_predictions)
LOGGER.info(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

# **Output Detailed Metrics for Each Label**
LOGGER.info("\n=== Detailed Metrics ===")
for label, metrics in results.items():
    LOGGER.info(f"Label: {label}")
    LOGGER.info(f"  True Positives: {metrics['true_positive']}")
    LOGGER.info(f"  False Positives: {metrics['false_positive']}")
    LOGGER.info(f"  True Negatives: {metrics['true_negative']}")
    LOGGER.info(f"  False Negatives: {metrics['false_negative']}")

# **Log False Positives**
LOGGER.info("\n=== False Positives ===")
for fp in false_positives:
    LOGGER.info(
        f"File: {fp['file']} - Predicted: {fp['predicted_label']} (Confidence: {fp['confidence']:.2f}), Real: {fp['real_label']}"
    )

# **Log False Negatives**
LOGGER.info("\n=== False Negatives ===")
for fn in false_negatives:
    LOGGER.info(
        f"File: {fn['file']} - Predicted: {fn['predicted_label']} (Confidence: {fn['confidence']:.2f}), Real: {fn['real_label']}"
    )

# **Optional: Save Detailed Metrics to a JSON File**
report_data = {
    "classification_report": report,
    "confusion_matrix": conf_matrix.tolist(),
    "overall_accuracy": overall_accuracy,
    "detailed_metrics": {label: metrics for label, metrics in results.items()},
    "false_positives": false_positives,
    "false_negatives": false_negatives,
}

with open("detailed_metrics.json", "w") as f:
    json.dump(report_data, f, indent=4)
LOGGER.info("Detailed metrics saved to detailed_metrics.json")
