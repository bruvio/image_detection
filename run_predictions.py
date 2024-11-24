import os
import cv2
import logging
from io import BytesIO
from collections import defaultdict
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from main import *

# Initialize logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define paths and settings
folder_list = ["dataset/images_test"]
widths = []
heights = []
aspect_ratios = []

model = load_model("image_classifier.keras")
threshold = 0.8

# Function to list files in a folder
def list_files_in_folder(folder, ext="PNG"):
    return [
        file for file in os.listdir(folder)
        if file.endswith(ext) and os.path.isfile(os.path.join(folder, file))
    ]

# Initialize metrics storage
results = defaultdict(lambda: {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0})
all_predictions = []
all_true_labels = []

# Process each folder
for root_folder in folder_list:
    for label_folder in os.listdir(root_folder):
        label_path = os.path.join(root_folder, label_folder)
        if not os.path.isdir(label_path):
            continue
        
        # Real label is the folder name
        real_label = label_folder
        
        LOGGER.info(f"Processing folder: {label_path} with label: {real_label}")
        image_paths = list_files_in_folder(label_path, ext="PNG")
        
        for file_name in image_paths:
            file_path = os.path.join(label_path, file_name)
            LOGGER.info("\n\n")
            LOGGER.info(f"Processing image: {file_path}")
            
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
                
                # Convert image to bytes
                with open(file_path, "rb") as image_file:
                    image_bytes = BytesIO(image_file.read())
                
                # Predict using the model
                result_payload = process_image_with_model(image_bytes, threshold, model)
                LOGGER.info(f"Raw predictions: {result_payload}")

                # Determine the label with the highest confidence
                predicted_label = max(result_payload, key=result_payload.get)
                confidence = result_payload[predicted_label]

                LOGGER.info(f"Processed response: {result_payload}")
                
                # Store predictions and labels for the report
                all_predictions.append(predicted_label)
                all_true_labels.append(real_label)
                
                # Calculate metrics
                for label in result_payload.keys():
                    if label == real_label:
                        if label == predicted_label:
                            results[label]["true_positive"] += 1
                        else:
                            results[label]["false_negative"] += 1
                    else:
                        if label == predicted_label:
                            results[label]["false_positive"] += 1
                        else:
                            results[label]["true_negative"] += 1
                
                # Log the prediction result
                if predicted_label == real_label:
                    LOGGER.info(f"Correct Prediction: {predicted_label} (Confidence: {confidence:.2f})")
                else:
                    LOGGER.warning(f"Incorrect Prediction: {predicted_label} (Confidence: {confidence:.2f}), Real Label: {real_label}")
                
                # Build consent dictionary
                consent = build_consent(result_payload, threshold)
                
                # Annotate the image
                output_directory = "results_model"
                file_name_prefix = os.path.splitext(os.path.basename(file_path))[0]
                annotated_image_path = annotate_image(
                    image=image, consent=consent, output_path=output_directory, file_name_prefix=file_name_prefix
                )
                LOGGER.info(f"Annotated image saved at: {annotated_image_path}")
            
            except Exception as e:
                LOGGER.exception(f"An unexpected error occurred while processing image {file_path}: {e}")

# Generate report after all predictions
LOGGER.info("\n=== Classification Report ===\n%s", classification_report(all_true_labels, all_predictions, target_names=list(results.keys())))



LOGGER.info("\n=== Confusion Matrix ===\n%s", confusion_matrix(all_true_labels, all_predictions, labels=list(results.keys())))


# Output detailed metrics for each label
LOGGER.info("\n=== Detailed Metrics ===")
for label, metrics in results.items():
    LOGGER.info(f"Label: {label}")
    LOGGER.info(f"  True Positives: {metrics['true_positive']}")
    LOGGER.info(f"  False Positives: {metrics['false_positive']}")
    LOGGER.info(f"  True Negatives: {metrics['true_negative']}")
    LOGGER.info(f"  False Negatives: {metrics['false_negative']}")


