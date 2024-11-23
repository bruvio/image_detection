from main import *
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
from sklearn.metrics import classification_report
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



def list_files_in_folder(folder_path, ext=None):
    """
    Returns a list of file names in the given folder, optionally filtering by extension.

    Parameters:
        folder_path (str): The path to the folder.
        ext (str, optional): The file extension to filter by (e.g., 'png'). Defaults to None.

    Returns:
        list: A list of file names matching the extension.
    """
    try:
        # List all files in the directory
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        if ext is not None:
            # Normalize the extension to lowercase
            ext = ext.lower()
            # Filter files by extension (case-insensitive)
            files = [f for f in files if os.path.splitext(f)[1].lower() == f".{ext}"]

        return files

    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied to access '{folder_path}'.")
        return []


from io import BytesIO

folder_list = [
    "dataset/images_test",
    # "dataset/circled_no",
    # "dataset/circled_yes",
    # "dataset/unticked",
    # "dataset/ticked",
]
widths = []
heights = []
aspect_ratios = []

threshold = 0.6
for folder in folder_list:
    image_paths = list_files_in_folder(folder, ext="PNG")

    LOGGER.info(folder)
    for file_name in image_paths:
        LOGGER.info("\n\n")
        full_file_path = os.path.join(folder, file_name)
        LOGGER.info(f"processing image {full_file_path}")
        filename = full_file_path.split("/")[-1]
        image_name = filename.split(".")[0]
        unique_strings = [
            "no1_93",
            "no1_86",
            "yes1_2",
            "no1_96",
            "no1_97",
            "yes1_79",
            "no1_27",
            "no1_2",
            "no4",
            "no1_71",
            "yes1_60",
            "no1_22",
        ]

        image = cv2.imread(full_file_path)
        h, w = image.shape[:2]
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)

        if file_name in unique_strings or 1:

            # if name  in unique_strings or 1:
            # break
            try:
                with open(full_file_path, "rb") as image_file:
                    image_bytes = BytesIO(image_file.read())

                result_payload = process_image_with_model(image_bytes, threshold, model)
                # Build consent dictionary
                consent = build_consent(result_payload, threshold)

                # Read the image using OpenCV for annotation
                image = cv2.imread(full_file_path)

                if image is None:
                    LOGGER.error("Failed to read the image for annotation.")

                # Prepare annotation parameters
                output_directory = "results_model"
                file_name_prefix = os.path.splitext(os.path.basename(full_file_path))[
                    0
                ]  # Use the original file name as prefix

                # Annotate the image
                annotated_image_path = annotate_image(
                    image=image, consent=consent, output_path=output_directory, file_name_prefix=file_name_prefix
                )

            except Exception as e:
                LOGGER.exception("An unexpected error occurred in the main workflow.")

