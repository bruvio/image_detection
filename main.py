import matplotlib.pyplot as plt
import itertools
import cv2
import numpy as np
import os
import logging
import glob

from tensorflow.keras.models import load_model
from io import BytesIO


# Set up logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def compute_confidence_from_match(match_value):
    """
    Converts a template matching value (0 to 1) to a confidence percentage (0 to 100).
    """
    return match_value * 100


def align_images(template, image, show_plots=False):
    """
    Aligns the template to the image using feature matching and homography.

    Parameters:
        template (numpy.ndarray): Grayscale template image.
        image (numpy.ndarray): Grayscale image to align with.
        show_plots (bool): If True, displays plots of matching keypoints.

    Returns:
        aligned_template (numpy.ndarray): Template aligned to the image.
        homography (numpy.ndarray): Homography matrix used for alignment.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)

    # Find the keypoints and descriptors with ORB
    keypoints_template, descriptors_template = orb.detectAndCompute(template, None)
    keypoints_image, descriptors_image = orb.detectAndCompute(image, None)

    if descriptors_template is None or descriptors_image is None:
        LOGGER.debug("Descriptors not found. Returning original template.")
        return template, None

    # Match descriptors using KNN with k=2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors_template, descriptors_image, k=2)

    # Apply ratio test
    good_matches = []
    ratio_threshold = 0.75
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        elif len(match) == 1:
            LOGGER.debug("Only one match found for a descriptor; skipping ratio test.")

    if len(good_matches) < 4:
        LOGGER.debug("Not enough good matches found. Returning original template.")
        return template, None

    # Extract location of good matches
    points_template = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_image = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    homography, mask = cv2.findHomography(points_template, points_image, cv2.RANSAC)

    if homography is not None:
        # Use homography to warp template to align with image
        height, width = image.shape
        aligned_template = cv2.warpPerspective(template, homography, (width, height))

        LOGGER.debug("Template aligned using homography.")

        if show_plots:
            # Draw matches
            img_matches = cv2.drawMatches(
                template,
                keypoints_template,
                image,
                keypoints_image,
                good_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            plt.figure(figsize=(12, 6))
            plt.imshow(img_matches)
            plt.title("Feature Matches")
            plt.axis("off")
            plt.show()

        return aligned_template, homography
    else:
        LOGGER.debug("Homography could not be computed. Returning original template.")
        return template, None


def template_match(image, template, method=cv2.TM_CCOEFF_NORMED, align=True, show_plots=False):
    """
    Performs template matching by cropping and aligning the template and the image.

    Parameters:
        image (numpy.ndarray): Grayscale image where we search for the template.
        template (numpy.ndarray): Grayscale template image to search for.
        method (int): Template matching method.
        align (bool): If True, aligns the template to the image before matching.
        show_plots (bool): If True, displays plots of intermediate steps.

    Returns:
        match_value (float): Similarity measure between the aligned images.
        location (tuple): Top-left corner of the best match (set to (0,0) in this approach).
        size (tuple): Size of the cropped region used in matching.
    """
    # **Add Dimension Checks**
    if image.ndim != 2:
        LOGGER.error(f"Expected 2D grayscale image, but got {image.ndim} dimensions: {image.shape}")
        raise ValueError("Input image must be a 2D grayscale image.")

    if template.ndim != 2:
        LOGGER.error(f"Expected 2D grayscale template, but got {template.ndim} dimensions: {template.shape}")
        raise ValueError("Template must be a 2D grayscale image.")

    # Step 1: Determine the cropping size
    img_height, img_width = image.shape
    tmpl_height, tmpl_width = template.shape

    crop_height = min(img_height, tmpl_height)
    crop_width = min(img_width, tmpl_width)

    LOGGER.debug(f"Cropping images to size: ({crop_width}, {crop_height})")

    # Step 2: Crop the images from the center
    def crop_center(img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]

    image_cropped = crop_center(image, crop_width, crop_height)
    template_cropped = crop_center(template, crop_width, crop_height)

    if show_plots:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image_cropped, cmap="gray")
        plt.title("Cropped Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(template_cropped, cmap="gray")
        plt.title("Cropped Template")
        plt.axis("off")
        plt.show()

    # Step 3: Align the template to the image
    if align:
        aligned_template, homography = align_images(template_cropped, image_cropped, show_plots=show_plots)
    else:
        aligned_template = template_cropped

    # Step 4: Compute similarity between the aligned images
    match_result = cv2.matchTemplate(image_cropped, aligned_template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
    similarity = max_val  # Since we're using TM_CCOEFF_NORMED

    LOGGER.debug(f"Similarity between aligned images: {similarity}")

    # Since we have cropped and aligned the images, location is set to (0,0)
    location = (0, 0)
    size = (crop_width, crop_height)

    return similarity, location, size


def get_templates_from_subfolder(base_folder, subfolder, extensions=("*.png", "*.PNG")):
    """
    Retrieves a list of template file paths from a specified subfolder within the base folder.

    Parameters:
        base_folder (str): The base directory containing subfolders with templates.
        subfolder (str): The subdirectory name (e.g., 'ticked', 'uncircled', 'unticked').
        extensions (tuple): File extensions to include.

    Returns:
        list: List of full paths to template files.
    """
    # Construct the path to the subfolder
    folder_path = os.path.join(base_folder, subfolder)

    # Verify that the subfolder exists
    if not os.path.isdir(folder_path):
        LOGGER.error(f"Subfolder '{subfolder}' does not exist in '{base_folder}'.")
        return []

    # Initialize an empty list to store template paths
    template_files = []

    # Iterate over the specified extensions to find matching files
    for ext in extensions:
        # Use glob to find files with the current extension
        matched_files = glob.glob(os.path.join(folder_path, ext))
        template_files.extend(matched_files)

    # Log the number of templates found
    LOGGER.info(f"Found {len(template_files)} templates in '{folder_path}'.")

    return template_files


def detect_templates(
    image_gray,
    templates,
    label,
    threshold=0.6,
    show_plots=False,
    method=cv2.TM_CCOEFF_NORMED,
    allow_resize=False,
    align=True,
):
    """
    Detects templates in the image and returns the detection status and confidence.

    Parameters:
        image_gray (numpy.ndarray): Grayscale image.
        templates (list): List of template file paths.
        label (str): Label for logging and display purposes.
        threshold (float): Threshold for detection (0-100).
        show_plots (bool): If True, displays matching results.
        method (int): Template matching method.
        allow_resize (bool): If True, allows resizing of templates.
        align (bool): If True, aligns the templates to the image before matching.

    Returns:
        detected (bool): True if detection is above threshold.
        confidence (float): Confidence value.
    """
    max_match_value = None
    best_template = None
    best_location = None
    best_size = None

    for template_path in templates:
        tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            LOGGER.error(f"Template image not found or failed to load: {template_path}")
            continue  # Skip to the next template

        if tmpl.ndim != 2:
            LOGGER.error(f"Template '{template_path}' is not a valid grayscale image. It has {tmpl.ndim} dimensions.")
            continue  # Skip to the next template

        # Skip resizing in this approach
        LOGGER.debug("Skipping resizing of templates.")

        # Perform template matching with cropping and alignment
        try:
            match_value, location, size = template_match(
                image_gray, tmpl, method=method, align=align, show_plots=show_plots
            )
        except ValueError as ve:
            LOGGER.error(f"Error in template_match: {ve}")
            continue  # Skip to the next template

        # Update best match based on the matching method
        if max_match_value is None:
            max_match_value = match_value
            best_template = template_path
            best_location = location
            best_size = size
        else:
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                # Lower values are better
                if match_value < max_match_value:
                    max_match_value = match_value
                    best_template = template_path
                    best_location = location
                    best_size = size
            else:
                # Higher values are better
                if match_value > max_match_value:
                    max_match_value = match_value
                    best_template = template_path
                    best_location = location
                    best_size = size

    if best_template is not None:
        LOGGER.debug(f"Template '{best_template}' has best match with value: {max_match_value}")
    else:
        LOGGER.debug("No valid template matches found.")

    # Compute confidence
    confidence = compute_confidence_from_match(max_match_value) if max_match_value is not None else 0  # 0 to 100

    detected = confidence >= threshold * 100

    # Visualization code (if applicable)
    if show_plots and best_template is not None:
        # Since we have cropped and aligned images, we can display them directly
        plt.figure()
        plt.imshow(image_gray, cmap="gray")
        plt.title(f"Best match for {label}")
        plt.axis("off")
        plt.show()

    return detected, confidence


def detect_text_circle(image_gray, text_bbox=None, show_plots=False):
    """
    Detects if the text is surrounded by a circle.
    """
    if text_bbox is not None:
        x_text, y_text, w_text, h_text = text_bbox
        roi = image_gray[y_text - 10 : y_text + h_text + 10, x_text - 10 : x_text + w_text + 10]
    else:
        roi = image_gray

    blurred = cv2.GaussianBlur(roi, (9, 9), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=15, maxRadius=50
    )

    confidence = 0
    detected = False

    if circles is not None:
        detected = True
        confidence = 80  # Assign a confidence value based on detection

        LOGGER.debug(f"Circles detected: {len(circles[0])}")

        if show_plots:
            output = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            for circle in circles[0, :]:
                cx, cy, radius = map(int, circle)
                LOGGER.debug(f"Circles detected with centre at {(cx, cy)} and radius {radius}")
                cv2.circle(output, (cx, cy), radius, (0, 255, 0), 2)
            plt.figure(figsize=(6, 6))
            plt.imshow(output)
            plt.title("Detected Circles Around Text")
            plt.axis("off")
            plt.show()
    else:
        LOGGER.debug("No circles detected around text.")

    return detected, confidence


def detect_tick_marks(checkbox_roi, show_plots=False):
    """
    Detects tick marks within a checkbox region by analyzing lines
    and checking for V shapes.

    Parameters:
        checkbox_roi (numpy.ndarray): Grayscale image of the checkbox region.
        show_plots (bool): If True, displays plots of detected lines.

    Returns:
        detected (bool): True if a tick mark is detected.
        confidence (float): Confidence value between 0 and 100.
    """
    # Preprocess ROI for edge detection
    roi_blur = cv2.GaussianBlur(checkbox_roi, (3, 3), 0)
    edges = cv2.Canny(roi_blur, threshold1=50, threshold2=150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=10, maxLineGap=5)

    confidence = 0
    detected = False

    if lines is not None and len(lines) > 0:

        LOGGER.debug(f"Lines detected in checkbox: {len(lines)}")

        # Analyze pairs of lines to detect V shapes
        combinations = list(itertools.combinations(lines, 2))
        shape_detected = False
        for idx, (line1, line2) in enumerate(combinations):
            x1_1, y1_1, x2_1, y2_1 = line1[0]
            x1_2, y1_2, x2_2, y2_2 = line2[0]

            # Calculate angles of the lines
            angle1 = np.degrees(np.arctan2(y2_1 - y1_1, x2_1 - x1_1))
            angle2 = np.degrees(np.arctan2(y2_2 - y1_2, x2_2 - x1_2))
            angle_between = abs(angle1 - angle2)
            if angle_between > 180:
                angle_between = 360 - angle_between

            # Calculate lengths of the lines
            length1 = np.hypot(x2_1 - x1_1, y2_1 - y1_1)
            length2 = np.hypot(x2_2 - x1_2, y2_2 - y1_2)

            # Check if lines intersect
            def lines_intersect(p1, p2, p3, p4):
                """Check if line segments (p1-p2) and (p3-p4) intersect."""
                s1_x = p2[0] - p1[0]
                s1_y = p2[1] - p1[1]
                s2_x = p4[0] - p3[0]
                s2_y = p4[1] - p3[1]

                denominator = -s2_x * s1_y + s1_x * s2_y
                if denominator == 0:
                    return False  # Lines are parallel

                s = (-s1_y * (p1[0] - p3[0]) + s1_x * (p1[1] - p3[1])) / denominator
                t = (s2_x * (p1[1] - p3[1]) - s2_y * (p1[0] - p3[0])) / denominator

                if (0 <= s <= 1) and (0 <= t <= 1):
                    return True
                return False

            intersects = lines_intersect((x1_1, y1_1), (x2_1, y2_1), (x1_2, y1_2), (x2_2, y2_2))

            if intersects:
                LOGGER.debug(
                    f"Lines {idx} intersect at angle {angle_between:.2f} degrees with lengths {length1:.2f}, {length2:.2f}"
                )

                # Define parameters for confidence calculation
                ideal_angle = 40.0  # degrees
                max_angle_deviation = 20.0  # degrees
                min_length = 5.0  # pixels
                max_length = 20.0  # pixels

                # Calculate angle score
                angle_diff = abs(angle_between - ideal_angle)
                angle_score = max(0, (max_angle_deviation - angle_diff) / max_angle_deviation)

                # Calculate length score
                min_len = min(length1, length2)
                length_score = max(0, min(1, (min_len - min_length) / (max_length - min_length)))

                # Calculate confidence
                confidence = angle_score * length_score * 100

                LOGGER.debug(f"V shape detected with confidence {confidence:.2f}")
                shape_detected = True
                break

        if not shape_detected:
            # If no V shape detected, but lines are present
            confidence = 0  # No confidence since no valid shape was found
            LOGGER.debug("Lines detected but no valid V shape formed.")

        detected = shape_detected

        if show_plots:
            roi_color = cv2.cvtColor(checkbox_roi, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.figure(figsize=(4, 4))
            plt.imshow(roi_color)
            plt.title("Detected Lines in Checkbox ROI")
            plt.axis("off")
            plt.show()

    else:
        LOGGER.debug("No lines detected in checkbox.")

    return detected, confidence


def determine_final_result(confidences, threshold):
    """
    Determines the final result based on the confidences of different tests.

    Parameters:
        confidences (list of tuples): List of (test_name, confidence_value)
        threshold (float): Threshold value for detection (0-100)

    Returns:
        final_result (bool): True if positive detection, False otherwise
        test_performed (str): The test that determined the final result
        confidence (float): The confidence of the test performed
    """
    # Define positive and negative tests
    positive_tests = ["tick_marks", "circled", "checkbox_checked"]
    negative_tests = ["not_circled", "checkbox_unchecked"]

    # Separate the confidences into positive and negative detections
    positive_detections = [(test, conf) for test, conf in confidences if test in positive_tests and conf >= threshold]
    negative_detections = [(test, conf) for test, conf in confidences if test in negative_tests and conf >= threshold]

    if positive_detections and negative_detections:
        # Conflict detected; compare confidences
        highest_positive = max(positive_detections, key=lambda x: x[1])
        highest_negative = max(negative_detections, key=lambda x: x[1])

        if highest_positive[1] >= highest_negative[1]:
            final_result = True
            test_performed, confidence = highest_positive
        else:
            final_result = False
            test_performed, confidence = highest_negative
    elif positive_detections:
        # Positive detection
        test_performed, confidence = max(positive_detections, key=lambda x: x[1])
        final_result = True
    elif negative_detections:
        # Negative detection
        test_performed, confidence = max(negative_detections, key=lambda x: x[1])
        final_result = False
    else:
        # No detections meet the threshold
        # Select the test with the highest confidence regardless of threshold
        test_performed, confidence = max(confidences, key=lambda x: x[1])
        final_result = False  # Default to False when uncertain

    return final_result, test_performed, confidence


def annotate_image(image, result_payload, output_path, file_name_prefix="result"):
    """
    Adds an annotation to the image based on the overall detection result and saves it.
    Reduces the font size to fit the text within the image width without scaling the image.

    Parameters:
        image (numpy.ndarray): The original image.
        result_payload (dict): The result from the process_image function.
        output_path (str): Directory where the annotated image will be saved.
        file_name_prefix (str): Prefix for the output file name.

    Returns:
        output_file_path (str): The path to the saved annotated image.
    """
    # Extract the overall detection result
    final_result = result_payload.get("final_result", False)
    test_performed = result_payload.get("test_performed", "")
    confidence = result_payload.get("confidence", 0)

    # Map test_performed to readable message
    test_messages = {
        "circled": "Circled",
        "not_circled": "Not Circled",
        "checkbox_checked": "Ticked",
        "checkbox_unchecked": "Not Ticked",
        "tick_marks": "Tick Marks Detected",
    }

    # Default to 'Unknown' if test_performed is not recognized
    annotation_text = test_messages.get(test_performed, "Unknown")

    # Prepare the annotation text with confidence
    annotation = f"{annotation_text} ({confidence:.1f}%)"

    # Set color based on final result
    if final_result:
        color = (0, 0, 255)  # Red for positive detection
    else:
        color = (0, 255, 0)  # Green for negative detection

    # Add the annotation to the image
    annotated_image = image.copy()
    img_height, img_width = annotated_image.shape[:2]

    # Starting position for the text
    x_start = 10
    y_start = 30

    # Initial font scale and thickness
    font_scale = 1.0
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    text_size, _ = cv2.getTextSize(annotation, font, font_scale, thickness)
    text_width = text_size[0]

    # Minimum font scale to maintain readability
    MIN_FONT_SCALE = 0.3

    # Calculate the maximum allowed width for the text
    max_text_width = img_width - 20  # Leave some margin

    # Scale down font if text is wider than image
    while text_width > max_text_width and font_scale > MIN_FONT_SCALE:
        font_scale -= 0.1
        text_size, _ = cv2.getTextSize(annotation, font, font_scale, thickness)
        text_width = text_size[0]

    # Check if font scale is above minimum threshold
    if font_scale < MIN_FONT_SCALE:
        font_scale = MIN_FONT_SCALE
        text_size, _ = cv2.getTextSize(annotation, font, font_scale, thickness)
        text_width = text_size[0]

    # Adjust y position if text exceeds image height
    if y_start + text_size[1] > img_height - 10:
        LOGGER.debug("Annotation exceeds image height; cannot add annotation.")
    else:
        # Add text to the image
        cv2.putText(annotated_image, annotation, (x_start, y_start), font, font_scale, color, thickness, cv2.LINE_AA)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Prepare the output file path
    output_file_name = f"{file_name_prefix}_annotated.png"
    output_file_path = os.path.join(output_path, output_file_name)

    # Save the annotated image
    cv2.imwrite(output_file_path, annotated_image)

    LOGGER.info(f"Annotated image saved to {output_file_path}")

    return output_file_path


def process_image(image_path, threshold=60, show_plots=False):
    """
    Main function to process the image and perform all detection tasks.

    Parameters:
        image_path (str): Path to the image to process.
        threshold (int): Threshold for detection (0-100).
        show_plots (bool): If True, displays matching results.

    Returns:
        dict: Payload containing detection results and confidences.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # **Load the image in grayscale directly to avoid potential issues**
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at path: {image_path}")
    LOGGER.info(f"\n\nProcessing image: {image_path}")

    # Initialize the results dictionary
    results = {}

    # Define base templates directory
    BASE_TEMPLATES_DIR = "templates"

    # Gather templates from subfolders
    templates_not_circled = get_templates_from_subfolder(BASE_TEMPLATES_DIR, "uncircled")
    templates_checked = get_templates_from_subfolder(BASE_TEMPLATES_DIR, "ticked")
    templates_unchecked = get_templates_from_subfolder(BASE_TEMPLATES_DIR, "unticked")

    # Detect if text is not circled using templates
    if templates_not_circled:
        LOGGER.debug("Comparing against templates of not circled yes/no")
        detected_not_circled, confidence_not_circled = detect_templates(
            image,
            templates_not_circled,
            label="Not Circled Text",
            show_plots=show_plots,
            threshold=threshold / 100,
        )
        results["not_circled"] = {
            "detected": detected_not_circled,
            "confidence": confidence_not_circled,
            "test": "Text not circled using templates",
        }
    else:
        LOGGER.warning("No templates found in 'uncircled' subfolder.")
        results["not_circled"] = {
            "detected": False,
            "confidence": 0,
            "test": "Text not circled using templates (No templates available)",
        }

    # Detect if checkbox is ticked using templates
    if templates_checked:
        LOGGER.debug("Comparing against templates of ticked yes/no")
        detected_checked, confidence_checked = detect_templates(
            image,
            templates_checked,
            label="Checked Checkbox",
            show_plots=show_plots,
            threshold=threshold / 100,
        )
        results["checkbox_checked"] = {
            "detected": detected_checked,
            "confidence": confidence_checked,
            "test": "Checkbox checked using templates",
        }
    else:
        LOGGER.warning("No templates found in 'ticked' subfolder.")
        results["checkbox_checked"] = {
            "detected": False,
            "confidence": 0,
            "test": "Checkbox checked using templates (No templates available)",
        }

    # Detect if checkbox is not ticked using templates
    if templates_unchecked:
        LOGGER.debug("Comparing against templates of not ticked yes/no")
        detected_unchecked, confidence_unchecked = detect_templates(
            image,
            templates_unchecked,
            label="Unchecked Checkbox",
            show_plots=show_plots,
            threshold=threshold / 100,
        )
        results["checkbox_unchecked"] = {
            "detected": detected_unchecked,
            "confidence": confidence_unchecked,
            "test": "Checkbox unchecked using templates",
        }
    else:
        LOGGER.warning("No templates found in 'unticked' subfolder.")
        results["checkbox_unchecked"] = {
            "detected": False,
            "confidence": 0,
            "test": "Checkbox unchecked using templates (No templates available)",
        }

    # Detect if text is circled
    detected_circled, confidence_circled = detect_text_circle(image, text_bbox=None, show_plots=show_plots)
    results["circled"] = {
        "detected": detected_circled,
        "confidence": confidence_circled,
        "test": "Text circled detection",
    }

    # Detect tick marks in checkbox area
    # For this, we need to identify the checkbox area first
    # Assuming the checkbox is detected via template matching
    # We will use the location from the 'checkbox_checked' detection if available

    checkbox_area = None
    if detected_checked and templates_checked:
        # Load the first template from the 'ticked' subfolder
        tmpl_checked = cv2.imread(templates_checked[0], cv2.IMREAD_GRAYSCALE)
        if tmpl_checked is None:
            LOGGER.error(f"Template image not found or failed to load: {templates_checked[0]}")
            raise ValueError(f"Template image not found or failed to load: {templates_checked[0]}")

        # Perform template matching to find the checkbox area
        try:
            match_value, location, size = template_match(image, tmpl_checked)
            x, y = location
            w, h = size
            checkbox_area = image[y : y + h, x : x + w]
            LOGGER.debug(f"Checkbox area located at (x: {x}, y: {y}, w: {w}, h: {h})")
        except ValueError as ve:
            LOGGER.error(f"Error in template_match while locating checkbox: {ve}")
    elif detected_unchecked and templates_unchecked:
        # Load the first template from the 'unticked' subfolder
        tmpl_unchecked = cv2.imread(templates_unchecked[0], cv2.IMREAD_GRAYSCALE)
        if tmpl_unchecked is None:
            LOGGER.error(f"Template image not found or failed to load: {templates_unchecked[0]}")
            raise ValueError(f"Template image not found or failed to load: {templates_unchecked[0]}")

        # Perform template matching to find the checkbox area
        try:
            match_value, location, size = template_match(image, tmpl_unchecked)
            x, y = location
            w, h = size
            checkbox_area = image[y : y + h, x : x + w]
            LOGGER.debug(f"Checkbox area located at (x: {x}, y: {y}, w: {w}, h: {h})")
        except ValueError as ve:
            LOGGER.error(f"Error in template_match while locating checkbox: {ve}")

    if checkbox_area is not None:
        detected_tick_marks, confidence_tick_marks = detect_tick_marks(checkbox_area, show_plots=show_plots)
    else:
        detected_tick_marks = False
        confidence_tick_marks = 0
        LOGGER.debug("Checkbox area not detected; skipping tick marks detection.")

    results["tick_marks"] = {
        "detected": detected_tick_marks,
        "confidence": confidence_tick_marks,
        "test": "Tick marks in checkbox area",
    }

    # Aggregate the confidence values
    confidences = [
        ("not_circled", confidence_not_circled),
        ("checkbox_checked", confidence_checked),
        ("checkbox_unchecked", confidence_unchecked),
        ("circled", confidence_circled),
        ("tick_marks", confidence_tick_marks),
    ]

    # Use the new function to determine the final result
    final_result, test_performed, confidence = determine_final_result(confidences, threshold)

    # Prepare the return payload
    payload = {
        "confidence": confidence,
        "final_result": final_result,
        "test_performed": test_performed,
        "all_attempts": results,
    }

    return payload


def annotate_image_openCV(image, result_payload, output_path, file_name_prefix="result"):
    """
    Adds an annotation to the image based on the overall detection result and saves it.
    Reduces the font size to fit the text within the image width without scaling the image.

    Parameters:
        image (numpy.ndarray): The original image.
        result_payload (dict): The result from the process_image function.
        output_path (str): Directory where the annotated image will be saved.
        file_name_prefix (str): Prefix for the output file name.

    Returns:
        output_file_path (str): The path to the saved annotated image.
    """
    # Extract the overall detection result
    final_result = result_payload.get("final_result", False)
    test_performed = result_payload.get("test_performed", "")
    confidence = result_payload.get("confidence", 0)

    # Map test_performed to readable message
    test_messages = {
        "circled": "Circled",
        "not_circled": "Not Circled",
        "checkbox_checked": "Ticked",
        "checkbox_unchecked": "Not Ticked",
        "tick_marks": "Tick Marks Detected",
    }

    # Default to 'Unknown' if test_performed is not recognized
    annotation_text = test_messages.get(test_performed, "Unknown")

    # Prepare the annotation text with confidence
    annotation = f"{annotation_text} ({confidence:.1f}%)"

    # Set color based on final result
    if final_result:
        color = (0, 0, 255)  # Red for positive detection
    else:
        color = (0, 255, 0)  # Green for negative detection

    # Add the annotation to the image
    annotated_image = image.copy()
    img_height, img_width = annotated_image.shape[:2]

    # Starting position for the text
    x_start = 10
    y_start = 30

    # Initial font scale and thickness
    font_scale = 1.0
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    text_size, _ = cv2.getTextSize(annotation, font, font_scale, thickness)
    text_width = text_size[0]

    # Minimum font scale to maintain readability
    MIN_FONT_SCALE = 0.3

    # Calculate the maximum allowed width for the text
    max_text_width = img_width - 20  # Leave some margin

    # Scale down font if text is wider than image
    while text_width > max_text_width and font_scale > MIN_FONT_SCALE:
        font_scale -= 0.1
        text_size, _ = cv2.getTextSize(annotation, font, font_scale, thickness)
        text_width = text_size[0]

    # Check if font scale is above minimum threshold
    if font_scale < MIN_FONT_SCALE:
        font_scale = MIN_FONT_SCALE
        text_size, _ = cv2.getTextSize(annotation, font, font_scale, thickness)
        text_width = text_size[0]

    # Adjust y position if text exceeds image height
    if y_start + text_size[1] > img_height - 10:
        LOGGER.debug("Annotation exceeds image height; cannot add annotation.")
    else:
        # Add text to the image
        cv2.putText(annotated_image, annotation, (x_start, y_start), font, font_scale, color, thickness, cv2.LINE_AA)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Prepare the output file path
    output_file_name = f"{file_name_prefix}_annotated.png"
    output_file_path = os.path.join(output_path, output_file_name)

    # Save the annotated image
    cv2.imwrite(output_file_path, annotated_image)

    LOGGER.info(f"Annotated image saved to {output_file_path}")

    return output_file_path


##### KERAS model


def get_consent_model(document_id):
    input_paths = [
        "no1.PNG",
        "no2.PNG",
        "yes1.PNG",
        "yes2.PNG",
    ]

    output_paths = [
        "no1_result.PNG",
        "no2_result.PNG",
        "yes1_result.PNG",
        "yes2_result.PNG",
    ]

    LOGGER.info(f"Analysing {document_id}")
    os.environ["S3_BUCKET"] = "hto-consent-audit-bucket-prod"

    folder = f"{document_id}-analysis/"
    consent = {}

    # Process each image using the model
    for input_path, output_path in zip(input_paths, output_paths):
        LOGGER.info("\n\n")
        LOGGER.debug(f"Reading image {input_path}")
        try:
            # Read the image bytes
            with open(folder + input_path, "rb") as f:
                input_bytesio = BytesIO(f.read())

            # Process the image with the model
            predicted_label, is_ticked, confidence = process_image_with_model(input_bytesio)

            # Generate key names (e.g., "no1", "confidence_no1")
            image_name = input_path.split(".")[0]  # Strip file extension
            consent[image_name] = is_ticked
            consent[f"confidence_{image_name}"] = (
                confidence["ticked_confidence"] * 100 if is_ticked else confidence["unticked_confidence"] * 100
            )

        except Exception as e:
            LOGGER.error(f"Error processing image {input_path}: {str(e)}")
            consent[input_path] = {"error": str(e)}

    return consent


def process_image_with_model(input_bytesio: BytesIO, threshold: float, model) -> dict:
    """
    Processes an image and predicts its status as ticked/unticked and circled/not circled.

    Args:
        input_bytesio (BytesIO): The image file in BytesIO format.
        threshold (float): The confidence threshold for including predictions.

    Returns:
        dict: A dictionary containing the prediction results and their confidence scores.
    """
    try:
        # Read image from BytesIO object
        img_bytes = np.frombuffer(input_bytesio.getvalue(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Image decoding failed. Please check the input image format.")

        # Preprocess image for the model
        img_resized = cv2.resize(img, (80, 140))  # Resize to 80x140 (width x height)
        img_reshaped = img_resized.reshape(-1, 140, 80, 1)  # Reshape to (batch_size, height, width, channels)
        img_normalized = img_reshaped / 255.0  # Normalize pixel values

        # Run the model to predict the status
        predictions = model.predict(img_normalized)
        LOGGER.info(
            f"Raw predictions: ticked {predictions[0][0]} - unticked {predictions[0][1]} - circled_yes {predictions[0][2]} - circled_no {predictions[0][3]}"
        )

        # Define label indices based on your model's output
        # Assuming the model outputs [ticked, unticked, circled_yes, circled_no]
        ticked_confidence = predictions[0][0]
        unticked_confidence = predictions[0][1]
        circled_yes_confidence = predictions[0][2]
        circled_no_confidence = predictions[0][3]

        response = {
            "ticked": float(ticked_confidence),
            "unticked": float(unticked_confidence),
            "circled_yes": float(circled_yes_confidence),
            "circled_no": float(circled_no_confidence),
        }

        LOGGER.info(f"Processed response: {response}")
        return response

    except Exception as e:
        LOGGER.exception("An error occurred in process_image_with_model.")
        return {"error": str(e)}


def build_consent(response_payload: dict, threshold: float) -> dict:
    """
    Builds a consent dictionary from the response payload by selecting the label with the highest confidence.

    Args:
        response_payload (dict): The response payload from process_image_with_model.
        threshold (float): The confidence threshold for determining certainty.

    Returns:
        dict: A dictionary with 'prediction' and 'confidence' keys.
              If an error is present, it includes an 'error' key.
    """
    try:
        if "error" in response_payload:
            LOGGER.error(f"Error in response payload: {response_payload['error']}")
            return {"prediction": "error", "confidence": 0.0}

        # Define label to status mapping
        label_mapping = {
            "ticked": "ticked",
            "unticked": "unticked",
            "circled_yes": "circled",
            "circled_no": "not circled",
        }

        # Find the label with the highest confidence
        max_label, max_confidence = max(response_payload.items(), key=lambda item: item[1])
        LOGGER.debug(f"Max label: {max_label} with confidence: {max_confidence}")

        # Check if the highest confidence meets the threshold
        if max_confidence >= threshold:
            status = label_mapping.get(max_label, "unknown")
            confidence_percentage = max_confidence * 100  # Convert to percentage
        else:
            status = "uncertain"
            confidence_percentage = max_confidence * 100  # Still represent as percentage

        consent = {"prediction": status, "confidence": round(confidence_percentage, 2)}  # Rounded to 2 decimal places

        LOGGER.info(f"Consent built: {consent}")
        return consent

    except Exception as e:
        LOGGER.exception("An exception occurred while building consent.")
        return {"prediction": "error", "confidence": 0.0}


def annotate_image(image, consent: dict, output_path: str, file_name_prefix: str = "result") -> str:
    """
    Adds an annotation to the image based on the consent dictionary and saves it.

    Parameters:
        image (numpy.ndarray): The original image.
        consent (dict): The consent dictionary containing prediction and confidence.
        output_path (str): Directory where the annotated image will be saved.
        file_name_prefix (str): Prefix for the output file name.

    Returns:
        str: The path to the saved annotated image.
    """
    try:
        # Extract information from consent
        prediction = consent.get("prediction", "Unknown")
        confidence = consent.get("confidence", 0.0)

        # Determine annotation text and color based on prediction
        if prediction == "error":
            annotation_text = "Error in processing image."
            color = (0, 0, 255)  # Red
        elif prediction == "uncertain":
            annotation_text = f"Uncertain ({confidence:.2f}%)"
            color = (0, 255, 255)  # Yellow
        elif prediction in ["ticked", "circled"]:
            annotation_text = f"{prediction.capitalize()} ({confidence:.2f}%)"
            color = (0, 0, 255)  # Red for positive detection
        elif prediction in ["unticked", "not circled"]:
            annotation_text = f"{prediction.replace('_', ' ').capitalize()} ({confidence:.2f}%)"
            color = (0, 255, 0)  # Green for negative detection
        else:
            annotation_text = f"{prediction.capitalize()} ({confidence:.2f}%)"
            color = (255, 255, 255)  # White for unknown

        # Add the annotation to the image
        annotated_image = image.copy()
        img_height, img_width = annotated_image.shape[:2]

        # Starting position for the text
        x_start = 10
        y_start = 30

        # Initial font scale and thickness
        font_scale = 1.0
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size
        text_size, _ = cv2.getTextSize(annotation_text, font, font_scale, thickness)
        text_width = text_size[0]

        # Minimum font scale to maintain readability
        MIN_FONT_SCALE = 0.3

        # Calculate the maximum allowed width for the text
        max_text_width = img_width - 20  # Leave some margin

        # Scale down font if text is wider than image
        current_font_scale = font_scale
        while text_width > max_text_width and current_font_scale > MIN_FONT_SCALE:
            current_font_scale -= 0.1
            text_size, _ = cv2.getTextSize(annotation_text, font, current_font_scale, thickness)
            text_width = text_size[0]

        # Check if font scale is above minimum threshold
        if current_font_scale < MIN_FONT_SCALE:
            current_font_scale = MIN_FONT_SCALE
            text_size, _ = cv2.getTextSize(annotation_text, font, current_font_scale, thickness)
            text_width = text_size[0]

        # Adjust y position if text exceeds image height
        if y_start + text_size[1] > img_height - 10:
            LOGGER.warning("Annotation exceeds image height; cannot add annotation.")
        else:
            # Add text to the image
            cv2.putText(
                annotated_image,
                annotation_text,
                (x_start, y_start),
                font,
                current_font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Prepare the output file path
        output_file_name = f"{file_name_prefix}_annotated.png"
        output_file_path = os.path.join(output_path, output_file_name)

        # Save the annotated image
        cv2.imwrite(output_file_path, annotated_image)

        LOGGER.info(f"Annotated image saved to {output_file_path}")
        return output_file_path

    except Exception as e:
        LOGGER.exception("An exception occurred while annotating the image.")
        return ""
