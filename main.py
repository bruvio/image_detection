import matplotlib.pyplot as plt
import itertools
import cv2
import numpy as np
import os
import logging

# Set up logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def compute_confidence_from_match(match_value):
    """
    Converts a template matching value (0 to 1) to a confidence percentage (0 to 100).
    """
    return match_value * 100


def template_match(image, template, method=cv2.TM_CCOEFF_NORMED, align=True):
    """
    Performs template matching, optionally aligning the template to the image.

    Parameters:
        image (numpy.ndarray): Grayscale image where we search for the template.
        template (numpy.ndarray): Grayscale template image to search for.
        method (int): Template matching method.
        align (bool): If True, aligns the template to the image before matching.

    Returns:
        match_value (float): Match value from template matching.
        location (tuple): Top-left corner of the best match.
        size (tuple): Size of the template used in matching.
    """
    if align:
        aligned_template, homography = align_template(image, template)
        template_to_use = aligned_template
    else:
        template_to_use = template

    # Ensure template is not larger than the image
    img_height, img_width = image.shape
    tmpl_height, tmpl_width = template_to_use.shape
    if tmpl_height > img_height or tmpl_width > img_width:

        LOGGER.debug("Template is larger than image. Resizing is required.")
        # Optionally, you can disable resizing here based on a condition
        # For now, let's proceed to match without resizing
        pass

    res = cv2.matchTemplate(image, template_to_use, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        match_value = min_val
        location = min_loc
    else:
        match_value = max_val
        location = max_loc

    size = (tmpl_width, tmpl_height)
    return match_value, location, size


def detect_templates(
    image_gray,
    templates,
    label,
    threshold=0.6,
    show_plots=False,
    method=cv2.TM_CCOEFF_NORMED,
    allow_resize=True,
    align=True,
):
    """
    Detects templates in the image and returns the highest match value.

    Parameters:
        image_gray (numpy.ndarray): Grayscale image.
        templates (list): List of template file paths.
        label (str): Label for logging and display purposes.
        threshold (float): Threshold for detection.
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
        tmpl = cv2.imread(template_path, 0)
        if tmpl is None:
            raise ValueError(f"Template image not found: {template_path}")

        # Optionally disable resizing
        if allow_resize:
            # Resize template if it's larger than the image
            img_height, img_width = image_gray.shape
            tmpl_height, tmpl_width = tmpl.shape
            if tmpl_height > img_height or tmpl_width > img_width:
                scale_factor = min(img_height / tmpl_height, img_width / tmpl_width, 1.0)
                new_width = max(int(tmpl_width * scale_factor), 1)
                new_height = max(int(tmpl_height * scale_factor), 1)
                tmpl = cv2.resize(tmpl, (new_width, new_height), interpolation=cv2.INTER_AREA)

                LOGGER.debug(f"Resized template '{template_path}' to ({new_width}, {new_height})")
                if show_plots:
                    # Plot the resized template
                    plt.figure()
                    plt.imshow(tmpl, cmap='gray')
                    plt.title(f"Resized Template '{template_path}'")
                    plt.axis('off')
                    plt.show()
        else:

            LOGGER.debug("Resizing of templates is disabled.")

        # Perform template matching with optional alignment
        match_value, location, size = template_match(image_gray, tmpl, method=method, align=align)

        

        # Update best match
        if max_match_value is None or (
            (method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and match_value < max_match_value)
            or (method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and match_value > max_match_value)
        ):
            max_match_value = match_value
            best_template = template_path
            best_location = location
            best_size = size
    LOGGER.debug(f"Best match with template '{best_template}' match value: {max_match_value}")

    # Compute confidence
    confidence = compute_confidence_from_match(max_match_value, method)
    detected = confidence >= threshold * 100

    # Visualization code (unchanged)
    if show_plots and best_template is not None and best_location is not None:
        top_left = best_location
        w, h = best_size
        bottom_right = (top_left[0] + w, top_left[1] + h)
        result_image = image_gray.copy()
        cv2.rectangle(result_image, top_left, bottom_right, 255, 2)
        plt.imshow(result_image, cmap="gray")
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
                LOGGER.debug(f"Circles detected with centre at {(cx, cy,)} and radius {radius}")
                cv2.circle(output, (cx, cy), radius, (0, 255, 0), 2)
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


def process_image(image_path, threshold=60, show_plots=False, allow_resize=False, align=False):
    """
    Main function to process the image and perform all detection tasks.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at path: {image_path}")
    LOGGER.info(f"\n\n processing image \t {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the results dictionary
    results = {}

    # Detect if text is not circled using templates
    templates_not_circled = [
        "template_yes.PNG",
        "template_yes_2.PNG",
        "template_yes_3.PNG",
        "template_no.PNG",
        "template_no_2.PNG",
        "template_no_3.PNG",
        "template_no_4.PNG",
        "template_no_5.PNG",
    ]
    detected_not_circled, confidence_not_circled = detect_templates(
        gray,
        templates_not_circled,
        label="Not Circled Text",
        threshold=threshold / 100,
        show_plots=show_plots,
        method=cv2.TM_CCOEFF_NORMED,
        allow_resize=allow_resize,
        align=align,
    )
    results["not_circled"] = {
        "detected": detected_not_circled,
        "confidence": confidence_not_circled,
        "test": "Text not circled using templates",
    }

    # Detect if checkbox is ticked using templates
    templates_checked = [
        "template_yes_check_6.PNG",
        "template_yes_check_5.PNG",
        "template_yes_check_4.PNG",
        "template_yes_check_2.PNG",
        "template_yes_check_3.PNG",
        "template_yes_check.PNG",
        "template_no_check.PNG",
        "template_no_check_2.PNG",
        "template_yes_check_hand.PNG",
        "template_no_check_hand.PNG",
        "template_no_check_hand_2.PNG",
    ]
    detected_checked, confidence_checked = detect_templates(
        gray,
        templates_checked,
        label="Checked Checkbox",
        show_plots=show_plots,
        threshold=threshold / 100,
        method=cv2.TM_CCOEFF_NORMED,
        allow_resize=allow_resize,
        align=align,
    )
    results["checkbox_checked"] = {
        "detected": detected_checked,
        "confidence": confidence_checked,
        "test": "Checkbox checked using templates",
    }

    # Detect if checkbox is not ticked using templates
    templates_unchecked = [
        "template_yes_uncheck.PNG",
        "template_yes_uncheck_2.PNG",
        "template_yes_uncheck_3.PNG",
        "template_yes_uncheck_4.PNG",
        "template_yes_uncheck_5.PNG",
        "template_no_uncheck.PNG",
        "template_no_uncheck_2.PNG",
        "template_no_uncheck_3.PNG",
        "template_no_uncheck_4.PNG",
        "template_no_uncheck_5.PNG",
        "template_no_uncheck_6.PNG",
    ]
    detected_unchecked, confidence_unchecked = detect_templates(
        gray,
        templates_unchecked,
        label="Unchecked Checkbox",
        show_plots=show_plots,
        threshold=threshold / 100,
        method=cv2.TM_CCOEFF_NORMED,
        allow_resize=allow_resize,
        align=align,
    )
    results["checkbox_unchecked"] = {
        "detected": detected_unchecked,
        "confidence": confidence_unchecked,
        "test": "Checkbox unchecked using templates",
    }

    # Detect if text is circled
    detected_circled, confidence_circled = detect_text_circle(gray, text_bbox=None, show_plots=show_plots)
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
    if detected_checked:
        # Load the template
        tmpl_checked = cv2.imread(templates_checked[0], 0)
        if tmpl_checked is None:
            raise ValueError(f"Template image not found: {templates_checked[0]}")

        # Perform template matching
        _, location, size = template_match(gray, tmpl_checked)
        x, y = location
        w, h = size
        checkbox_area = gray[y : y + h, x : x + w]

    elif detected_unchecked:
        # Load the template
        tmpl_unchecked = cv2.imread(templates_unchecked[0], 0)
        if tmpl_unchecked is None:
            raise ValueError(f"Template image not found: {templates_unchecked[0]}")

        # Perform template matching
        _, location, size = template_match(gray, tmpl_unchecked)
        x, y = location
        w, h = size
        checkbox_area = gray[y : y + h, x : x + w]

    if checkbox_area is not None:
        detected_tick_marks, confidence_tick_marks = detect_tick_marks(checkbox_area, show_plots=show_plots)
    else:
        detected_tick_marks = False
        confidence_tick_marks = 0

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

    # Find the highest confidence value
    highest_confidence_test, highest_confidence = max(confidences, key=lambda x: x[1])

    # Determine the final result based on the threshold
    if highest_confidence >= threshold:
        final_result = True
    else:
        final_result = False

    # Prepare the return payload
    payload = {
        "confidence": highest_confidence,
        "final_result": final_result,
        "test_performed": highest_confidence_test,
        "all_attempts": results,
    }

    return payload


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


import cv2
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def align_template(image, template):
    """
    Aligns the template to the image using feature matching and homography.

    Parameters:
        image (numpy.ndarray): Grayscale image where we search for the template.
        template (numpy.ndarray): Grayscale template image to align.

    Returns:
        aligned_template (numpy.ndarray): Aligned template image.
        homography (numpy.ndarray): Homography matrix used for alignment.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)

    # Find the keypoints and descriptors with ORB
    keypoints_image, descriptors_image = orb.detectAndCompute(image, None)
    keypoints_template, descriptors_template = orb.detectAndCompute(template, None)

    if descriptors_image is None or descriptors_template is None:
        # if debug:
        LOGGER.debug("Descriptors not found. Returning original template.")
        return template, None

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_template, descriptors_image)

    if len(matches) < 4:
        # if debug:
        LOGGER.debug("Not enough matches found. Returning original template.")
        return template, None

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points_template = np.zeros((len(matches), 2), dtype=np.float32)
    points_image = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_template[i, :] = keypoints_template[match.queryIdx].pt
        points_image[i, :] = keypoints_image[match.trainIdx].pt

    # Find homography
    homography, mask = cv2.findHomography(points_template, points_image, cv2.RANSAC)

    if homography is not None:
        # Use homography to warp template to align with image
        height, width = image.shape
        aligned_template = cv2.warpPerspective(template, homography, (width, height))
        # if debug:
        LOGGER.debug("Template aligned using homography.")
        return aligned_template, homography
    else:
        # if debug:
        LOGGER.debug("Homography could not be computed. Returning original template.")
        return template, None


def compute_confidence_from_match(match_value, method):
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # Lower values are better
        confidence = (1 - match_value) * 100
    else:
        # Higher values are better
        confidence = match_value * 100
    return confidence
