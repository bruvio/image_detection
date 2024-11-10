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


def template_match(image, template, method=cv2.TM_CCOEFF_NORMED, debug=False):
    """
    Performs template matching and returns the maximum match value and location.
    Resizes the template if it is larger than the image.

    Parameters:
        image (numpy.ndarray): Grayscale image where we search for the template.
        template (numpy.ndarray): Grayscale template image to search for.
        method (int): Template matching method.
        debug (bool): If True, prints debug information.

    Returns:
        max_val (float): Maximum match value.
        max_loc (tuple): Location of the best match.
        template_size (tuple): Size of the template used for matching.
    """
    img_height, img_width = image.shape[:2]
    tmpl_height, tmpl_width = template.shape[:2]

    if debug:
        LOGGER.debug(f"Image size: {img_width}x{img_height}")
        LOGGER.debug(f"Template size before resizing: {tmpl_width}x{tmpl_height}")

    # Check if template is larger than image and resize if necessary
    if tmpl_height > img_height or tmpl_width > img_width:
        scale_factor = min(img_height / tmpl_height, img_width / tmpl_width)
        new_width = max(int(tmpl_width * scale_factor), 1)
        new_height = max(int(tmpl_height * scale_factor), 1)
        template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)
        tmpl_height, tmpl_width = template.shape[:2]
        if tmpl_height == 0 or tmpl_width == 0:
            raise ValueError("Template size after resizing is zero. Cannot perform template matching.")
        if debug:
            LOGGER.debug(f"Resized template size: {tmpl_width}x{tmpl_height}")

    res = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc, template.shape[::-1]  # Return match value, location, and template size


def detect_templates(image_gray, templates, label, threshold=0.6, show_plots=False):
    """
    Detects templates in the image and returns the highest match value.
    """
    max_match_value = 0
    best_template = None
    best_location = None
    best_size = None
    for template in templates:
        tmpl = cv2.imread(template, 0)
        if tmpl is None:
            raise ValueError(f"Template image not found: {template}")

        # Resize template if it's larger than the image
        img_height, img_width = image_gray.shape
        tmpl_height, tmpl_width = tmpl.shape
        if tmpl_height > img_height or tmpl_width > img_width:
            scale_factor = min(img_height / tmpl_height, img_width / tmpl_width, 1.0)
            new_width = int(tmpl_width * scale_factor)
            new_height = int(tmpl_height * scale_factor)
            tmpl = cv2.resize(tmpl, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # if debug:
            LOGGER.debug(f"Resized template '{template}' to ({new_width}, {new_height})")

        match_value, location, size = template_match(image_gray, tmpl)
        # if debug:
        LOGGER.debug(f"Template '{template}' match value: {match_value}")
        if match_value > max_match_value:
            max_match_value = match_value
            best_template = template
            best_location = location
            best_size = size

    confidence = compute_confidence_from_match(max_match_value)
    detected = confidence >= threshold * 100

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
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=50, maxRadius=500
    )

    confidence = 0
    detected = False

    if circles is not None:
        detected = True
        confidence = 80  # Assign a confidence value based on detection
        # if debug:
        LOGGER.debug(f"Circles detected: {len(circles[0])}")
        cx, cy, radius = map(int, circle)
        LOGGER.debug(f"Circles detected with centre at {(cx, cy,)} and radius {radius}")

        if show_plots:
            output = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            for circle in circles[0, :]:
                cx, cy, radius = map(int, circle)
                cv2.circle(output, (cx, cy), radius, (0, 255, 0), 2)
            plt.imshow(output)
            plt.title("Detected Circles Around Text")
            plt.axis("off")
            plt.show()
    else:
        # if debug:
        LOGGER.debug("No circles detected around text.")

    return detected, confidence


def detect_tick_marks(checkbox_roi, show_plots=False):
    """
    Detects tick marks within a checkbox region by analyzing lines
    and checking for V or X shapes.

    Parameters:
        checkbox_roi (numpy.ndarray): Grayscale image of the checkbox region.
        debug (bool): If True, prints debug information.
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

        # Analyze pairs of lines to detect V or X shapes
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
                # # if debug:
                LOGGER.debug(
                    f"Lines {idx} intersect at angle {angle_between:.2f} degrees with lengths {length1:.2f}, {length2:.2f}"
                )

                # Check for V shape (tick mark)
                if 20 <= angle_between <= 60:
                    # Further checks can be added for line lengths and positions
                    shape_detected = True
                    confidence = min(100, (100 - angle_between) + (min(length1, length2) * 2))
                    # # if debug:
                    LOGGER.debug(f"V shape detected with confidence {confidence:.2f}")
                    break

                # Check for X shape (cross mark)
                elif 80 <= angle_between <= 100:
                    shape_detected = True
                    confidence = min(90, (angle_between) + (min(length1, length2) * 2))
                    # # if debug:
                    LOGGER.debug(f"X shape detected with confidence {confidence:.2f}")
                    break

        if not shape_detected:
            # If no V or X shape detected, but lines are present
            confidence = 50  # Lower confidence
            # # if debug:
            LOGGER.debug("Lines detected but no V or X shape formed.")

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
        # if debug:
        LOGGER.debug("No lines detected in checkbox.")

    return detected, confidence


def process_image(image_path, threshold=60, show_plots=False):
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
    templates_not_circled = ["template_yes.png", "template_no.png"]
    detected_not_circled, confidence_not_circled = detect_templates(
        gray,
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

    # Detect if checkbox is ticked using templates
    templates_checked = ["template_yes_check.png", "template_no_check.png"]
    detected_checked, confidence_checked = detect_templates(
        gray,
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

    # Detect if checkbox is not ticked using templates
    templates_unchecked = ["template_yes_uncheck.png", "template_no_uncheck.png"]
    detected_unchecked, confidence_unchecked = detect_templates(
        gray,
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
