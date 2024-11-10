import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import logging
import os

# Set up logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# def detect_checkbox_with_tick(gray, area_threshold=500, epsilon_multiplier=0.032, debug=True, show_plots=False):

#     """
#     Detects a checkbox in the image and checks if it contains a tick mark.

#     Parameters:
#         gray (numpy.ndarray): Grayscale image.
#         area_threshold (int): Minimum area of contours to be considered.
#         debug (bool): If True, prints debug information.
#         show_plots (bool): If True, shows plots of detected lines.

#     Returns:
#         tick_mark_detected (bool): True if a tick mark is detected in the checkbox.
#         detection_info (dict): Information about the detection.
#     """
#     LOGGER.info("detecting checkbox in the image")
#     # Thresholding
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 21, 3
#     )

#     # Find contours
#     contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = contours_info[-2]

#     # Find the largest square contour (checkbox)
#     checkbox_contour = None
#     max_area = area_threshold


#     # Create a copy of the image for visualization
#     image_copy = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

#     # Loop over contours
#     for idx, c in enumerate(cnts):
#         area = cv2.contourArea(c)
#         LOGGER.debug(f"Contour {idx}: area = {area}")

#         # Draw current contour
#         contour_image = image_copy.copy()
#         cv2.drawContours(contour_image, [c], -1, (255, 0, 0), 2)  # Blue contour

#         if area > max_area:
#             epsilon = epsilon_multiplier * cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, epsilon, True)
#             LOGGER.debug(f"Contour {idx}: polygon has {len(approx)} edges")

#             # Draw approximated polygon
#             cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 2)  # Green polygon

#             if len(approx) > 4:
#                 x, y, w, h = cv2.boundingRect(approx)
#                 aspect_ratio = float(w) / h
#                 LOGGER.debug(f"Contour {idx}: aspect ratio = {aspect_ratio}")

#                 # Annotate the image with properties
#                 cv2.putText(contour_image, f"Area: {area:.1f}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#                 cv2.putText(contour_image, f"Edges: {len(approx)}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#                 cv2.putText(contour_image, f"AR: {aspect_ratio:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#                 # if 0.9 < aspect_ratio < 1.1:
#                 checkbox_contour = c
#                 max_area = area
#                 LOGGER.debug(f"Contour {idx}: checkbox candidate found")

#                     # Highlight the checkbox contour in red
#                 cv2.drawContours(contour_image, [approx], -1, (0, 0, 255), 2)  # Red contour

#             if show_plots:
#                 plt.figure(figsize=(6, 6))
#                 plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
#                 plt.title(f"Contour {idx} Processing")
#                 plt.axis('off')
#                 plt.show()

#     if checkbox_contour is None:

#         LOGGER.debug("No checkbox detected.")
#         return False, {"num_lines": 0, "avg_angle": 0, "avg_length": 0}

#     # Get bounding box of the checkbox
#     x, y, w, h = cv2.boundingRect(checkbox_contour)
#     LOGGER.debug(f"Checkbox bounding box - x: {x}, y: {y}, w: {w}, h: {h}")

#     # Extract ROI
#     checkbox_roi = gray[y:y+h, x:x+w]

#     # Preprocess ROI for edge detection
#     roi_blur = cv2.GaussianBlur(checkbox_roi, (3, 3), 0)
#     edges = cv2.Canny(roi_blur, threshold1=50, threshold2=150, apertureSize=3)

#     # Apply Hough Line Transform
#     lines = cv2.HoughLinesP(
#         edges, rho=1, theta=np.pi/180, threshold=10,
#         minLineLength=30, maxLineGap=5
#     )

#     # Analyze lines
#     tick_mark_detected = False
#     detection_info = {"num_lines": 0, "avg_angle": 0, "avg_length": 0}
#     if lines is not None:
#         detection_info["num_lines"] = len(lines)
#         angles = []
#         lengths = []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
#             length = np.hypot(x2 - x1, y2 - y1)
#             angles.append(angle)
#             lengths.append(length)
#             # # if debug:
#             LOGGER.debug(f"Line: ({x1}, {y1}) to ({x2}, {y2}), Angle: {angle:.2f}, Length: {length:.2f}")
#             if (20 < angle < 60) and length > 30:
#                 tick_mark_detected = True
#                 break  # Stop if tick mark detected
#             else:
#                 print(f"line lenght {length} and angle {angle} do not match logic")

#         # Populate detection_info with line analysis results
#         if angles:
#             detection_info["avg_angle"] = np.mean(angles)
#         if lengths:
#             detection_info["avg_length"] = np.mean(lengths)

#     if show_plots:
#         roi_color = cv2.cvtColor(checkbox_roi, cv2.COLOR_GRAY2BGR)
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(roi_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         plt.imshow(roi_color, cmap='gray')
#         plt.title("Detected Lines in Checkbox ROI")
#         plt.axis('off')
#         plt.show()

#     return tick_mark_detected, detection_info

# def detect_text_circle(gray, text_bbox=None, debug=True, show_plots=False):
#     """
#     Detects if the text is surrounded by a circular contour.
#     """
#     LOGGER.info("detecting circled text")
#     if text_bbox is None:
#         # # if debug:
#         LOGGER.debug("No text bounding box provided. Skipping circle detection.")
#         return False

#     x_text, y_text, w_text, h_text = text_bbox
#     # Focus on the region around the text
#     margin = 10
#     x_start = max(x_text - margin, 0)
#     y_start = max(y_text - margin, 0)
#     x_end = x_text + w_text + margin
#     y_end = y_text + h_text + margin
#     roi = gray[y_start:y_end, x_start:x_end]

#     # Detect circles in the ROI
#     circles = cv2.HoughCircles(
#         roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
#         param1=50, param2=30, minRadius=0, maxRadius=0
#     )

#     if circles is not None:
#         # # if debug:
#         LOGGER.debug(f"Circles detected: {len(circles[0])}")
#         if show_plots:
#             output = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
#             for circle in circles[0, :]:
#                 cx, cy, radius = map(int, circle)
#                 cv2.circle(output, (cx, cy), radius, (0, 255, 0), 2)
#             plt.imshow(output)
#             plt.title("Detected Circles Around Text")
#             plt.axis('off')
#             plt.show()
#         return True
#     else:
#         # if debug:
#             LOGGER.debug("No circled text detected.")
#         return False


# def explore_detection_parameters(image_path, param_ranges, debug=True):
#     """
#     Explores detection parameters over specified ranges and visualizes the results.

#     Parameters:
#         image_path (str): Path to the image file.
#         param_ranges (dict): Dictionary of parameter ranges to explore.
#         debug (bool): If True, prints debug information.
#     """
#     from itertools import product

#     # Extract parameter ranges
#     clahe_clip_limits = param_ranges.get('clahe_clip_limits', [1.5])
#     tile_grid_sizes = param_ranges.get('tile_grid_sizes', [(4, 4)])
#     block_sizes = param_ranges.get('block_sizes', [21])
#     C_values = param_ranges.get('C_values', [3])
#     area_thresholds = param_ranges.get('area_thresholds', [100])

#     combinations = list(product(clahe_clip_limits, tile_grid_sizes, block_sizes, C_values, area_thresholds))

#     for params in combinations:
#         clahe_clip_limit, tile_grid_size, block_size, C_value, area_threshold = params

#         # Load and preprocess the image
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError(f"Failed to load image at path: {image_path}")

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Apply CLAHE
#         clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tile_grid_size)
#         enhanced = clahe.apply(gray)

#         # Thresholding
#         thresh = cv2.adaptiveThreshold(
#             enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV, block_size, C_value
#         )

#         # Show thresholded image
#         plt.imshow(thresh, cmap='gray')
#         plt.title(f'CLAHE: {clahe_clip_limit}, Grid: {tile_grid_size}, Block: {block_size}, C: {C_value}')
#         plt.axis('off')
#         plt.show()

#         # Proceed with detection
#         ticked, detection_info = detect_checkbox_with_tick(
#             gray, area_threshold=area_threshold, show_plots=True
#         )

#         print(f"Parameters: CLAHE: {clahe_clip_limit}, Grid: {tile_grid_size}, Block: {block_size}, C: {C_value}, Area Threshold: {area_threshold}")
#         print(f"Ticked: {ticked}, Detection Info: {detection_info}")


# def template_matching_text_detection(gray_image, debug=True, show_plots=False):
#     """
#     Uses template matching to detect "YES" or "NO" in the image.
#     Resizes the template if it's larger than the image.

#     Parameters:
#         gray_image (numpy.ndarray): Grayscale image.
#         debug (bool): If True, prints debug information.
#         show_plots (bool): If True, displays the matching result.

#     Returns:
#         text_detected (str): Detected text ("YES" or "NO") or None.
#         text_bbox (tuple): Bounding box of detected text (x, y, w, h) or None.
#     """
#     # Load templates
#     template_yes = cv2.imread('template_yes.png', 0)
#     template_no = cv2.imread('template_no.png', 0)
#     templates = [('YES', template_yes), ('NO', template_no)]

#     text_detected = None
#     text_bbox = None
#     max_match_value = 0
#     method = cv2.TM_CCOEFF_NORMED

#     img_height, img_width = gray_image.shape

#     for label, template in templates:
#         if template is None:
#             raise ValueError(f"Template image for '{label}' not found.")

#         # Resize the template if necessary
#         tmpl_height, tmpl_width = template.shape

#         # Check if template is larger than image and resize if needed
#         scale_factor = 1.0
#         if tmpl_height > img_height or tmpl_width > img_width:
#             # Calculate the scale factor to resize the template
#             height_scale = img_height / tmpl_height
#             width_scale = img_width / tmpl_width
#             scale_factor = min(height_scale, width_scale, 1.0)  # Ensure scale_factor <= 1.0

#             new_width = int(tmpl_width * scale_factor)
#             new_height = int(tmpl_height * scale_factor)
#             template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)

#             # if debug:
#                 LOGGER.debug(f"Resized template '{label}' to ({new_width}, {new_height})")

#         w, h = template.shape[::-1]

#         # Proceed with template matching
#         res = cv2.matchTemplate(gray_image, template, method)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#         # if debug:
#             LOGGER.debug(f"Template matching '{label}': max_val={max_val}")

#         # Update the best match if this template match is better
#         if max_val > max_match_value and max_val > 0.7:  # Threshold can be adjusted
#             max_match_value = max_val
#             text_detected = label
#             top_left = max_loc
#             x_text, y_text = top_left
#             w_text, h_text = w, h
#             text_bbox = (x_text, y_text, w_text, h_text)

#             if show_plots:
#                 # Draw rectangle on a copy of the image
#                 img_display = gray_image.copy()
#                 bottom_right = (x_text + w_text, y_text + h_text)
#                 cv2.rectangle(img_display, top_left, bottom_right, 255, 2)
#                 plt.imshow(img_display, cmap='gray')
#                 plt.title(f"Detected '{label}' via Template Matching")
#                 plt.axis('off')
#                 plt.show()

#     return text_detected, text_bbox


# def detect_selection(image_path, check_checkbox=True, check_circle=True, debug=True, show_plots=False):
#     """
#     Main function to detect if a checkbox is ticked or if the text is circled.
#     """
#     if not os.path.isfile(image_path):
#         raise FileNotFoundError(f"Image file not found: {image_path}")

#     # Load and preprocess the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Failed to load image at path: {image_path}")

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # OCR to detect "YES" or "NO"
#     ocr_result_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
#     text_detected = None
#     text_bbox = None

#     for i in range(len(ocr_result_data['text'])):
#         word = ocr_result_data['text'][i].strip().upper()
#         if word in ["YES", "NO"]:
#             text_detected = word
#             (x_text, y_text, w_text, h_text) = (
#                 ocr_result_data['left'][i],
#                 ocr_result_data['top'][i],
#                 ocr_result_data['width'][i],
#                 ocr_result_data['height'][i]
#             )
#             text_bbox = (x_text, y_text, w_text, h_text)
#             break

#     # If OCR fails, use template matching
#     if text_bbox is None:
#         # if debug:
#             LOGGER.debug("No 'YES' or 'NO' detected in OCR. Using template matching.")
#         text_detected, text_bbox = template_matching_text_detection(gray, show_plots=show_plots)

#     if text_bbox is None:
#         # if debug:
#             LOGGER.debug("Text detection failed after template matching.")
#         # Proceed to detect circles without relying on text_bbox
#         detection_info = {"text_detected": None}
#         if check_circle:
#             # if debug:
#                 LOGGER.debug("Attempting circle detection without text_bbox.")
#             circled = detect_circle_in_image(gray, show_plots=show_plots)
#             detection_info["circle_detected"] = circled
#             if circled:
#                 return True, detection_info
#         # Optionally, proceed with checkbox detection
#         if check_checkbox:
#             ticked, checkbox_info = detect_checkbox_with_tick(gray, show_plots=show_plots)
#             detection_info.update(checkbox_info)
#             if ticked:
#                 detection_info["checkbox_detected"] = True
#                 return True, detection_info
#             detection_info["checkbox_detected"] = False
#         # No selection detected
#         return False, detection_info

#     detection_info = {"text_detected": text_detected}

#     # if debug:
#         LOGGER.debug(f"Text detected: {text_detected} at {text_bbox}")

#     # Proceed with checkbox detection
#     if check_checkbox:
#         ticked, checkbox_info = detect_checkbox_with_tick(gray, show_plots=show_plots)
#         detection_info.update(checkbox_info)
#         if ticked:
#             detection_info["checkbox_detected"] = True
#             return True, detection_info
#         detection_info["checkbox_detected"] = False

#     # Proceed with circle detection using text_bbox
#     if check_circle:
#         circled = detect_text_circle(gray, text_bbox, show_plots=show_plots)
#         detection_info["circle_detected"] = circled
#         if circled:
#             return True, detection_info

#     # No selection detected
#     return False, detection_info


# def detect_circle_in_image(gray, dp=1.2, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100, debug=True, show_plots=False):
#     """
#     Detects circles in the entire image using Hough Circle Transform.

#     Parameters:
#         gray (numpy.ndarray): Grayscale image.
#         dp (float): Inverse ratio of the accumulator resolution to the image resolution.
#         minDist (int): Minimum distance between the centers of the detected circles.
#         param1 (int): Gradient value used to handle edge detection in the Yuen et al. method.
#         param2 (int): Accumulator threshold for the circle centers at the detection stage.
#         minRadius (int): Minimum circle radius.
#         maxRadius (int): Maximum circle radius.
#         debug (bool): If True, prints debug information.
#         show_plots (bool): If True, shows plots of detected circles.

#     Returns:
#         circled (bool): True if a circle is detected.
#     """
#     # Apply blur to reduce noise
#     blurred = cv2.medianBlur(gray, 5)

#     # Detect circles in the image
#     circles = cv2.HoughCircles(
#         blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
#         param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
#     )

#     if circles is not None and len(circles[0]) > 0:
#         # if debug:
#             LOGGER.debug(f"Circles detected: {len(circles[0])}")
#         if show_plots:
#             output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#             for circle in circles[0, :]:
#                 cx, cy, radius = map(int, circle)
#                 cv2.circle(output, (cx, cy), radius, (0, 255, 0), 2)
#             plt.imshow(output)
#             plt.title("Detected Circles in Image")
#             plt.axis('off')
#             plt.show()
#         return True
#     else:
#         # if debug:
#             LOGGER.debug("No circles detected in image.")
#         return False


# def explore_circle_detection_parameters(gray_image, dp_values, minDist_values, param1_values, param2_values, minRadius_values, maxRadius_values):
#     from itertools import product

#     parameter_combinations = list(product(dp_values, minDist_values, param1_values, param2_values, minRadius_values, maxRadius_values))

#     for dp, minDist, param1, param2, minRadius, maxRadius in parameter_combinations:
#         circled = detect_circle_in_image(
#             gray_image,
#             dp=dp,
#             minDist=minDist,
#             param1=param1,
#             param2=param2,
#             minRadius=minRadius,
#             maxRadius=maxRadius,
#             debug=True,
#             show_plots=True
#         )
#         print(f"dp: {dp}, minDist: {minDist}, param1: {param1}, param2: {param2}, minRadius: {minRadius}, maxRadius: {maxRadius}")
#         print(f"Circle detected: {circled}")


####
import cv2
import numpy as np
import pytesseract
import os
import logging
import matplotlib.pyplot as plt

# Set up logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def compute_confidence_from_match(match_value):
    """
    Converts a template matching value (0 to 1) to a confidence percentage (0 to 100).
    """
    return match_value * 100


def template_match(image, template, method=cv2.TM_CCOEFF_NORMED):
    """
    Performs template matching and returns the maximum match value and location.
    """
    res = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc, template.shape[::-1]  # Return match value, location, and template size


def detect_templates(image_gray, templates, label, threshold=0.7, show_plots=False):
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
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=50, maxRadius=150
    )

    confidence = 0
    detected = False

    if circles is not None:
        detected = True
        confidence = 80  # Assign a confidence value based on detection
        # if debug:
        LOGGER.debug(f"Circles detected: {len(circles[0])}")
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


# def detect_tick_marks(checkbox_roi, show_plots=False):
#     """
#     Detects tick marks within a checkbox region.
#     """
#     # Preprocess ROI for edge detection
#     roi_blur = cv2.GaussianBlur(checkbox_roi, (3, 3), 0)
#     edges = cv2.Canny(roi_blur, threshold1=50, threshold2=150, apertureSize=3)

#     # Apply Hough Line Transform
#     lines = cv2.HoughLinesP(
#         edges, rho=1, theta=np.pi/180, threshold=10,
#         minLineLength=10, maxLineGap=5
#     )

#     confidence = 0
#     detected = False

#     if lines is not None and len(lines) > 0:
#         detected = True
#         confidence = 80  # Assign a confidence value based on detection
#         # if debug:
#             LOGGER.debug(f"Lines detected in checkbox: {len(lines)}")
#         if show_plots:
#             roi_color = cv2.cvtColor(checkbox_roi, cv2.COLOR_GRAY2BGR)
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(roi_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             plt.imshow(roi_color)
#             plt.title("Detected Tick Marks")
#             plt.axis('off')
#             plt.show()
#     else:
#         # if debug:
#             LOGGER.debug("No tick marks detected in checkbox.")

# return detected, confidence


import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools


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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the results dictionary
    results = {}

    # Detect if text is not circled using templates
    templates_not_circled = ["template_yes.png", "template_no.png"]
    detected_not_circled, confidence_not_circled = detect_templates(
        gray, templates_not_circled, label="Not Circled Text", show_plots=show_plots
    )
    results["not_circled"] = {
        "detected": detected_not_circled,
        "confidence": confidence_not_circled,
        "test": "Text not circled using templates",
    }

    # Detect if checkbox is ticked using templates
    templates_checked = ["template_yes_check.png", "template_no_check.png"]
    detected_checked, confidence_checked = detect_templates(
        gray, templates_checked, label="Checked Checkbox", show_plots=show_plots
    )
    results["checkbox_checked"] = {
        "detected": detected_checked,
        "confidence": confidence_checked,
        "test": "Checkbox checked using templates",
    }

    # Detect if checkbox is not ticked using templates
    templates_unchecked = ["template_yes_uncheck.png", "template_no_uncheck.png"]
    detected_unchecked, confidence_unchecked = detect_templates(
        gray, templates_unchecked, label="Unchecked Checkbox", show_plots=show_plots
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
        # Extract the checkbox area using the best location and size
        # Re-run the template matching to get the location
        _, location, size = template_match(gray, cv2.imread(templates_checked[0], 0))
        x, y = location
        w, h = size
        checkbox_area = gray[y : y + h, x : x + w]
    elif detected_unchecked:
        # Extract the checkbox area from the unchecked checkbox
        _, location, size = template_match(gray, cv2.imread(templates_unchecked[0], 0))
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
