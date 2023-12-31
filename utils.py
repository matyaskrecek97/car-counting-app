import cv2
import numpy as np
from tflite_support.task import processor

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 255, 0)  # green


def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
    categories: list,
) -> np.ndarray:
    for detection in detection_result.detections:
        category = detection.categories[0]

        # if categories arg is not None, draw all categories
        # else draw only the categories matching the detection
        category_name = category.category_name
        if categories is not None:
            if category_name not in categories:
                continue

        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

        # Draw label and score
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x,
                         _MARGIN + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        # Draw center point
        center_x = int(bbox.origin_x + bbox.width / 2)
        center_y = int(bbox.origin_y + bbox.height / 2)
        cv2.circle(image, (center_x, center_y), 5, _TEXT_COLOR, -1)

    return image
