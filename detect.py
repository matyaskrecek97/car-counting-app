import sys
import cv2
# import utils

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from centroid_tracker import CentroidTracker


def initialize_camera(camera_id, width, height):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def initialize_detector(model_path, num_threads, enable_edgetpu):
    base_options = core.BaseOptions(
        file_name=model_path, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    return detector


def process_frame(cap, detector, tracker):
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from camera.'
            )

        image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        detection_result = detector.detect(input_tensor).detections

        if not detection_result:
            continue

        objects = []

        for detection in detection_result:
            bounding_box = [detection.bounding_box.origin_x,
                            detection.bounding_box.origin_y,
                            detection.bounding_box.origin_x + detection.bounding_box.width,
                            detection.bounding_box.origin_y + detection.bounding_box.height]
            startX, startY, endX, endY = bounding_box
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            objects.append((cX, cY, startX, startY, endX, endY))

        tracked_objects = tracker.update(objects)

        for (objectID, centroid) in tracked_objects.items():
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Car counting app', image)

    cap.release()
    cv2.destroyAllWindows()


def run():
    camera_id = 0
    width, height = 640, 480
    num_threads = 4
    enable_edgetpu = False
    model_path = 'efficientdet_lite0.tflite'

    cap = initialize_camera(camera_id, width, height)
    detector = initialize_detector(model_path, num_threads, enable_edgetpu)

    maxDisappeared = 50
    tracker = CentroidTracker(maxDisappeared)

    process_frame(cap, detector, tracker)


if __name__ == '__main__':
    run()
