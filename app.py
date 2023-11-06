import threading
import sys
import cv2
import utils
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from tracker import EuclideanDistTracker
from event_sender import EventSender


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


def process_frame(cap, detector, tracker, roi, sender):
    counted_ids = set()
    clear_interval = 1000
    frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from camera.'
            )

        image = cv2.flip(image, 1)

        # Define the ROI (Region of Interest)
        cv2.rectangle(image, (roi[0], roi[1]),
                      (roi[2], roi[3]), (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        detection_result = detector.detect(input_tensor)

        image = utils.visualize(image, detection_result, ["car"])

        # Extract bounding box coordinates from detection result
        objects_rect = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            objects_rect.append((int(bbox.origin_x), int(
                bbox.origin_y), int(bbox.width), int(bbox.height)))

        # Update object tracker
        objects_bbs_ids = tracker.update(objects_rect)

        # Count cars based on objects_bbs_ids result
        for obj_bb_id in objects_bbs_ids:
            x, y, w, h, object_id = obj_bb_id
            if object_id not in counted_ids:
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2

                # Count if the object is within the ROI
                if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                    counted_ids.add(object_id)
                    sender.send_detection_event()

        frame_count += 1

        # Clear the counted_ids after clear_interval frames
        if frame_count % clear_interval == 0:
            counted_ids.clear()

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
    tracker = EuclideanDistTracker()
    sender = EventSender()

    cap = initialize_camera(camera_id, width, height)
    detector = initialize_detector(model_path, num_threads, enable_edgetpu)

    # Define ROI as a square in the center
    roi_size = 400  # Adjust as needed
    roi_x1 = (width - roi_size) // 2
    # roi_y1 = (height - roi_size) // 2
    roi_y1 = 500
    roi_x2 = roi_x1 + roi_size
    roi_y2 = roi_y1 + roi_size
    roi = (roi_x1, roi_y1, roi_x2, roi_y2)

    # Create a thread for the heartbeat
    heartbeat_thread = threading.Thread(target=sender.start_hearbeat)
    # Set the thread as a daemon so it exits when the main thread exits
    heartbeat_thread.daemon = True
    heartbeat_thread.start()

    process_frame(cap, detector, tracker, roi, sender)


if __name__ == '__main__':
    run()
