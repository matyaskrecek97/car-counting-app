import sys
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


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


def initialize_tracker_line(image, line_params):
    cv2.line(image, (line_params['start_x'], line_params['start_y']),
             (line_params['end_x'], line_params['end_y']), (0, 255, 0), 2)


def process_frame(cap, detector, line_params):
    count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from camera.'
            )

        image = cv2.flip(image, 1)

        initialize_tracker_line(image, line_params)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        detection_result = detector.detect(input_tensor)

        image = utils.visualize(image, detection_result)

        # Display the count
        cv2.putText(image, f'Count: {count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

    custom_line_params = {
        'start_x': width // 2,
        'start_y': 0,
        'end_x': width // 2,
        'end_y': height
    }

    process_frame(cap, detector, custom_line_params)


if __name__ == '__main__':
    run()
