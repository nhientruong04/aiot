from common.infer_model import HailoInfer
from common.toolbox import init_input_source, get_labels, load_json_file, preprocess, FrameRateTracker, extract_objects
from common.tracker.byte_tracker import BYTETracker
import cv2
import threading
from .preprocess import preprocess_from_cap

from loguru import logger
import numpy as np

def main(yolo_net: str, input: str, labels: str, yolo_config: str,
         resolution: str = 'hd', show_fps: bool = True, batch_size: int = 1) -> None:
    labels = get_labels(labels)
    config_data = load_json_file(yolo_config)

    cap, images = init_input_source(input, batch_size, resolution)
    tracker = None
    fps_tracker = None

    if show_fps:
        fps_tracker = FrameRateTracker()

    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    signal = threading.Event()
    signal.clear()

    yolo = HailoInfer(yolo_net, batch_size)
    model_height, model_width, _ = yolo.get_input_shape()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_from_cap(frame, model_width, model_height)

    yolo.close()

if __name__ == '__main__':
    yolo_net = '/home/nhien/aiot/models/yolo/yolov10n_cubes.hef'
    input = '/home/nhien/aiot/data/croissant.mp4'
    labels = '/home/nhien/aiot/code/common/agnostic.txt'
    yolo_config = "/home/nhien/aiot/code/sync_infer/config.json"
    main(yolo_net, input, labels, yolo_config)
