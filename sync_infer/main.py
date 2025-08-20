from common.infer_model import HailoInfer
from common.toolbox import init_input_source, get_labels, load_json_file, preprocess, FrameRateTracker, extract_objects
from common.tracker.byte_tracker import BYTETracker
from typing import Callable
from functools import partial
import cv2
import threading
import queue
from .preprocess import preprocess_from_cap
from .postprocess import inference_result_handler
from .visualize import visualize

import time
import os
from loguru import logger
import numpy as np

def main(yolo_net: str, input: str, labels: str, yolo_config: str, output_dir: str,
         resolution: str = 'hd', show_fps: bool = True, batch_size: int = 1) -> None:
    os.makedirs(output_dir, exist_ok=True)

    labels = get_labels(labels)
    config_data = load_json_file(yolo_config)

    cap, images = init_input_source(input, batch_size, resolution)
    tracker = None
    fps_tracker = None

    if show_fps:
        fps_tracker = FrameRateTracker()

    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(orig_width, orig_height)

    visualize_queue = queue.Queue()
    yolo_postprocess_handler_fn = partial(
        inference_result_handler, orig_height=orig_height,
        orig_width=orig_width, config_data=config_data
    )

    signal_thread_kill_event = threading.Event()
    signal_thread_kill_event.clear()
    signal = threading.Event()

    yolo = HailoInfer(yolo_net, batch_size)
    model_height, model_width, _ = yolo.get_input_shape()

    signal_thread = threading.Thread(target=esp_signal, args=(1, signal, signal_thread_kill_event))
    visualize_thread = threading.Thread(
        target=visualize, args=(visualize_queue, cap, output_dir, labels, fps_tracker)
    )

    visualize_thread.start()
    signal_thread.start()

    try:
        image_id = 0
        if show_fps:
            fps_tracker.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = preprocess_from_cap(frame, model_width, model_height)

            if signal.is_set():
                output = yolo.run_sync(processed_frame)
                processed_dets = yolo_postprocess_handler_fn(output)

                extracted_objects = extract_objects(frame, processed_dets['detection_boxes'])
                for extracted_object in extracted_objects:
                    cv2.imwrite(os.path.join(output_dir, f"output_{image_id}.png"), extracted_object)
                    image_id += 1

                visualize_queue.put((frame, processed_dets))
                signal.clear()
            else:
                visualize_queue.put((frame, None))

    except Exception as e:
        logger.error(f'Encountered error in infer pipeline: {e}.')


    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())

    logger.info('Inference was successful!')

    signal_thread_kill_event.set()
    signal_thread.join()
    visualize_queue.put(None)
    visualize_thread.join()
    yolo.close()


def esp_signal(duration, event: threading.Event, kill_event: threading.Event):
    while True:
        event.set()
        time.sleep(duration)

        if kill_event.is_set():
            break


if __name__ == '__main__':
    yolo_net = '/home/nhien/aiot/models/yolo/yolov10n_cubes.hef'
    # input = '/home/nhien/aiot/data/croissant.mp4'
    input = '/home/nhien/Downloads/cubes.mp4'
    labels = '/home/nhien/aiot/code/common/agnostic.txt'
    yolo_config = '/home/nhien/aiot/code/sync_infer/config.json'
    output_dir = './output'
    main(yolo_net, input, labels, yolo_config, output_dir)
