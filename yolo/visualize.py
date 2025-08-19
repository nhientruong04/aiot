import queue
import os
import cv2
from typing import Callable, List, Optional, Dict, Tuple, Any
import numpy as np
from common.toolbox import id_to_color

####################################################################
# Visualization
####################################################################

def visualize(output_queue: queue.Queue, cap: cv2.VideoCapture, save_stream_output: bool, output_dir: str, labels: List, quiet: bool = True,
              tracker: Optional["BYTETracker"] = None, fps_tracker: Optional["FrameRateTracker"] = None, side_by_side: bool = False) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue containing (frame, results, boxes) to visualize.
        cap (cv2.VideoCapture): VideoCapture object (camera or video file) or None.
        save_stream_output (bool): Whether to save output video stream to disk.
        output_dir (str): Directory where output video will be saved.
        labels (list): List of class labels.
        quiet (bool): choose whether to show run real-time with cv2 window. Default True to make no window.
        tracker (BYTETracker, optional): ByteTrack tracker instance.
        fps_tracker (FrameRateTracker, optional): Instance of a frame rate tracking class to monitor and log FPS.
        side_by_side (bool): If True, assumes callback generates a side-by-side comparison (original vs. processed),
                             and output frame width will be doubled.
    """

    image_id = 0
    out = None

    if cap is not None:
        if not quiet:
            #Create a named window
            cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
            #Set the window to fullscreen
            cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if save_stream_output:
            # Read video dimensions
            base_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            base_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Double width if side-by-side visualization is expected
            frame_width = base_width * 2 if side_by_side else base_width
            frame_height = base_height

            # Setup video writer
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "output.avi")
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                cap.get(cv2.CAP_PROP_FPS),
                (frame_width, frame_height)
            )

    while True:

        result = output_queue.get()
        if result is None: break

        # WARNING: the extra has been discarded
        # Unpack the result tuple into original frame, inference results, and optional extra context
        original, infer = result  # result can be (original, infer) or (original, infer, extra)

        frame_with_detections = draw_detections(infer, original, labels, tracker)

        if fps_tracker is not None:
            fps_tracker.increment()

        if cap is not None:
            if not quiet:
                # Display output
                cv2.imshow("Output", frame_with_detections)
            if save_stream_output:
                out.write(frame_with_detections)
        else:
            cv2.imwrite(os.path.join(output_dir, f"output_{image_id}.png"), frame_with_detections)

        # Wait for key press "q"
        image_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Close the window and release the camera
            if save_stream_output:
                out.release()  # Release the VideoWriter object
            cap.release()
            # if not quiet:

            cv2.destroyAllWindows()
            break

    if cap is not None and save_stream_output:
        out.release()  # Release the VideoWriter object

    output_queue.task_done()  # Indicate that processing is complete


def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None):
    """
    Draw detections or tracking results on the image.

    Args:
        detections (dict): Raw detection outputs.
        img_out (np.ndarray): Image to draw on.
        labels (list): List of class labels.
        enable_tracking (bool): Whether to use tracker output (ByteTrack).
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Annotated image.
    """

    #extract detection data from the dictionary
    boxes = detections["detection_boxes"]  # List of [xmin,ymin,xmaxm, ymax] boxes
    scores = detections["detection_scores"]  # List of detection confidences
    num_detections = detections["num_detections"]  # Total number of valid detections
    classes = detections["detection_classes"]  # List of class indices per detection

    if tracker:
        dets_for_tracker = []

        #Convert detection format to [xmin,ymin,xmaxm ymax,score] for tracker
        for idx in range(num_detections):
            box = boxes[idx]  #[x, y, w, h]
            score = scores[idx]
            dets_for_tracker.append([*box, score])

        #skip tracking if no detections passed
        if not dets_for_tracker:
            return img_out

        #run BYTETracker and get active tracks
        online_targets = tracker.update(np.array(dets_for_tracker))

        #draw tracked bounding boxes with ID labels
        for track in online_targets:
            track_id = track.track_id  #unique tracker ID
            x1, y1, x2, y2 = track.tlbr  #bounding box (top-left, bottom-right)
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            color = tuple(id_to_color(classes[best_idx]).tolist())  # color based on class
            if best_idx is None:
                draw_detection(img_out, [xmin, ymin, xmax, ymax], f"ID {track_id}",
                               track.score * 100.0, color, track=True)
            else:
                draw_detection(img_out, [xmin, ymin, xmax, ymax], [labels[classes[best_idx]], f"ID {track_id}"],
                               track.score * 100.0, color, track=True)



    else:
        #No tracking — draw raw model detections
        for idx in range(num_detections):
            color = tuple(id_to_color(classes[idx]).tolist())  #color based on class
            draw_detection(img_out, boxes[idx], [labels[classes[idx]]], scores[idx] * 100.0, color)

    return img_out


def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False):
    """
    Draw box and label for one detection.

    Args:
        image (np.ndarray): Image to draw on.
        box (list): Bounding box coordinates.
        labels (list): List of labels (1 or 2 elements).
        score (float): Detection score.
        color (tuple): Color for the bounding box.
        track (bool): Whether to include tracking info.
    """
    ymin, xmin, ymax, xmax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Compose texts
    top_text = f"{labels[0]}: {score:.1f}%" if not track or len(labels) == 2 else f"{score:.1f}%"
    bottom_text = None

    if track:
        if len(labels) == 2:
            bottom_text = labels[1]
        else:
            bottom_text = labels[0]


    # Set colors
    text_color = (255, 255, 255)  # white
    border_color = (0, 0, 0)      # black

    # Draw top text with black border first
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, border_color, 2, cv2.LINE_AA)
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, text_color, 1, cv2.LINE_AA)

    # Draw bottom text if exists
    if bottom_text:
        pos = (xmax - 50, ymax - 6)
        cv2.putText(image, bottom_text, pos, font, 0.5, border_color, 2, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, 0.5, text_color, 1, cv2.LINE_AA)


def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Finds the index of the detection box with the highest IoU relative to the given tracking box.

    Args:
        track_box (list or tuple): The tracking box in [x_min, y_min, x_max, y_max] format.
        detection_boxes (list): List of detection boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        int or None: Index of the best matching detection, or None if no match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes:
        IoU = (area of intersection) / (area of union)
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxA (list or tuple): [x_min, y_min, x_max, y_max]
        boxB (list or tuple): [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU value between 0 and 1.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)
