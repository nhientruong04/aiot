import cv2
import numpy as np


def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    for i, x in enumerate(box):
        box[i] = int(x * size)
        if (input_width != size) and (i % 2 != 0):
            box[i] -= padding_length
        if (input_height != size) and (i % 2 == 0):
            box[i] -= padding_length

    return box


def inference_result_handler(detections: list, orig_width: int, orig_height: int, config_data) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.
        config_data (Dict): Loaded JSON config containing post-processing metadata.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)


    #values used for scaling coords and removing padding
    img_height, img_width = orig_height, orig_width
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= score_threshold:
                denorm_bbox = denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)
                all_detections.append((score, class_id, denorm_bbox))

    #sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    #take top max_boxes
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }
