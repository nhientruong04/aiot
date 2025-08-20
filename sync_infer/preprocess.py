import cv2
import numpy as np

def preprocess_from_cap(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Process frames from the camera stream and enqueue them.
    Args:
        frame (np.ndarray): Raw camera/video frame.
        width (int): Model input width.
        height (int): Model input height.
    Returns:
        np.ndarray: Preprocessed frame.
    """

    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = preprocess_fn(processed_frame, width, height)

    return processed_frame


def preprocess_fn(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image (np.ndarray): Input image.
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        np.ndarray: Preprocessed and padded image.
    """
    img_h, img_w, _ = image.shape[:3]
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)

    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    x_offset = (model_w - new_img_w) // 2
    y_offset = (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image

    return padded_image
