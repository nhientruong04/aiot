from threading import Lock, Event
from common.infer_model import HailoInfer

import os
from typing import List, Optional
import cv2
import numpy as np

class ADModel:
    def __init__(self, teacher_hef: str, student_hef: str, autoencoder_hef: str,
                 specs_path: str, output_dir: Optional[str] = None):
        assert os.path.isfile(specs_path), f'File {specs_path} does not exist'
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'ad_model')
            os.makedirs(self.output_dir)

        self.teacher = HailoInfer(teacher_hef)
        self.student = HailoInfer(student_hef)
        self.autoencoder = HailoInfer(autoencoder_hef)

        model_height, model_width, _ = self.teacher.get_input_shape()
        self.target_size = max(model_height, model_width)

        self._load_specs(specs_path)

        self.counter = 0

    def _load_specs(self, specs_path: str) -> None:
        specs = np.load(specs_path)

        # WARNING: fix teacher_mean and teacher_std naming consistency
        self.teacher_mean = specs.get('channel_mean') or specs.get('teacher_mean')
        self.teacher_std = specs.get('channel_std') or specs.get('teacher_std')
        assert self.teacher_mean is not None and self.teacher_std is not None

        self.q_st_start = specs.get('q_st_start', None)
        self.q_st_end = specs.get('q_st_end', None)
        self.q_ae_start = specs.get('q_ae_start', None)
        self.q_ae_end = specs.get('q_ae_end', None)

        self.threshold = specs.get('threshold')

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Pad shorter edge with 0 and resize.
        Args:
            img (np.ndarray): Input image in HWC format.
            target_size (int): Desired output size (square).
        Returns:
            np.ndarray: Square image of shape (target_size, target_size, C).
        """
        print(img.shape)
        # NOTE: sync this with the yolo pipeline or add the conversion layer to model script
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        max_dim = max(h, w)

        if h==self.target_size and w==self.target_size:
            if img.dtype != np.uint8:
                return np.array(img, dtype=np.uint8)

            return img

        # Create a square canvas filled with pad_value
        canvas = np.zeros((max_dim, max_dim, c), dtype=np.uint8)

        # Compute top-left corner for centering
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2

        # Paste original image
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

        # Resize to target square
        resized = cv2.resize(canvas, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        return resized


    def infer(self, inputs: List[np.ndarray]) -> List[str]:
        ret = list()

        for input in inputs:
            preprocessed_input = self._preprocess(input)
            map_combined = self._predict(preprocessed_input)

            if self.output_dir is not None:
                file = os.path.join(self.output_dir, self.counter + '.tiff')
                tifffile.imwrite(file, map_combined)

            y_score = np.max(map_combined)

            if y_score >= self.threshold:
                ret.append('defect')
            else:
                ret.append('good')

            self.counter += 1

        return ret

    def _predict(self, preprocessed_input: np.ndarray) -> np.ndarray:
        teacher_output = self.teacher.run_sync(preprocessed_input)
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
        student_output = self.student.run_sync(preprocessed_input)
        autoencoder_output = self.autoencoder.run_sync(preprocessed_input)

        map_st = np.mean((teacher_output - student_output[:, :, :384])**2, axis=2, keepdims=True)
        map_ae = np.mean((teacher_output - student_output[:, :, 384:])**2, axis=2, keepdims=True)

        if self.q_st_start is not None:
            map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
        if self.q_ae_start is not None:
            map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
        map_combined = 0.5 * map_st + 0.5 * map_ae

        return map_combined

    def close(self):
        self.teacher.close()
        self.student.close()
        self.autoencoder.close()

if __name__ == '__main__':
    from loguru import logger
    teacher_hef = '../models/milkpack/teacher_max.hef'
    student_hef = '../models/milkpack/student_max.hef'
    autoencoder_hef = '../models/milkpack/autoencoder_max.hef'
    specs_path = 'specs.npz'

    ad_model = ADModel(teacher_hef, student_hef, autoencoder_hef, specs_path)

    try:
        inputs = np.random.randint(0, 255, (16, 600, 545, 3), dtype=np.uint8)
        inputs = [np.squeeze(input) for input in np.split(inputs, 16)]
        outputs = ad_model.infer(inputs)
        print(outputs)
    except Exception as e:
        logger.error(f'Error occured during execution, message: {e}')
        raise
    finally:
        ad_model.close()

