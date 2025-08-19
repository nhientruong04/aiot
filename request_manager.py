from threading import Lock
import numpy as np
import cv2
import tifffile
from multiprocessing import Queue
import os
from loguru import logger

OUTPUT_DIR = None
# OUTPUT_DIR = "output_ad"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

class RequestManager:
    # FIX: change all specs to rely on computed npz file instead of manual argument feed
    def __init__(self, request_processed_callback, teacher_input_queue: Queue,
                 student_input_queue: Queue, specs_path: str, input_size: int):
        self.processed_callback = request_processed_callback
        self.processed_ids = list()
        self.input_queues = [student_input_queue, teacher_input_queue]
        self.input_size = input_size
        # request map
        self._map = dict()
        self._lock = Lock()

        self._load_specs(specs_path)

    def _load_specs(self, specs_path: str) -> None:
        assert os.path.isfile(specs_path), f'File {specs_path} does not exist'
        specs = np.load(specs_path)

        # WARNING: fix teacher_mean and teacher_std naming consistency
        self.teacher_mean = specs.get('channel_mean', None)
        self.teacher_std = specs.get('channel_std', None)
        self.q_st_start = specs.get('q_st_start', None)
        self.q_st_end = specs.get('q_st_end', None)
        self.q_ae_start = specs.get('q_ae_start', None)
        self.q_ae_end = specs.get('q_ae_end', None)

    def id_exist(self, request_id: int) -> bool:
        if request_id in self._map.keys():
            return True

        if request_id in self.processed_ids:
            raise Exception("Error with id processing follow track ids.")

        return False

    def _preprocess(self, img: np.ndarray, target_size: int):
        """
        Pad shorter edge with 0 and resize.
        Args:
            img (np.ndarray): Input image in HWC format.
            target_size (int): Desired output size (square).
        Returns:
            np.ndarray: Square image of shape (target_size, target_size, C).
        """
        print(img.shape)
        h, w, c = img.shape
        max_dim = max(h, w)

        if h==target_size and w==target_size:
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
        resized = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        return resized

    def create_request(self, request_id: int, input: np.array) -> bool:
        """
        The process using this method should use RequestManager.id_exist() to check id validity
        before creating the request.
        """

        # NOTE: this should be removed because the id should be checked before calling this method
        # hence this is redundant
        #
        if request_id in self._map.keys():
            raise Exception("Duplicate request_id.")

        self._map[request_id] = {"teacher": None, "student": None}

        preprocessed_input = self._preprocess(input, self.input_size)
        for input_queue in self.input_queues:
            request = {"id": request_id,
                       "input": preprocessed_input.copy()
                       }
            input_queue.put(request.copy())


    def push(self, request_id, output, model_name):
        assert model_name in ["teacher", "student"], f"The model_name should be either 'teacher' or 'student', not {model_name}"

        if request_id not in self._map.keys():
            raise Exception(f"No request was created for request_id {request_id}")

        with self._lock:
            # check for racing by threads
            # just in case
            if self._map[request_id][model_name] is not None:
                raise Exception("Racing")
            self._map[request_id][model_name] = output

            request_result = self._map[request_id]
            if request_result["teacher"] is not None and request_result["student"] is not None:
                logger.info(f"Start aggregating request with id {request_id}")

                self.processed_ids.append(request_id)
                del self._map[request_id]
                self.__aggregate_result((request_id, request_result))

    def __aggregate_result(self, results: tuple):
        request_id, request_result = results

        map_st = self.__get_anomaly_map(teacher_output=request_result["teacher"],
                                             student_output=request_result["student"])

        map_st = np.squeeze(map_st, axis=2)
        map_st = np.pad(map_st, pad_width=(4, 4), constant_values=0)
        map_st = cv2.resize(map_st, (768, 768), interpolation=cv2.INTER_LINEAR)

        if OUTPUT_DIR is not None:
            file = os.path.join(
                OUTPUT_DIR, request_id + '.tiff')
            tifffile.imwrite(file, map_combined)
        # WARNING: need to implement the classify threshold
        # self.processed_callback()

    def __get_anomaly_map(self, teacher_output, student_output):
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std

        map_st = np.mean((teacher_output - student_output)
                        ** 2, axis=2, keepdims=True)
        map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)

        return map_st

    def close(self):
        for input_queue in self.input_queues:
            input_queue.put(None)

if __name__ == "__main__":
    def dummy_callback():
        pass

    student_input_queue = Queue()
    teacher_input_queue = Queue()

    request_manager = RequestManager(dummy_callback, teacher_input_queue, student_input_queue, 'specs.npz', 256)

    image = np.random.randint(0,255, (900, 700, 3), dtype=np.uint8)
    request_manager.create_request(0, image)

    request_manager.push(0, np.random.random((56, 56, 384)), "student")
    request_manager.push(0, np.random.random((56, 56, 384)), "teacher")

    print(request_manager.processed_ids)
