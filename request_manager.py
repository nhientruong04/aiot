from threading import Lock
import numpy as np
import cv2

class RequestManager:
    def __init__(self, request_processed_callback, teacher_mean, teacher_std, q_st_start=2.9, q_st_end=3.1):
        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std
        self.q_st_start = q_st_start
        self.q_st_end = q_st_end
        self.processed_callback = request_processed_callback
        # request map
        self._map = dict()
        self._lock = Lock()

    def push(self, request_id, output, model_name):
        assert model_name in ["teacher", "student"], f"The model_name should be either 'teacher' or 'student', not {model_name}"

        with self._lock:
            if request_id not in self._map.keys():
                self._map[request_id] = {"teacher": None, "student": None}

            # check for racing by threads
            # just in case 
            if self._map[request_id][model_name] is not None:
                raise Exception("Racing")
            self._map[request_id][model_name] = output

            request_result = self._map[request_id]
            if request_result["teacher"] is not None and request_result["student"] is not None:
                print(f"Start aggregating request with id {request_id}")
                del self._map[request_id]
                self._aggregate_result(request_result)

    def _aggregate_result(self, request_result):
        map_st = self._get_anomaly_map(teacher_output=request_result["teacher"],
                                             student_output=request_result["student"])

        map_st = np.squeeze(map_st, axis=2)
        map_st = np.pad(map_st, pad_width=(4, 4), constant_values=0)
        map_st = cv2.resize(map_st, (256, 256), interpolation=cv2.INTER_LINEAR)

    def _get_anomaly_map(self, teacher_output, student_output):
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std

        map_st = np.mean((teacher_output - student_output)
                        ** 2, axis=2, keepdims=True)
        map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)

        return map_st
