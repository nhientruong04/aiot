from threading import Lock

class RequestManager:
    def __init__(self, request_processed_callback, teacher_mean, teacher_std):
        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std
        self.processed_callback = request_processed_callback
        # request map
        self._map = dict()
        self._lock = Lock()

    def push(self, request_id, output, model_name):
        assert model_name in ["teacher", "student"], f"The model_name should be either 'teacher' or 'student', not {model_name}"

        if request_id not in self._map.keys():
            self._map[request_id] = {"teacher": None, "student": None}

        self._map[request_id][model_name] = output

        request_result = self._map[request_id]
        if request_result["teacher"] is not None and request_result["student"] is not None:
            del self._map[request_id]
            self._aggregate_result(request_result)

    def _aggregate_result(request_result):
        print(request_result)
