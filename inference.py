import numpy as np
import queue
from functools import partial
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
import multiprocessing
from multiprocessing import Queue, Process

# Optional: define a callback function that will run after the inference job is done
# The callback must have a keyword argument called "completion_info".
# That argument will be passed by the framework.
def post_infer_callback(completion_info, request_id, output_queue, bindings):
    if completion_info.exception:
        print(f"In callback of {multiprocessing.current_process().name}, something went wrong.")

    # TODO: post infer processing for final output
    output = bindings.output().get_buffer()
    output_queue.put({"id": request_id, "output": output})

timeout_ms = 1000


def infer(hef_path, input_queue, output_queue, should_use_multi_process_service=True):
    # Create a VDevice
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = "SHARED"
    if should_use_multi_process_service:
        params.multi_process_service = should_use_multi_process_service

    with VDevice(params) as vdevice:

        # Create an infer model from an HEF:
        infer_model = vdevice.create_infer_model(hef_path)

        infer_model.input().set_format_type(FormatType.FLOAT32)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        # Once the infer model is set, configure the infer model
        with infer_model.configure() as configured_infer_model:
            jobs = list()

            while True:
                request = input_queue.get(block=True)
                if request is None:
                    break

                request_id = request["id"]
                input = request["input"]

                # Create bindings for it and set buffers
                bindings = configured_infer_model.create_bindings()
                bindings.input().set_buffer(input)
                bindings.output().set_buffer(np.empty(infer_model.output().shape).astype(np.float32))

                configured_infer_model.wait_for_async_ready(timeout_ms=1000)

                job = configured_infer_model.run_async([bindings], partial(post_infer_callback, request_id=request_id,
                                                                           output_queue=output_queue, bindings=bindings))
                jobs.append(job)

            # Wait for the all jobs
            for job in jobs:
                job.wait(timeout_ms)

            print("All jobs completed")
            # close the receive thread
            output_queue.put(None)

if __name__ == "__main__":
    dataset = np.random.randint(low=0, high=255, size=(2, 256, 256, 3)).astype(np.float32)
    data = [np.squeeze(item) for item in np.split(dataset, 2)]

    request_queue = Queue()
    for item in data:
        request_queue.put(item)
    request_queue.put(None)

    student_input_queue = Queue()
    student_output_queue = Queue()
    teacher_input_queue = Queue()
    teacher_output_queue = Queue()

    process_pool = [
        Process(target=infer, name="[Model Process] Student", args=("/home/nhien/aiot/data/student.hef", student_input_queue, student_output_queue)),
        Process(target=infer, name="[Model Process] Teacher", args=("/home/nhien/aiot/data/teacher.hef", teacher_input_queue, teacher_output_queue))
    ]

    for worker in process_pool:
        worker.start()

    i = 0
    while i<2:
        input = request_queue.get()

        if input is None:
            break

        request = {"id": i, "input": input.copy()}

        student_input_queue.put(request.copy())
        teacher_input_queue.put(request.copy())

        i += 1

    student_input_queue.put(None)
    teacher_input_queue.put(None)

    print("Getting student outputs")
    while True:
        result = student_output_queue.get()
        if result is None:
            break
        print(result["output"])

    print("Getting teacher outputs")
    while True:
        result = teacher_output_queue.get()
        if result is None:
            break
        print(result["output"])

    for worker in process_pool:
        print(f"Terminating {worker.name}")
        worker.join()
