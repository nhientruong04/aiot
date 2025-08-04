import time
import os
import datetime
import cv2
from itertools import count
from functools import partial
from multiprocessing import Process, Queue
from threading import Thread, Event
import queue
import numpy as np
from inference import infer
from request_manager import RequestManager
import threading

teacher_mean = 1
teacher_std = 1.2929

def main():
    stop_event = Event()
    request_manager = RequestManager(info_callback, teacher_mean, teacher_std)

    # requests = [np.squeeze(item).astype(np.float32) for item in np.random.randint(low=0, high=255, size=(3, 256, 256, 3))]
    # request_queue = Queue()
    # for item in requests:
    #     request_queue.put(item)
    #
    # request_queue.put(None)
    # this sentinel only serves to ensure all requests is sent for processing before closing the send thread
    # hence could be removed in production phase

    # benchmark
    request_queue = Queue()
    request_num = 0
    root_dir = "/home/nhien/aiot/data/images"

    for image_path in os.listdir(root_dir):
        image = cv2.imread(os.path.join(root_dir, image_path))
        image = cv2.resize(image, (256, 256)).astype(np.float32)
        request_queue.put(image)
        request_num += 1

    request_queue.put(None)
    # end benchmark

    student_input_queue = Queue()
    student_output_queue = Queue()
    teacher_input_queue = Queue()
    teacher_output_queue = Queue()

    send_fn = partial(send, stop_event=stop_event, student_input_queue=student_input_queue, teacher_input_queue=teacher_input_queue)
    receive_fn = partial(receive, stop_event=stop_event, request_manager=request_manager)

    process_pool = [
        Process(target=infer, name="[Model Process] Student", args=("/home/nhien/aiot/data/student.hef", student_input_queue, student_output_queue)),
        Process(target=infer, name="[Model Process] Teacher", args=("/home/nhien/aiot/data/teacher.hef", teacher_input_queue, teacher_output_queue))
    ]

    thread_pool = [
        Thread(target=send_fn, name="[Send Thread]", args=(request_queue,)),
        Thread(target=receive_fn, name="[Receive Thread] Teacher", args=(teacher_output_queue,)),
        Thread(target=receive_fn, name="[Receive Thread] Student", args=(student_output_queue,)),
    ]

    print(f"Starting all processes and threads at {datetime.time()}...")
    start_time = time.time()

    for worker in process_pool:
        worker.start()

    for worker in thread_pool:
        worker.start()

    stop_event.set()

    for worker in thread_pool:
        worker.join()

    for worker in process_pool:
        worker.join()

    duration = time.time() - start_time
    print(f"System ended in {duration:.2f} seconds.")
    # benchmark
    print(f"Processed on average {request_num/duration:.2f} requests/sec.")
    # end benchmark

def send(request_queue, stop_event, student_input_queue, teacher_input_queue):
    counter = count(start=0)

    # while not stop_event:
    while True:
        input = request_queue.get(block=True)

        if input is None:
            break

        request = {"id": next(counter), "input": input.copy()}

        student_input_queue.put(request.copy())
        teacher_input_queue.put(request.copy())

    student_input_queue.put(None)
    teacher_input_queue.put(None)

    print(f"{threading.current_thread().name} closed")

def receive(output_queue, stop_event, request_manager):
    thread_name = threading.current_thread().name
    model_name = thread_name[thread_name.index("]")+2:].lower()

    # while not stop_event:
    while True:
        output = output_queue.get(block=True)
        if output is None:
            break

        request_id = output["id"]
        output = output["output"]

        request_manager.push(request_id, output, model_name)

    print(f"{threading.current_thread().name} closed")

def info_callback():
    pass

if __name__ == "__main__":
    main()
