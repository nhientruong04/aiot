from itertools import count
from functools import partial
from multiprocessing import Process, Queue
from threading import Thread, Lock, Event
import queue
import numpy as np
from inference import infer
import threading

def main():
    request_map = dict()
    lock = Lock()
    stop_event = Event()

    requests = [np.squeeze(item).astype(np.float32) for item in np.random.randint(low=0, high=255, size=(10, 256, 256, 3))]
    request_queue = Queue()
    for item in requests:
        request_queue.put(item)
    # this sentinel only serves to ensure all requests is sent for processing before closing the send thread
    # hence could be removed in production phase
    request_queue.put(None)

    student_input_queue = Queue()
    student_output_queue = Queue()
    teacher_input_queue = Queue()
    teacher_output_queue = Queue()

    send_fn = partial(send, stop_event=stop_event, student_input_queue=student_input_queue, teacher_input_queue=teacher_input_queue)
    receive_fn = partial(receive, lock=lock, stop_event=stop_event, request_map=request_map)

    process_pool = [
        Process(target=infer, name="[Model Process] Student", args=("/home/nhien/aiot/data/student.hef", student_input_queue, student_output_queue)),
        Process(target=infer, name="[Model Process] Teacher", args=("/home/nhien/aiot/data/teacher.hef", teacher_input_queue, teacher_output_queue))
    ]

    thread_pool = [
        Thread(target=send_fn, name="[Send Thread]", args=(request_queue,)),
        Thread(target=receive_fn, name="[Receive Thread] Teacher", args=(teacher_output_queue,)),
        Thread(target=receive_fn, name="[Receive Thread] Student", args=(student_output_queue,)),
    ]

    for worker in process_pool:
        worker.start()

    for worker in thread_pool:
        worker.start()

    stop_event.set()

    for worker in thread_pool:
        worker.join()

    for worker in process_pool:
        print(f"Terminating {worker.name}...")
        worker.join()

    # print(request_map.keys())
    # print(request_map[0])

def send(request_queue, stop_event, student_input_queue, teacher_input_queue):
    counter = count(start=0)

    # while not stop_event:
    while True:
        input = request_queue.get(block=True)

        if (input is None):
            break

        request = {"id": next(counter), "input": input.copy()}

        student_input_queue.put(request.copy())
        teacher_input_queue.put(request.copy())

    student_input_queue.put(None)
    teacher_input_queue.put(None)

    print(f"{threading.current_thread().name} closed")

def receive(output_queue, lock, stop_event, request_map):
    thread_name = threading.current_thread().name
    model_name = thread_name[thread_name.index("]")+2:].lower()

    # while not stop_event:
    while True:
        output = output_queue.get(block=True)
        if output is None:
            break
        request_id = output["id"]

        with lock:
            if request_id not in request_map:
                value = {"teacher": None, "student": None}
                request_map[request_id] = value

            request_map[request_id][model_name] = output["output"]
    
    print(f"{threading.current_thread().name} closed")

if __name__ == "__main__":
    main()
