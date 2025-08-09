import numpy as np
from multiprocessing import Queue
from tqdm import tqdm
from loguru import logger
from typing import List
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType

# WARNING: This is a very very unoptimized piece
def teacher_normalization(teacher_hef_path: str, dataset_path: str):
    dataset = np.load(dataset_path)
    assert dataset.dtype == np.uint8, f'Expected uint8, got {dataset.dtype}'
    items: List = np.split(dataset, dataset.shape[0], axis=0)

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    with VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(teacher_hef_path)

        infer_model.input().set_format_type(FormatType.UINT8)
        infer_model.output().set_format_type(FormatType.FLOAT32)


        with infer_model.configure() as teacher:

            # teacher_mean
            output_means = list()
            logger.info("Calculating teacher's channels mean.")
            for item in tqdm(items):
                bindings = teacher.create_bindings()
                bindings.input().set_buffer(item)
                bindings.output().set_buffer(np.empty(infer_model.output().shape, dtype=np.float32))

                teacher.run([bindings], 10000)

                output = bindings.output().get_buffer()
                output_mean = np.mean(output, axis=(0,1))
                output_means.append(output_mean)

            channel_mean = np.mean(np.stack(output_means, axis=0), axis=0)
            channel_mean = channel_mean[None, None, :]
            print(channel_mean.shape)

            # teacher_std
            output_stds = list()
            logger.info('Calculating teacher\'s channels std.')
            for item in tqdm(items):
                bindings = teacher.create_bindings()
                bindings.input().set_buffer(item)
                bindings.output().set_buffer(np.empty(infer_model.output().shape, dtype=np.float32))

                teacher.run([bindings], 10000)

                output = bindings.output().get_buffer()
                distance = (output - channel_mean) ** 2
                distance_mean = np.mean(distance, axis=(0,1))
                output_stds.append(distance_mean)

            channel_std = np.mean(np.stack(output_stds, axis=0), axis=0)
            channel_std = channel_std[None, None, :]
            print(channel_std.shape)

    return channel_mean, channel_std

if __name__ == "__main__":
    teacher_normalization('../data/teacher.hef', '../data/v3_train_dataset.npy')
