import numpy as np
from multiprocessing import Queue
from tqdm import tqdm
from typing import List
from common.infer_model import HailoInfer
from typing import Optional, Tuple

# WARNING: This is a very very unoptimized piece
def teacher_normalization(teacher: HailoInfer, dataset_path: str):
    dataset = np.load(dataset_path)
    assert dataset.dtype == np.uint8, f'Expected uint8, got {dataset.dtype}'
    items: List = np.split(dataset, dataset.shape[0], axis=0)

    # teacher_mean
    output_means = list()

    for item in tqdm(items, desc='Calculating teacher\'s channels mean.'):
        output = teacher.run_sync(item)
        output_mean = np.mean(output, axis=(0,1))
        output_means.append(output_mean)

    channel_mean = np.mean(np.stack(output_means, axis=0), axis=0)
    channel_mean = channel_mean[None, None, :]

    # teacher_std
    output_stds = list()

    for item in tqdm(items, desc='Calculating teacher\'s channels std.'):
        output = teacher.run_sync(item)
        distance = (output - channel_mean) ** 2
        distance_mean = np.mean(distance, axis=(0,1))
        output_stds.append(distance_mean)

    channel_std = np.mean(np.stack(output_stds, axis=0), axis=0)
    channel_std = channel_std[None, None, :]

    return channel_mean, channel_std

def map_normalization(teacher: HailoInfer, student: HailoInfer, channel_mean: np.float32,
                      channel_std: np.float32,  dataset_path: str):
    dataset = np.load(dataset_path)
    assert dataset.dtype == np.uint8, f'Expected uint8, got {dataset.dtype}'
    items: List = np.split(dataset, dataset.shape[0], axis=0)

    maps = list()

    for item in tqdm(items, desc='Calculating map normalization indices.'):
        teacher_output = teacher.run_sync(item)
        teacher_output = (teacher_output - channel_mean) / channel_std
        student_output = student.run_sync(item)
        map = np.mean((teacher_output - student_output)**2, axis=2, keepdims=True)
        maps.append(map)

    maps_st = np.concatenate(maps)
    q_st_start = np.quantile(maps_st, 0.9)
    q_st_end = np.quantile(maps_st, 0.995)

    return q_st_start, q_st_end

# channel_norm_file should be the training dataset and map_norm_file should be the validation dataset
def prepare_specs(teacher_hef: str, student_hef: str, channel_norm_file: str,
                  map_norm_file: str, save_file: Optional[str] = None) -> dict[str, np.array]:
    teacher = HailoInfer(teacher_hef, input_type='UINT8', output_type='FLOAT32')
    student = HailoInfer(student_hef, input_type='UINT8', output_type='FLOAT32')

    channel_mean, channel_std = teacher_normalization(teacher, '../data/v3_train_dataset.npy')
    q_st_start, q_st_end = map_normalization(teacher, student, channel_mean, channel_std, '../data/v3_train_dataset.npy')

    if save_file is not None:
        _, ext = save_file.split('.')
        file_path = None
        if ext == 'npz':
            file_path = save_file
        else:
            file_path = f'{save_file}.npz'

        print(f'Saving to {file_path}.')
        np.savez(file_path, channel_mean=channel_mean, channel_std=channel_std,
                 q_st_start=q_st_start, q_st_end=q_st_end)

    teacher.close()
    student.close()

    return {
        'channel_mean': channel_mean,
        'channel_std': channel_std,
        'q_st_start': q_st_start,
        'q_st_end': q_st_end
    }


if __name__ == "__main__":
    specs = prepare_specs('../data/teacher.hef', '../data/student.hef', 
                  '../data/v3_train_dataset.npy', '../data/v3_train_dataset.npy', 'specs.npz')

    print(specs['q_st_start'], specs['q_st_end'])
