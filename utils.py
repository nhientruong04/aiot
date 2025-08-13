import numpy as np
from multiprocessing import Queue
from tqdm import tqdm
from typing import List
from common.infer_model import HailoInfer
from typing import Optional

# WARNING: This is a very very unoptimized piece
def teacher_normalization(teacher_hef_path: str, dataset_path: str,
                          save_path: Optional[str] = None):
    dataset = np.load(dataset_path)
    assert dataset.dtype == np.uint8, f'Expected uint8, got {dataset.dtype}'
    items: List = np.split(dataset, dataset.shape[0], axis=0)

    teacher = HailoInfer('../data/teacher.hef', input_type='UINT8', output_type='FLOAT32')

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

    teacher.close()

    if save_path is not None:
        _, ext = save_path.split('.')
        file_path = None
        if ext == 'npz':
            file_path = save_path
        else:
            file_path = f'{save_path}.npz'

        print(f'Saving to {file_path}.')
        np.savez(file_path, channel_mean=channel_mean, channel_std=channel_std)

    return channel_mean, channel_std

if __name__ == "__main__":
    channel_mean, channel_std = teacher_normalization('../data/teacher.hef', '../data/v3_train_dataset.npy',
                                                      'teacher_norm.npz')
