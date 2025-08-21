import numpy as np
import os
from multiprocessing import Queue
from tqdm import tqdm
from typing import List
from common.infer_model import HailoInfer
from typing import Optional, Tuple
from dataset import Dataset, DataLoader
from PIL import Image
from utils import AD_Evaluation
import tifffile


def test(test_loader: DataLoader, teacher: HailoInfer, student: HailoInfer, autoencoder: HailoInfer,
         teacher_mean: np.float32, teacher_std: np.float32, q_st_start: Optional[np.array] = None,
         q_st_end: Optional[np.array] = None, q_ae_start: Optional[np.array] = None,
         q_ae_end: Optional[np.array] = None, test_output_dir: Optional[str] = None,
         plot_dir: Optional[str] = None, desc='Running inference'):
    assert test_loader.batch_size == 1

    y_true = []
    y_score = []

    for images, targets, paths in tqdm(test_loader, desc=desc):
        image, target, path = images[0], targets[0], paths[0]
        with Image.open(path) as img:
            orig_width, orig_height = img.width, img.height

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)

        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, target)):
                os.makedirs(os.path.join(test_output_dir, target))
            file = os.path.join(
                test_output_dir, target, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if target == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

    evaluation_model = AD_Evaluation(y_true=y_true, y_score=y_score, plot_dir=plot_dir)
    metrics = evaluation_model.evaluate()

    return metrics


def map_normalization(teacher: HailoInfer, student: HailoInfer, autoencoder: HailoInfer,
                      teacher_mean: np.float32, teacher_std: np.float32, loader: DataLoader):
    """
    dataset.DataLoader loader: dataloader for train or val split. The original code uses val split.
    """
    assert loader.batch_size == 1

    maps_st = list()
    maps_ae = list()

    for images, _, _ in tqdm(loader, desc='Calculating map normalization indices'):
        map_combined, map_st, map_ae = predict(images[0], teacher, student, autoencoder,
                                               teacher_mean, teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)

    maps_st = np.concatenate(maps_st)
    maps_ae = np.concatenate(maps_ae)

    q_st_start = np.quantile(maps_st, 0.9)
    q_st_end = np.quantile(maps_st, 0.995)
    q_ae_start = np.quantile(maps_ae, 0.9)
    q_ae_end = np.quantile(maps_ae, 0.995)

    return q_st_start, q_st_end, q_ae_start, q_ae_end


def teacher_normalization(teacher: HailoInfer, train_loader: DataLoader):
    assert train_loader.batch_size == 1

    # teacher_mean
    output_means = list()

    for images, _, _ in tqdm(train_loader, desc='Calculating teacher\'s channels mean'):
        output = teacher.run_sync(images[0])
        output_mean = np.mean(output, axis=(0,1))
        output_means.append(output_mean)

    channel_mean = np.mean(np.stack(output_means, axis=0), axis=0)
    channel_mean = channel_mean[None, None, :]

    # teacher_std
    output_stds = list()

    for images, _, _ in tqdm(train_loader, desc='Calculating teacher\'s channels std'):
        output = teacher.run_sync(images[0])
        distance = (output - channel_mean) ** 2
        distance_mean = np.mean(distance, axis=(0,1))
        output_stds.append(distance_mean)

    channel_std = np.mean(np.stack(output_stds, axis=0), axis=0)
    channel_std = channel_std[None, None, :]

    return channel_mean, channel_std


def predict(image: np.array, teacher: HailoInfer, student: HailoInfer,
            autoencoder: HailoInfer, teacher_mean: np.array, teacher_std: np.array,
            q_st_start: Optional[np.array] = None, q_st_end: Optional[np.array] = None,
            q_ae_start: Optional[np.array] = None, q_ae_end: Optional[np.array] = None):

    teacher_output = teacher.run_sync(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student.run_sync(image)
    autoencoder_output = autoencoder.run_sync(image)

    # WARNING: this could be a potential error if fc layer change number of output channels
    map_st = np.mean((teacher_output - student_output[:, :, :384])**2, axis=2, keepdims=True)
    map_ae = np.mean((teacher_output - student_output[:, :, 384:])**2, axis=2, keepdims=True)

    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae

    return map_combined, map_st, map_ae

if __name__ == "__main__":
    teacher_hef = '../models/milkpack/teacher_max.hef'
    student_hef = '../models/milkpack/student_max.hef'
    autoencoder_hef = '../models/milkpack/autoencoder_max.hef'
    train_dir = '/home/nhien/aiot/data/milkpack/train'
    test_dir = '/home/nhien/aiot/data/milkpack/test'

    teacher = HailoInfer(teacher_hef, input_type='UINT8', output_type='FLOAT32')
    student = HailoInfer(student_hef, input_type='UINT8', output_type='FLOAT32')
    autoencoder = HailoInfer(autoencoder_hef, input_type='UINT8', output_type='FLOAT32')

    model_height, model_width, _ = student.get_input_shape()
    size = (model_height, model_width)

    test_set = Dataset(test_dir)
    test_loader = DataLoader(test_set, shuffle=True, resize=size)

    train_set = Dataset(train_dir)
    train_loader = DataLoader(train_set, shuffle=True, resize=size)

    specs_file = np.load('./specs.npz')
    specs = {key: specs_file[key] for key in specs_file.files}

    metrics = test(test_loader, teacher, student, autoencoder, test_output_dir='runs/milkpack_outputs', **specs)

    teacher.close()
    student.close()
    autoencoder.close()
