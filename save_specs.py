from dataset import Dataset, DataLoader
from common.infer_model import HailoInfer
from efficientad import teacher_normalization, map_normalization
import numpy as np
import os


def main(save_path: str, teacher_hef: str, student_hef: str, autoencoder_hef: str,
         train_dir: str) -> None:
    if os.path.isfile(save_path):
        raise Exception('File exists!')

    teacher = HailoInfer(teacher_hef, input_type='UINT8', output_type='FLOAT32')
    student = HailoInfer(student_hef, input_type='UINT8', output_type='FLOAT32')
    autoencoder = HailoInfer(autoencoder_hef, input_type='UINT8', output_type='FLOAT32')

    model_height, model_width, _ = student.get_input_shape()

    train_set = Dataset(train_dir)
    train_loader = DataLoader(train_set, shuffle=True, resize=(model_height, model_width))

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(teacher, student, autoencoder,
                                                                   teacher_mean, teacher_std, train_loader)

    dir_name = os.path.dirname(save_path)
    # some intermediate directories
    if dir_name != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(save_path, teacher_mean=teacher_mean, teacher_std=teacher_std,
            q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end)

    teacher.close()
    student.close()
    autoencoder.close()


if __name__ == '__main__':
    teacher_hef = '../models/milkpack/teacher_max.hef'
    student_hef = '../models/milkpack/student_max.hef'
    autoencoder_hef = '../models/milkpack/autoencoder_max.hef'
    train_dir = '../data/milkpack/train'
    save_path = 'specs.npz'

    main(save_path, teacher_hef, student_hef, autoencoder_hef, train_dir)
