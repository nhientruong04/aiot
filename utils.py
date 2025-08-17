import numpy as np
import os
from multiprocessing import Queue
from tqdm import tqdm
from typing import List
from common.infer_model import HailoInfer
from typing import Optional, Tuple
from datetime import datetime
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import pandas as pd

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

def map_normalization_noae(teacher: HailoInfer, student: HailoInfer, channel_mean: np.float32,
                      channel_std: np.float32, dataset_path: str):
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


class AD_Evaluation():
    def __init__(self, y_true: np.array, y_score: np.array, plot_dir: Optional[str] = None):
        self.y_true = y_true
        self.y_score = y_score
        plot_dir = plot_dir if plot_dir is not None else datetime.now().strftime('%Y-%m-%d-%H')
        self.plot_dir = os.path.join('.', 'runs', plot_dir)
        self.__post_init__()

    def __post_init__(self):
        self.__aggregate()
        os.makedirs(self.plot_dir, exist_ok=True)

    def __aggregate(self):
        # Calculate ROC
        self.fpr, self.tpr, thresholds = roc_curve(y_true=self.y_true, y_score=self.y_score)
        # Calculate AUC
        self.auc = roc_auc_score(y_true=self.y_true, y_score=self.y_score)

        # Find best threshold
        J = self.tpr - self.fpr
        self.ix = J.argmax()
        self.J_threshold = thresholds[self.ix]

        # Find Precision and Recall
        self.precision, self.recall, self.pr_thresholds = precision_recall_curve(y_true=self.y_true, y_score=self.y_score)
        self.ap = average_precision_score(y_true=self.y_true, y_score=self.y_score)

    def visualize_pr_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.recall, self.precision, lw=2, label=f'PR curve (AP = {self.ap:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)

        return fig

    def visualize_roc(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot ROC curve
        ax.plot(self.fpr, self.tpr, color='blue', lw=2, label=f'ROC curve (AUC = {self.auc:.3f})')
        ax.plot([0,1], [0,1], color='gray', linestyle='--', label='Random guess')

        # Mark optimal threshold
        ax.scatter(self.fpr[self.ix], self.tpr[self.ix], color='red', label=f'Best thr={self.J_threshold:.2f}')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        return fig

    def evaluate(self, show: bool = False) -> dict:
        roc_fig = self.visualize_roc()
        pr_fig = self.visualize_pr_curve()

        roc_fig.savefig(os.path.join(self.plot_dir, 'roc_curve.png'))
        pr_fig.savefig(os.path.join(self.plot_dir, 'precision_recall_curve.png'))

        if show:
            roc_fig.show()
            pr_fig.show()

        plt.close(roc_fig)
        plt.close(pr_fig)

        metrics = {
            'AUC': self.auc,
            'AP': self.ap,
            'Youden\'s J threshold': self.J_threshold
        }

        df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        print(df.to_string(index=False))

        return metrics

if __name__ == "__main__":
    evaluation_model = AD_Evaluation(np.random.randint(0, 2, 10), np.random.random(10))
    evaluation_model.evaluate()
