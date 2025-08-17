import os
import numpy as np
import cv2
import random
from typing import Tuple, Optional

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []
        self.mode = None  # "npy" or "folder"

        # Case 1: npy file
        if data_path.endswith(".npy"):
            self.mode = "npy"
            self.data = np.load(data_path)
            self.length = len(self.data)

        # Case 2: folder structure with class subdirs
        elif os.path.isdir(data_path):
            self.mode = "folder"
            classes = sorted(os.listdir(data_path))

            for cls in classes:
                cls_path = os.path.join(data_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                for fname in os.listdir(cls_path):
                    fpath = os.path.join(cls_path, fname)
                    if os.path.isfile(fpath):
                        self.samples.append((fpath, cls))  # store path + label name
            self.length = len(self.samples)
        else:
            raise ValueError("Path must be either a .npy file or a directory with subfolders")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == "npy":
            img = self.data[idx]
            return img, None, None

        elif self.mode == "folder":
            fpath, cls = self.samples[idx]
            # load image with cv2
            img = cv2.imread(fpath)
            if img is None:
                raise ValueError(f"Could not read image: {fpath}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, cls, fpath


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False,
                 resize: Optional[Tuple] = None) -> Tuple[np.array, Optional[str], Optional[str]]:
        """
        resize: tuple (H, W) to resize all images, or None to keep original size

        return: 
            tuple of (numpy_image, target, path) for dataset created with folder path, or
            tuple of (numpy_image, None, None) for dataset created with .npy file
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.resize = resize

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        batch_imgs, batch_targets, batch_paths = [], [], []
        for idx in batch_indices:
            img, target, path = self.dataset[idx]

            if self.resize is not None:
                H, W = self.resize
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

            batch_imgs.append(img)
            batch_targets.append(target)
            batch_paths.append(path)


        # Convert to numpy array only if images have same shape
        try:
            batch_imgs = np.stack(batch_imgs, axis=0)
        except:
            raise Exception("Images are not the same size.")

        return batch_imgs, batch_targets, batch_paths

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))


if __name__ == '__main__':
    root_dir = '/home/nhien/aiot/data/bottle/test'
    size = (256, 256)

    dataset = Dataset(root_dir)
    dataloader = DataLoader(dataset, shuffle=True, resize=size)

    for i, item in enumerate(dataloader):
        image, target, path = item
        print(target, path)
        print(image.dtype)
