from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.size = 0
        self.X = None
        self.y = None
        self.transforms = transforms
        with gzip.open(image_filename, "rb") as image_file, gzip.open(label_filename, "rb") as label_file:
            # Read and unpack the magic number from the image file
            magic, num_images = struct.unpack(">II", image_file.read(8))
            #print(f"Magic number: {magic}, Number of images: {num_images}")
            # Read and unpack the dimensions
            rows, cols = struct.unpack(">II", image_file.read(8))
            #print(f"Image dimensions: {rows}x{cols}")
            dim = rows * cols
            X = np.ndarray((num_images, dim), dtype=np.float32)
            total_bytes = num_images * dim
            all_bytes = image_file.read(total_bytes)
            all_colors = np.frombuffer(all_bytes, dtype=np.uint8).reshape((num_images, dim))
            X = all_colors.astype(np.float32) / 255.0

            # Similarly, read the magic number and number of items from the label file
            label_magic, num_labels = struct.unpack(">II", label_file.read(8))
            #print(f"Label file magic number: {label_magic}, Number of labels: {num_labels}")
            assert num_images == num_labels
            self.size = num_labels
            y = np.ndarray(num_labels, dtype=np.uint8)
            total_bytes = num_labels
            all_bytes = label_file.read(total_bytes)
            y = np.frombuffer(all_bytes, dtype=np.uint8)

            self.X = X
            self.y = y

    def __getitem__(self, index) -> object:
        X = self.X[index]
        y = self.y[index]
        if not self.transforms is None:
            for transform in self.transforms:
                X = np.reshape(X, (28, 28, 1))
                X = transform(X)
                X = np.reshape(X, (784,))
        return X, y

    def __len__(self) -> int:
        return self.size