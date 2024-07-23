import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset




class CIFAR10Dataset(Dataset):
    @staticmethod
    def __unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    @staticmethod
    def __Xbatch_split(batch_file : np.ndarray) -> np.ndarray:
        """
        data -- a 10000x3072 numpy array of uint8s. 
        Each row of the array stores a 32x32 colour image. The first 1024 entries 
        contain the red channel values, the next 1024 the green, and the final 1024 the blue.
        The image is stored in row-major order,
        so that the first 32 entries of the array are the red channel values of the first row of the image.
        """
        assert batch_file.shape == (10000, 3072)
        out = batch_file.reshape(10000, 3, 32, 32)
        return out

    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """

        train_files = ['data_batch_' + str(i) for i in range(1, 6)]
        test_files = ['test_batch']

        if train:
            infiles = train_files
        else:
            infiles = test_files
        
        data_arrays = [self.__unpickle(os.path.join(base_folder, file)) for file in infiles]
        Xdata_arrays = [self.__Xbatch_split(batch_file[b'data']) for batch_file in data_arrays]
        ydata_arrays = [batch_file[b'labels'] for batch_file in data_arrays]

        self.X = np.concatenate(Xdata_arrays, axis = 0)
        self.X = self.X.astype(np.float32)
        self.X /= 255.
        self.y = np.concatenate(ydata_arrays, axis = 0)

        assert len(self.X) == len(self.y)
            


    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        assert index < len(self.X) and self.X[index].shape == (3, 32, 32)

        return self.X[index], self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
