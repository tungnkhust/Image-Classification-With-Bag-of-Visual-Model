import os
from abc import abstractmethod, ABC
from src.utils import get_label_from_path, read_image
import numpy as np


class Generator(ABC):
    @abstractmethod
    def __len__(self):
        """
        get number of batches
        :return: (int) num of batches
        """
        pass

    @abstractmethod
    def __iter__(self, **kwargs):
        """
        Read image and return batch of image
        :param kwargs:
        :return:
        """
        pass


class DirGenerator(Generator):
    def __init__(self, img_dir, label2idx, size, batch_size=64, shuffle=False):
        """
        Read data from dir, and iterator return images and label
        :param img_dir:
        :param label2idx:
        :param size: (tuple) size of image after return
        :param batch_size:
        :param shuffle:
        """
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = size
        self.samples = []
        sub_dirs = os.listdir(img_dir)
        for sub_dir in sub_dirs:
            sub_dir = os.path.join(img_dir, sub_dir)
            filenames = os.listdir(sub_dir)
            for file in filenames:
                file_path = os.path.join(sub_dir, file)
                label = get_label_from_path(file_path, label2idx)
                self.samples.append((file_path, label))

    def __len__(self):
        return int(len(self.samples)/self.batch_size)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

        for i in range(0, len(self)-1):
            samples = self.samples[i*self.batch_size: (i+1)*self.batch_size]
            images = [read_image(s[0], size=self.size) for s in samples]

            labels = [s[1] for s in samples]
            yield images, labels
