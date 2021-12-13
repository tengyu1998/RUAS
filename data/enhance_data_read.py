import numpy as np
import torch
import torch.utils.data
# import config
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os

class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = glob(self.low_img_dir)

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        if self.task == 'test':
            img_name = self.train_low_data_names[index].split('\\')[-1]
            return torch.from_numpy(low), img_name

        return torch.from_numpy(low)

    def __len__(self):
        return self.count