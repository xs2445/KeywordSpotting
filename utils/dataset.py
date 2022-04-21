import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import glob
import os
import uuid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

DEFAULT_TRANSFORMS = T.Compose([
    # T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

class SpectrogramDataset(Dataset):
    def __init__(self, path, dataset_name, annotations, class_dict, transforms=None) -> None:
        """
        Build a dataset with annotations.
        annotations = [
            {
                "path": ,
                "class_index": ,
                "class_name": 
            },
            ...,
        ]\n
        class_dict = {
            "0": class_name0,
            ...,
        }
        - path: path of dataset
        - dataset_name: name of specific dataset
        - class_dict: label and class_name in dictionary
        - annotations: the annotation of this dataset
        - transforms: any transformation to the data
        """
        self. path = path
        self.class_dict = class_dict
        self.class_names = list(class_dict.values())
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.annotations = annotations
        # self._refresh()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        spec_img = np.load(annotation['path'])
        spec_img -= np.min(spec_img)
        spec_img /= np.max(spec_img)
        
        return torch.tensor(spec_img), annotation['class_index']

    def class_count(self, class_name):
        count = 0
        class_path = os.path.join(self.path, self.dataset_name, class_name)
        for img_name in os.listdir(class_path):
            if img_name.endswith('.npy'):
                count += 1

        return count

    def get_random_data(self):
        """
        get random data from dataset
        """
        random_class = random.choice(self.class_names)
        random_path = random.choice(
            os.listdir(
                os.path.join(self.path, self.dataset_name, random_class)))
        random_path = os.path.join(self.path, self.dataset_name, random_class, random_path)
        img_sample = np.load(random_path)

        return img_sample.squeeze()

    def grid_visualization(self, grid_dims=(5,5)):
        cols = []
        for _ in range(grid_dims[0]):
            row = []
            for _ in range(grid_dims[1]):
                row.append(self.get_random_data())
            cols.append(np.hstack(row))
        grid = np.vstack(cols)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis('off')
        plt.title(self.dataset_name)
        plt.show()

    @classmethod
    def make_dataset(cls, path, dataset_name, class_names=None):
        
        folder = os.path.join(path, dataset_name)

        annotations = []
        class_dict = {}

        class_index = 0

        if class_names == None:
            class_names = os.listdir(folder)

        for class_name in class_names:
            class_path = os.path.join(folder, class_name)
            class_dict[class_index] = class_name
            for spec_img in os.listdir(class_path):
                if spec_img.endswith('npy'):
                    annotations.append({
                        "path": os.path.join(class_path, spec_img),
                        "class_index": class_index,
                        "class_name": class_name
                    })
            class_index += 1

        return annotations, class_dict

    @classmethod
    def split_dataset(cls, path, dataset_name, split_rate=[0.8,0.1,0.1], class_names=None):
        """
        Make and split data set to train, test, validation sets

        - path: path of dataset
        - dataset_name: name of specific dataset
        - split_rate: rate to partition the whole dataset to train, test, val
        - class_names: specific classes to be loaded
        """

        annotations, class_dict = cls.make_dataset(path, dataset_name, class_names)

        end_train = int(len(annotations)*split_rate[0])
        end_test = end_train + int(len(annotations)*split_rate[1])

        random.shuffle(annotations)
        
        train_set = cls(path, dataset_name, annotations[:end_train], class_dict)
        test_set = cls(path, dataset_name, annotations[end_train:end_test], class_dict)
        val_set = cls(path, dataset_name, annotations[end_test:], class_dict)

        return train_set, test_set, val_set







