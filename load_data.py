import torch
import idx2numpy
import numpy as np

from typing import Any
from collections import OrderedDict
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, images_path, labels_path, transforms=None, set_float=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transforms = transforms
        self.set_float = set_float
        self.images_array = idx2numpy.convert_from_file(self.images_path)
        self.labels_array = idx2numpy.convert_from_file(self.labels_path)

    def __len__(self) -> int:
        return len(self.images_array)
    
    def __getitem__(self, index) -> tuple:
        image = np.array(self.images_array[index])
        label = np.array(self.labels_array[index])
        
        image = image / 255.0
        if self.transforms:
            image = self.transforms(image)

        if self.set_float:
            image = image.to(torch.float)

        return (image, label)

    def stats(self) -> tuple:
        label_counts = OrderedDict()
        label_percentages = OrderedDict()

        for label in self.labels_array:
            if label not in label_counts:
                label_counts[label] = 1.0
            else:
                label_counts[label] += 1.0

        for label in label_counts:
            label_percentages[label] = label_counts[label] / len(self.labels_array)

        label_counts = OrderedDict(sorted(label_counts.items(), key=lambda t: t[0]))
        label_percentages = OrderedDict(sorted(label_percentages.items(), key=lambda t: t[0]))

        self.label_counts = label_counts
        self.label_percentages = label_percentages

        return (label_counts, label_percentages)

    def print_stats(self):
        print("Label counts")
        for label in self.label_counts:
            print(f"\t{label}\t{self.label_counts[label]}")

        print()

        print("Label percentages")
        for label in self.label_percentages:
            print(f"\t{label}\t{self.label_percentages[label]:.6f}")
