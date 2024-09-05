import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CatImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        """
        :param image_dir:  image foder
        :param label_file:  label file  (txt file)
        :param transform:
        """
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform

        # 读取标签文件， 重构图像名和标签的映射关系
        self.image_labels = {}
        with open(label_file, "r") as f:
            for line in f:
                image_name, label = line.strip().split('\t')
                self.image_labels[image_name] = int(label)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        # 获取图像名和标签
        image_name = list(self.image_labels.keys())[idx]
        label = self.image_labels[image_name]

        # 加载图像
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            iamge = self.transform(image)

        return iamge, label