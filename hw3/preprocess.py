import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Preprocessor:
    r''' Perform data augmentation in training, which is not required in testing.
    '''
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
        transforms.RandomRotation(15), # 隨機旋轉圖片
        transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
    ])

    @staticmethod
    def readfile(path, label: bool):
        ''' Read image data of Food-11 by `cv2`.
        Args:
            label: boolean, indicate whether to return label (y) or not. '''
        image_dir = sorted(os.listdir(path))
        x = np.zeros((len(image_dir), 128, 128, 3), dtype= np.uint8)
        y = np.zeros((len(image_dir)), dtype= np.uint8)
        for i, file in enumerate(image_dir):
            img = cv2.imread(os.path.join(path, file))
            x[i, :, :] = cv2.resize(img, (128, 128))
            if label:
                y[i] = int(file.split("_")[0])  # filename's format: "[class]_[index].jpg"
        return (x, y) if label else x


class ImgDataset(Dataset):

    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            # label is required to be a LongTensor
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X