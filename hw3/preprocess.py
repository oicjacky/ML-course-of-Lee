import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class Preprocessor:
    r''' Perform data augmentation in training, which is not required in testing.
    '''
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
        # transforms.RandomResizedCrop(96), # 隨機擷取約75%圖片區域(96/128)
        transforms.RandomCrop(size=(128, 128)), # 隨機擷取圖片某區域
        # transforms.RandomAffine(degrees=30, translate=(0.1, 0.3), scale=(0.5, 0.75)),
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.RandomRotation(30), # 隨機旋轉圖片
        transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        transforms.Normalize(mean=[0.3438, 0.4511, 0.5551], std=[0.2811, 0.2740, 0.2711]), #NOTE: default, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3438, 0.4511, 0.5551], std=[0.2811, 0.2740, 0.2711]), #NOTE: default, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ])

    @staticmethod
    def readfile(path, label: bool, return_path: bool= False):
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
        if label:
            if not return_path:
                return x, y
            else:
                return x, y, image_dir
        else:
            return x


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


if __name__ == "__main__":

    DATA_DIR = r'E:\Download\dataset\food-11'
    
    print("[Reading image] using opencv(cv2) read images into np.array")
    _p = lambda p: os.path.join(DATA_DIR, p)
    train_x, train_y = Preprocessor.readfile(_p("training"), True)
    val_x, val_y = Preprocessor.readfile(_p("validation"), True)
    test_x = Preprocessor.readfile(_p("testing"), False)
    eval_x, eval_y = Preprocessor.readfile(_p("evaluation"), True)
    print("Size of training data, validation data, testing data = {}, {}, {}".format(
        len(train_x), len(val_x), len(eval_x)))

    train_set = ImgDataset(train_x, train_y, Preprocessor.train_transform)
    val_set = ImgDataset(val_x, val_y, Preprocessor.test_transform)
    eval_set = ImgDataset(eval_x, eval_y, Preprocessor.test_transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=32, shuffle=False)
    
    def online_mean_and_sd(loader):
        """Compute the mean and sd in an online fashion:
            Var[x] = E[X^2] - E^2[X]
            
        Reference: 
            1. [How do they know mean and std, the input value of transforms.Normalize](https://stackoverflow.com/questions/57532661/how-do-they-know-mean-and-std-the-input-value-of-transforms-normalize)
            2. [About Normalization using pre-trained vgg16 networks](https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9?u=kharshit)
        """
        current_total = 0
        fst_moment, snd_moment = torch.empty(3), torch.empty(3)

        for data in loader:
            #NOTE: data consist of X, Y
            data, _ = data
            b, c, h, w = data.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(data, dim=[0, 2, 3])
            sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
            fst_moment = (current_total * fst_moment + sum_) / (current_total + nb_pixels)
            snd_moment = (current_total * snd_moment + sum_of_square) / (current_total + nb_pixels)
            current_total += nb_pixels
        return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    mean, std = online_mean_and_sd(train_loader)
    print('The mean and std used by `transforms.Normalize` is', mean, 'and', std)

    import pdb; pdb.set_trace()