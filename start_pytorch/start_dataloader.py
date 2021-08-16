''' [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

PyTorch provides two data primitives:
    |_ torch.utils.data
        |_ dataset
        |_ dataloader
    
It also proide a number of pre-loaded datasets, 
    - [Image](https://pytorch.org/vision/stable/datasets.html)
    - [Text](https://pytorch.org/text/stable/datasets.html)
    - [Audio](https://pytorch.org/audio/stable/datasets.html)
'''
import torch
from torch.utils.data import dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image


LABELS_MAP = { 0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot" }

def loading_a_dataset():
    ''' Fashion-MNIST: A dataset of Zalando's article images 
    consisting of 60,000 training and 10,000 test examples. 
    Each comprises a 28x28 grayscale image and an associated
    label from one of 10 classes.
    '''
    training_data = datasets.FashionMNIST(root='data', train=True,
                                          download=True, transform=ToTensor())
    testing_data = datasets.FashionMNIST(root='data', train=False,
                                          download=True, transform=ToTensor())
    return training_data, testing_data


def visualizing_data(labels_map, training_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


class CustomImageDataset(dataset.Dataset):
    ''' a custom dataset must implement:
    1. __init__  2. __len__  3. __getitem__

    Parameters:
    ===========
    img_dir: the directory where FashionMNIST images are stored
    annotaions_file: file path of CSV file that the label of data is stored 
    '''
    def __init__(self, annotations_file, img_dir,
                 transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # like: tshirt1.jpg, 0 \n ankleboot999.jpg, 9
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        'the number of samples in our dataset.'
        return len(self.img_labels)

    def __getitem__(self, idx):
        ''' Based on `idx`, get image by `read_image` and retrieve the corresponding
        label from the `self.img_labels`.
        '''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':

    training_data, testing_data = loading_a_dataset()

    visualizing_data(LABELS_MAP, training_data)


    print('\n## Preparing data for training with DataLoaders')
    from torch.utils.data.dataloader import DataLoader
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)


    print('''\n## display image and label
    `DataLoader` will iterate through the dataset, each iteration returns
    a batch of data. After we iterate over all batches the data is shuffled.''')
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")