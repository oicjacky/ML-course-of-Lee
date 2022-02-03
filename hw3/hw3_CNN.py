""" [Colab hw3_CNN.ipynb](https://colab.research.google.com/drive/1rDbT2_7ULwKUMO1oJ7w6oD20K5d75bBW#scrollTo=FsqMZlfj08vl)
Notes
=====
Dataset:
    在 PyTorch 中，我們可以利用 torch.utils.data 的 Dataset 及 DataLoader 來"包裝" data，使後續的 training 及 testing 更為方便。
    Dataset 需要 overload 兩個函數：__len__ 及 __getitem__
    __len__ 必須要回傳 dataset 的大小，而 __getitem__ 則定義了當程式利用 [ ] 取值時，dataset 應該要怎麼回傳資料。
    實際上我們並不會直接使用到這兩個函數，但是使用 DataLoader 在 enumerate Dataset 時會使用到，沒有實做的話會在程式運行階段出現 error。
"""
import os
import time
import numpy as np 
import pandas as pd
import cv2
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class Preprocessor:

    # training 要做 data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
        transforms.RandomRotation(15), # 隨機旋轉圖片
        transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    ])
    # testing 時不需做 data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
    ])

    @staticmethod
    def readfile(path, label):
        """ label: boolean, indicate whether to return value of y or not. """
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
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__() # same as: super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def train_loop(train_loader, model, loss, optimizer,
                train_acc, train_loss):
    size = len(train_loader.dataset)
    for batch_index, data in enumerate(train_loader):
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值
        
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
        if batch_index % 100 == 0:
            _loss, current = batch_loss.item(), batch_index * len(data[1])
            print(f"loss: {_loss:>7f}  [{current:>5d}/{size:>5d}]")
    return train_acc, train_loss

def test_loop(val_loader, model, loss, val_acc, val_loss):
    size = len(val_loader.dataset)
    with torch.no_grad():
        for batch_index, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
            if batch_index % 100 == 0:
                _loss, current = batch_loss.item(), batch_index * len(data[1])
                print(f"loss: {_loss:>7f}  [{current:>5d}/{size:>5d}]")
    return val_acc, val_loss


if __name__ == "__main__":
    
    DATA_DIR = r'E:\Download\dataset\food-11'
    _p = lambda p: os.path.join(DATA_DIR, p)
    BATCH_SIZE = 32  # 128 will lead insufficient memory 
    LEARNING_RATE = 0.001
    NUM_EPOCH = 30


    print("[Reading image] using opencv(cv2) read images into np.array")
    train_x, train_y = Preprocessor.readfile(_p("training"), True)
    val_x, val_y = Preprocessor.readfile(_p("validation"), True)
    test_x = Preprocessor.readfile(_p("testing"), False)
    print("Size of training data, validation data, testing data = {}, {}, {}".format(
        len(train_x), len(val_x), len(test_x)))
    

    print("[Preprocessing] create dataset and initialize dataloader")
    train_set = ImgDataset(train_x, train_y, Preprocessor.train_transform)
    val_set = ImgDataset(val_x, val_y, Preprocessor.test_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    
    print("[Modeling]")
    model = Classifier().cuda()
    loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE) # Adam optimizer
    
    for epoch in range(NUM_EPOCH):
        print("[EPOCH] Now is {}".format(epoch))
        if epoch > 5:
            print("Break manually.")
            break
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        print("[TRAIN]")
        model.train() # 確保 model 是在 train model (開啟 Dropout 等...
        train_acc, train_loss = train_loop(train_loader, model, loss, optimizer, train_acc, train_loss)
        print("[VALIDATE]")
        model.eval()
        val_acc, val_loss = test_loop(val_loader, model, loss, val_acc, val_loss)

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, NUM_EPOCH, time.time()-epoch_start_time, \
            train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

    print("Done")
    # try:
    # setting device on GPU if available, else CPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)
    # print()
    # ---------------------------------
    # ---- for epoch ... MAIN CODE ---- 
    # ----         PUT HERE        ---- 
    # ---------------------------------
    # except Exception as err:
    #     print(err)
    #     #Additional Info when using cuda
    #     if device.type == 'cuda':
    #         print(torch.cuda.get_device_name(0))
    #         print('Memory Usage:')
    #         print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    #         print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')