""" [Colab hw3_CNN.ipynb](https://colab.research.google.com/drive/1rDbT2_7ULwKUMO1oJ7w6oD20K5d75bBW#scrollTo=FsqMZlfj08vl)
Notes
=====
Dataset:
    在 PyTorch 中，我們可以利用 torch.utils.data 的 Dataset 及 DataLoader 來"包裝" data，使後續的 training 及 testing 更為方便。
    Dataset 需要 overload 兩個函數：__len__ 及 __getitem__
    __len__ 必須要回傳 dataset 的大小，而 __getitem__ 則定義了當程式利用 [ ] 取值時，dataset 應該要怎麼回傳資料。
    實際上我們並不會直接使用到這兩個函數，但是使用 DataLoader 在 enumerate Dataset 時會使用到，沒有實做的話會在程式運行階段出現 error。
"""
import configparser
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocess import Preprocessor, ImgDataset
from model import Classifier
from training import train_loop
from testing import test_loop

def main():
    pass


if __name__ == "__main__":
    
    #TODO: config.ini
    config = configparser.ConfigParser()
    config.read(r'./config.ini')

    DATA_DIR = config.get('Data', 'data_dir')
    BATCH_SIZE = config.getint('Model', 'batch_size')  # 128 will lead insufficient memory 
    LEARNING_RATE = config.getfloat('Model', 'learning_rate')
    EPOCH = config.getint('Model', 'epoch')
    PREDICTION = config.get('File', 'prediction')

    print("[Reading image] using opencv(cv2) read images into np.array")
    _p = lambda p: os.path.join(DATA_DIR, p)
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
    
    for epoch in range(EPOCH):
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
        val_acc, val_loss, prediction = test_loop(val_loader, model, loss, val_acc, val_loss)

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, EPOCH, time.time()-epoch_start_time, \
            train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
        
        #將結果寫入 csv 檔
        with open(PREDICTION, 'w') as f:
            f.write('Id,Category\n')
            for i, y in  enumerate(prediction):
                f.write('{},{}\n'.format(i, y))

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