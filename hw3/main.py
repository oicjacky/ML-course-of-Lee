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
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocess import Preprocessor, ImgDataset
from model import Classifier, VGG16
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
    EXPONENTIAL_LR = config.getboolean('Model', 'exponential_lr')
    EXP_GAMMA = config.getfloat('Model', 'exponential_gamma')
    PREDICTION = config.get('File', 'prediction')
    CHECKPOINT_MODEL = config.get('File', 'checkpoint_model')


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
    model = Classifier()
    #model = VGG16(linear_batch_norm=False, linear_dropout=False, linear_dropout_rate=0.5)
    model.cuda()
    loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE) # Adam optimizer
    if EXPONENTIAL_LR:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=EXP_GAMMA, verbose=True)
    
    for epoch in range(EPOCH):
        print("[EPOCH] Now is {}".format(epoch))
        # if epoch > 1:
        #     print("Break manually.")
        #     break
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        best_acc = 0

        print("[TRAIN]")
        model.train() # 確保 model 是在 train model (開啟 Dropout 等...
        train_acc, train_loss = train_loop(train_loader, model, loss, optimizer, train_acc, train_loss)
        print("[VALIDATE]")
        model.eval()
        val_acc, val_loss = test_loop(val_loader, model, loss, val_acc, val_loss)

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, EPOCH, time.time()-epoch_start_time, \
             train_acc/len(train_set), train_loss/len(train_set), \
             val_acc/len(val_set), val_loss/len(val_set)))
        if val_acc/len(val_set) > best_acc:
            torch.save(model, os.path.join('.', CHECKPOINT_MODEL))
            print('saving model with acc {:.3f}'.format(val_acc/len(val_set)*100))
            early_stop = 0
        else:
            early_stop += 1
        if early_stop > 2:
                print(f'Early stopping with acc {val_acc/len(val_set)*100:.3f}!')
        if EXPONENTIAL_LR:
            scheduler.step()

    #TODO: use all train + valid data to re-train a model
    print("[TEST]")
    test_set = ImgDataset(test_x, transform=Preprocessor.test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print('loading checkpoint model...')
    model = torch.load(os.path.join('.', CHECKPOINT_MODEL))
    model.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.tolist()
    #將結果寫入 csv 檔
    with open(PREDICTION, 'w') as f:
        f.write('Id,Category\n')
        for i, y in  enumerate(prediction):
            f.write('{},{}\n'.format(i, y))

    print("Done")