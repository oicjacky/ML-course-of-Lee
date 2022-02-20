import configparser
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from preprocess import Preprocessor, ImgDataset
from model import Classifier, VGG16
from training import train_loop
from testing import test_loop
from logger import LOGGER


if __name__ == "__main__":
    #TODO: config.ini
    config = configparser.ConfigParser()
    config.read(r'./config_final.ini')

    DATA_DIR = config.get('Data', 'data_dir')
    BATCH_SIZE = config.getint('Model', 'batch_size')  # 128 will lead insufficient memory 
    LEARNING_RATE = config.getfloat('Model', 'learning_rate')
    EPOCH = config.getint('Model', 'epoch')
    EXPONENTIAL_LR = config.getboolean('Model', 'exponential_lr')
    EXP_GAMMA = config.getfloat('Model', 'exponential_gamma')
    PREDICTION = config.get('File', 'prediction')
    CHECKPOINT_MODEL = config.get('File', 'checkpoint_model')

    LOGGER.info("[Reading image] using opencv(cv2) read images into np.array")
    _p = lambda p: os.path.join(DATA_DIR, p)
    train_x, train_y = Preprocessor.readfile(_p("training"), True)
    val_x, val_y = Preprocessor.readfile(_p("validation"), True)
    LOGGER.info("Size of training data, validation data = {}, {}".format(len(train_x), len(val_x)))
    train_val_x = np.concatenate((train_x, val_x), axis=0)
    train_val_y = np.concatenate((train_y, val_y), axis=0)
    LOGGER.info("[Preprocessing] create dataset and initialize dataloader")
    train_val_set = ImgDataset(train_val_x, train_val_y, Preprocessor.train_transform)
    train_val_loader = DataLoader(train_val_set, batch_size=BATCH_SIZE, shuffle=True)
    
    LOGGER.info("[Modeling]")
    model = VGG16(linear_batch_norm=True, linear_dropout=True, linear_dropout_rate=0.5)
    model.cuda()
    loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE) # Adam optimizer
    if EXPONENTIAL_LR:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=EXP_GAMMA, verbose=True)
    
    best_acc = 0
    for epoch in range(EPOCH):
        LOGGER.info("[EPOCH] Now is {}".format(epoch))
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
    
        LOGGER.info("[TRAIN]")
        model.train() # 確保 model 是在 train model (開啟 Dropout 等...
        train_acc, train_loss = train_loop(train_val_loader, model, loss, optimizer, train_acc, train_loss)
        
        LOGGER.info('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
            (epoch + 1, EPOCH, time.time()-epoch_start_time, \
             train_acc/len(train_val_set), train_loss/len(train_val_set)))

        if train_acc > best_acc:
            torch.save(model, os.path.join('.', CHECKPOINT_MODEL))
            LOGGER.info('saving model with acc {:.3f}'.format(train_acc/len(train_val_set)*100))
            best_acc = train_acc

        if EXPONENTIAL_LR:
            scheduler.step()
    
    LOGGER.info("Done")