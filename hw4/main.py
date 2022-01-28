'''
Train and Test the model.
'''
import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from utils import (TRAIN_DATA_PATH, TRAIN_NOLABEL_DATA_PATH, TEST_DATA_PATH, MODEL_CONFIG, 
                    load_training_data, load_testing_data)
from preprocess import Preprocess, TwitterDataset
from model import LSTM_Net, training, testing


def main():
    # 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 處理好各個 data 的路徑
    train_with_label = TRAIN_DATA_PATH
    train_no_label = TRAIN_NOLABEL_DATA_PATH
    testing_data = TEST_DATA_PATH
    w2v_path = MODEL_CONFIG['w2v_path'] # 處理 word to vec model 的路徑

    # 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
    fix_embedding = True # fix embedding during training
    sen_len = MODEL_CONFIG['sen_len']
    batch_size = MODEL_CONFIG['batch_size']
    epoch = MODEL_CONFIG['epoch']
    lr = MODEL_CONFIG['lr']
    embedding_dim = MODEL_CONFIG['embedding_dim']
    hidden_dim = MODEL_CONFIG['hidden_dim']
    num_layers = MODEL_CONFIG['num_layers']
    dropout = MODEL_CONFIG['dropout']
    num_workers = 2

    # model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
    model_dir = r'.' # model directory for checkpoint model

    print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)

    # 對 input 跟 labels 做預處理
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 製作一個 model 的對象
    model = LSTM_Net(embedding, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, fix_embedding=fix_embedding)
    model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

    # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
    X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

    # 把 data 做成 dataset 供 dataloader 取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    # 把 data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = num_workers)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = num_workers)

    # 開始訓練
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)


    # 開始測試模型並做預測
    print("loading testing data ...")
    test_x = load_testing_data(testing_data)
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = num_workers)
    print('\nload model ...')
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    outputs = testing(batch_size, test_loader, model, device)

    # 寫到 csv 檔案供上傳 Kaggle
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    print("save csv ...")
    tmp.to_csv('predict.csv', index=False)
    print("Finish Predicting")

    # 以下是使用 command line 上傳到 Kaggle 的方式
    # 需要先 pip install kaggle、Create API Token，詳細請看 https://github.com/Kaggle/kaggle-api 以及 https://www.kaggle.com/code1110/how-to-submit-from-google-colab
    # kaggle competitions submit [competition-name] -f [csv file path]] -m [message]
    # e.g., kaggle competitions submit ml-2020spring-hw4 -f output/predict.csv -m "......"


if __name__ == "__main__":

    main()