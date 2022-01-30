'''
Train and Test the model.
'''
import os
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from utils import (TRAIN_DATA_PATH, TRAIN_NOLABEL_DATA_PATH, TEST_DATA_PATH, MODEL_CONFIG, PREDICTION,
                    evaluation, load_training_data, load_testing_data)
from preprocess import Preprocess, TwitterDataset
from model import LSTM_Net


def training(batch_size: int, n_epoch: int, lr: float, model_dir: str,
             train: torch.utils.data.DataLoader, valid: torch.utils.data.DataLoader,
             model: LSTM_Net, device: torch.device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    
    model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數
    criterion = nn.BCELoss() # 定義損失函數，這裡我們使用 binary cross entropy loss
    t_batch, v_batch  = len(train), len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr) # 將模型的參數給 optimizer，並給予適當的 learning rate
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    total_loss, total_acc, best_acc, early_stop = 0, 0, 0, 0

    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做 training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model(inputs) # 將 input 餵給模型
            outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
            loss = criterion(outputs, labels) # 計算此時模型的 training loss
            loss.backward() # 算 loss 的 gradient
            optimizer.step() # 更新訓練模型的參數
            correct = evaluation(outputs, labels) # 計算此時模型的 training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # 這段做 validation
        model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
                labels = labels.to(device, dtype=torch.float) # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
                outputs = model(inputs) # 將 input 餵給模型
                outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
                loss = criterion(outputs, labels) # 計算此時模型的 validation loss
                correct = evaluation(outputs, labels) # 計算此時模型的 validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                if not torch.cuda.is_available():
                    torch.save(model, "{}/ckpt.model".format(model_dir))
                else:
                    torch.save(model, "{}/ckpt_cpu.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
                early_stop = 0
            else:
                early_stop += 1
            if early_stop > 2:
                print(f'Early stopping with acc {best_acc:.3f}!')
                break
        print('-----------------------------------------------')
        model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數（因為剛剛轉成 eval 模式）
        scheduler.step()


def testing(batch_size: int, test_loader: torch.utils.data.DataLoader,
            model: LSTM_Net, device: torch.device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為正面
            outputs[outputs<0.5] = 0 # 小於 0.5 為負面
            ret_output += outputs.int().tolist()
    return ret_output


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
    if not torch.cuda.is_available():
        model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    else:
        model = torch.load(os.path.join(model_dir, 'ckpt_cpu.model'))
    outputs = testing(batch_size, test_loader, model, device)

    # 寫到 csv 檔案供上傳 Kaggle
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(PREDICTION, index=False)
    print("Finish Predicting")


if __name__ == "__main__":

    main()