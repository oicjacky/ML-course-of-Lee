import numpy as np
from model import Classifier
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss


def train_loop(train_loader: DataLoader, model: Classifier,
               loss: CrossEntropyLoss, optimizer: Optimizer,
               train_acc: float, train_loss: float):
    size = len(train_loader.dataset)
    for batch_index, data in enumerate(train_loader):
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值
        
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
        # if batch_index % 100 == 0:
        _loss, current = batch_loss.item(), batch_index * len(data[1])
        print(f"loss: {_loss:>7f}  [{current:>5d}/{size:>5d}]", end='\r')
    print()
    return train_acc, train_loss