import torch
import numpy as np
from model import Classifier
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


def test_loop(val_loader: DataLoader, model: Classifier,
              loss: CrossEntropyLoss,
              val_acc: float, val_loss: float):
    count, size = 0, len(val_loader.dataset)
    with torch.no_grad():
        for data in val_loader:
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
            count += len(data[1])
            print(f"loss: {val_loss/count:>7f}  [{count:>5d}/{size:>5d}]", end='\r')
        print()
    return val_acc, val_loss