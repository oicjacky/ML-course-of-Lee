import torch
import numpy as np
from model import Classifier
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


def test_loop(val_loader: DataLoader, model: Classifier,
              loss: CrossEntropyLoss, device,
              val_acc: float, val_loss: float):
    count, size = 0, len(val_loader.dataset)
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            val_pred = model(inputs)
            batch_loss = loss(val_pred, labels)

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == labels.cpu().numpy())
            val_loss += batch_loss.item()
            count += len(labels)
            print(f"[{count:>5d}/{size:>5d}] loss: {val_loss/count:>7f} accuracy: {val_acc/count:>7f}", end='\r')
        print()
    return val_acc, val_loss