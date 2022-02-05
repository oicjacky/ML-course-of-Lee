import configparser
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from preprocess import Preprocessor, ImgDataset


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(r'./config.ini')
    DATA_DIR = config.get('Data', 'data_dir')
    PREDICTION = config.get('File', 'prediction')
    CHECKPOINT_MODEL = config.get('File', 'checkpoint_model')


    print("[Reading evaluation data] using opencv(cv2) read images into np.array")
    _p = lambda p: os.path.join(DATA_DIR, p)
    eval_x, eval_y = Preprocessor.readfile(_p("evaluation"), True)
    eval_set = ImgDataset(eval_x, eval_y, Preprocessor.test_transform)
    eval_loader = DataLoader(eval_set, batch_size=32, shuffle=False)
    

    print('loading checkpoint model...')
    model = torch.load(os.path.join('.', CHECKPOINT_MODEL))
    model.eval()
    prediction = []
    count, size, test_acc = 0, len(eval_loader.dataset), 0
    with torch.no_grad():
        for data in eval_loader:
            test_pred = model(data[0].cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)

            test_acc += np.sum(test_label == data[1].numpy())
            count += len(data[1])
            prediction += test_label.tolist()
            print(f"[{count:>5d}/{size:>5d}] accuracy: {test_acc/count:>7f}", end='\r')
        print()

    #將結果寫入 csv 檔
    with open(PREDICTION, 'w') as f:
        f.write('Id,Category,GroundTrue\n')
        for i, (pred_y, true_y) in  enumerate(zip(prediction, eval_y)):
            f.write('{},{},{}\n'.format(i, pred_y, true_y))
    print("Done")