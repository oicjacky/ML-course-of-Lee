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

    #NOTE: confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    target_names_tbl = {
        0 : 'Bread', #麵包
        1 : 'Dairy product', #如起司、牛奶、奶油
        2 : 'Dessert', #甜食
        3 : 'Egg', #蛋
        4 : 'Fried food', #炸物
        5 : 'Meat', #肉類
        6 : 'Noodles/Pasta', #麵食
        7 : 'Rice', #米飯
        8 : 'Seafood', #海鮮
        9 : 'Soup', #湯
        10: 'Vegetable/Fruit'#蔬菜水果
    }
    def show_confusion_mat(y, y_pred, target_names, title=None):
        confusion_mat = confusion_matrix(y, y_pred, normalize='true')
        print(classification_report(y, y_pred, target_names=target_names))
        sns.heatmap(confusion_mat,
                    square = True, annot = True, fmt = '.2f', cbar = True,
                    xticklabels = target_names,
                    yticklabels = target_names)
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.title(title)
        plt.show()
        #plt.savefig("food_confusion_mat.png", dpi=600, format='png')
    show_confusion_mat(eval_y, prediction, list(target_names_tbl.values()))