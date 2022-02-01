import configparser
import torch
import numpy as np
from gensim.utils import simple_preprocess
from typing import Tuple, Union, List


def load_training_data(path = "training_label.txt") -> Union[Tuple[List,List], List]:
    # 如果是 'training_label.txt'，需要讀取 label，如果是 'training_nolabel.txt'，不需要讀取 label
    if 'training_label' in path:
        with open(path, 'r', encoding= 'utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r', encoding= 'utf-8') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


def load_training_with_simple_preprocess(path) -> Union[Tuple[List,List], List]:
    ''' Use `simple_preprocess` into `load_training_data` '''
    if 'training_label' in path:
        with open(path, 'r', encoding= 'utf-8') as f:
            lines = f.readlines()
        x = [simple_preprocess(line[2:]) for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r', encoding= 'utf-8') as f:
            lines = f.readlines()
            x = [simple_preprocess(line) for line in lines]
        return x


def load_testing_data(path = 'testing_data.txt') -> List:
    with open(path, 'r', encoding= 'utf-8') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X


def evaluation(outputs, labels):
    '''
    outputs: probability (float)
    labels: labels
    '''
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為正面
    outputs[outputs<0.5] = 0 # 小於 0.5 為負面
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


config = configparser.ConfigParser()
#NOTE: this is a quick fix for reading different config files.
if torch.cuda.is_available():
    config.read(r'./config.ini')
else:
    config.read(r'./config_cpu.ini')
TRAIN_DATA_PATH = config.get('Data', 'TRAIN_DATA_PATH')
TRAIN_NOLABEL_DATA_PATH = config.get('Data', 'TRAIN_NOLABEL_DATA_PATH')
TEST_DATA_PATH = config.get('Data', 'TEST_DATA_PATH')

MODEL_CONFIG = dict(
    embedding_dim = config.getint('Model', 'embedding_dim'),
    hidden_dim = config.getint('Model', 'hidden_dim'),
    num_layers = config.getint('Model', 'num_layers'),
    dropout = config.getfloat('Model', 'dropout'),
    w2v_path = config.get('Model', 'w2v_path'),
    sen_len = config.getint('Model', 'sen_len'),
    batch_size = config.getint('Model', 'batch_size'),
    epoch = config.getint('Model', 'epoch'),
    lr = config.getfloat('Model', 'lr'),
)
PREDICTION = config.get('File', 'prediction')


if __name__ == '__main__':
    
    random_number = np.random.randint(10000)
    print('Randomly choose a data, index: ', random_number)
    print(load_training_data(TRAIN_DATA_PATH)[0][random_number], load_training_data(TRAIN_DATA_PATH)[1][random_number])
    print(load_testing_data(TEST_DATA_PATH)[random_number])