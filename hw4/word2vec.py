# w2v.py
# 這個 block 是用來訓練 word to vector 的 word embedding
# 注意！這個 block 在訓練 word to vector 時是用 cpu，可能要花到 10 分鐘以上
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec
from utils import load_training_data, load_testing_data, TRAIN_DATA_PATH, TEST_DATA_PATH, TRAIN_NOLABEL_DATA_PATH

def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    model = word2vec.Word2Vec(sentences=x, vector_size=250, window=5, min_count=5, workers=2, epochs=10, sg=1)
    return model

if __name__ == "__main__":
    path_prefix = r'.'
    
    print("loading training data ...")
    train_x, y = load_training_data(TRAIN_DATA_PATH)
    train_x_no_label = load_training_data(TRAIN_NOLABEL_DATA_PATH)

    print("loading testing data ...")
    test_x = load_testing_data(TEST_DATA_PATH)

    #model = train_word2vec(train_x + train_x_no_label + test_x)
    model = train_word2vec(train_x + test_x)
    print(model)

    print("saving model ...")
    model.save(os.path.join(path_prefix, 'w2v_all.model'))