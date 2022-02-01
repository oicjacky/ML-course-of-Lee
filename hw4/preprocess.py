import os
import torch
import gensim
from gensim.models import Word2Vec


class Preprocess():
    '''
    Steps:
        0. Initialize `Word2vec` model and
            word embedding look-up matrix with `make_embedding`.
        1. Then, generate X and Y by
            X = `sentence_word2idx` with shape (num_data, feature_dim)
            Y = `labels_to_tensor`  with shape (num_data,)
    '''
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    
    def get_w2v_model(self):
        ''' Load pre-trained `Word2Vec` model. '''
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    
    def add_embedding(self, word):
        ''' Set a randomized representation vector to these word embeddings.
        It must be "<PAD>" or "<UNK>".
        '''
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得訓練好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 製作一個 word2idx 的 dictionary
        # 製作一個 idx2word 的 list
        # 製作一個 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.key_to_index):
            #e.g. self.word2index['he'] = 1 
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
            print('get words #{}'.format(i+1), end='\r')
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 將 "<PAD>" 跟 "<UNK>" 加進 embedding 裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    
    def pad_sequence(self, sentence):
        # 將每個句子變成一樣的長度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    
    def sentence_word2idx(self):
        # 把句子裡面的字轉成相對應的 index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    
    def labels_to_tensor(self, y):
        # 把 labels 轉成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)


class BagOfWord:

    def __init__(self, sentences):
        self.sentences = sentences
        self.vocab = gensim.corpora.Dictionary(sentences) # `sentences` is tokenized documents
        self._check_memory()

    def _check_memory(self, threshold=1e8):
        vector_len = len(self.sentences)*len(self.vocab)
        assert vector_len < threshold, f"To large dimension of input data {vector_len}!"

    def sentence_word2idx(self):
        ''' Return torch.Tensor with shape (len(sentences), len(vocab)) '''
        sentence_list = []
        for sen in self.sentences:
            bow_vector = torch.zeros((len(self.vocab), 1), dtype=torch.int8)
            for id, val in self.vocab.doc2bow(sen):
                bow_vector[id] = val
            sentence_list.append(bow_vector)
        return torch.cat(sentence_list, dim=1).T
    
    def labels_to_tensor(self, y):
        ''' The same as `Preprocess.labels_to_tensor` '''
        y = [int(label) for label in y]
        return torch.LongTensor(y)


class TwitterDataset(torch.utils.data.Dataset):
    """ Expected data shape like:(data_num, feature_dim).
    Data can be a list of numpy array or a list of lists
    Methods:
        __len__ will return the number of data.
        __getitem__ return the data by index.
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    from utils import load_training_data, TRAIN_DATA_PATH
    w2v_path = os.path.join('.', 'w2v_all.model')
    train_x, y = load_training_data(TRAIN_DATA_PATH)
    sen_len = 20
    batch_size = 128

    #NOTE: Preprocess
    # preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    # embedding = preprocess.make_embedding(load=True)
    # train_x = preprocess.sentence_word2idx()
    # y = preprocess.labels_to_tensor(y)
    
    #NOTE: Bag of Word
    train_x, y = train_x[:5000], y[:5000]
    preprocess = BagOfWord(train_x)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 把 data 做成 dataset 供 dataloader 取用
    train_dataset = TwitterDataset(X = train_x, y = y)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 2)
    for i, (inputs, labels) in enumerate(train_loader):
        print(f'[{i}/{len(train_loader)}] batch data with size {batch_size}')
        print(inputs, inputs.size(), labels, sep='\n')
        if i == 2:
            import pdb
            pdb.set_trace()

    print("Done")