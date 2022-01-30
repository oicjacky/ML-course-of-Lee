''' RNN Model '''
import torch
from torch import nn


class LSTM_Net(nn.Module):

    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers,
                 dropout=0.5, fix_embedding=True, bidirectional=False,
                 concate=False):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer, 以embedding_matrix為weight製作torch.nn.Embedding, 之後給word index就會回傳embedding vector.
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.concate = concate
        linear_dim = 2*hidden_dim if bidirectional else hidden_dim
        if self.concate: linear_dim *= 3
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(linear_dim, 1, bias=True),
                                        nn.Sigmoid())

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        if self.concate:
            # 取每一個時間點的min, max, mean 組合起來
            x = torch.cat([x.min(dim=1).values , x.max(dim=1).values , x.mean(dim=1)] , dim=1)
        else:
            # 取用 LSTM 最後一個時間點的 hidden state
            x = x[:, -1, :] 
        x = self.classifier(x)
        return x