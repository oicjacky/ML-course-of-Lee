''' RNN Model '''
import torch
from torch import nn


class LSTM_Net(nn.Module):
    r'''
    Args:
        embedding: A word embedding matrix where rows represent word indices
            and columns represent embedded features.
        embedding_dim: Dimension of an embedding vector(a word).
        hidden_dim: Dimension of LSTM hidden states.
        num_layers: Number of recurrent layers, here LSTM.
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``.
        dropout: A dropout probability of nn.Dropout layer after LSTM layer.
        fix_embedding: 是否將 embedding fix 住，如果 fix_embedding 為 False，
            在訓練過程中，embedding 也會跟著被訓練.
        concate: Wether to concate more informations which is the statistics
            (min, max, mean) of LSTM's hidden states.
    '''
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers,
                 dropout=0.5, fix_embedding=True, bidirectional=False,
                 concate=False):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer, 以embedding_matrix為weight製作torch.nn.Embedding, 之後給word index就會回傳embedding vector.
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
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
        inputs = self.embedding(inputs) # inputs 的 dimension (batch, seq_len, embedding_dim) 
        x, _ = self.lstm(inputs, None) # x 的 dimension (batch, seq_len, hidden_dim)
        if self.concate:
            # 取每一個時間點的min, max, mean 組合起來
            x = torch.cat([x.min(dim=1).values , x.max(dim=1).values , x.mean(dim=1)] , dim=1)
        else:
            # 取LSTM最後一個時間點的hidden state
            x = x[:, -1, :] 
        x = self.classifier(x)
        return x


class DNN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        return self.dnn(inputs.float())