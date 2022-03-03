import configparser
import logging
import nltk
import torch

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, optimizer, store_model_path, step):
    torch.save(model.state_dict(), f'{store_model_path}/model_{step}.ckpt')
    return

def load_model(model, load_model_path):
    print(f'Load model from {load_model_path}')
    model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
    return model

def build_model(config, en_vocab_size, cn_vocab_size):
    # 建構模型
    encoder = Encoder(en_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
    decoder = Decoder(cn_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout, config.attention)
    model = Seq2Seq(encoder, decoder, DEVICE)
    print(model)
    # 建構 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(optimizer)
    if config.load_model:
        model = load_model(model, config.load_model_path)
    model = model.to(DEVICE)
    return model, optimizer

def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences


def computebleu(sentences, targets):
    '''[BLEU with n-gram precision](https://coladrill.github.io/2018/10/20/%E6%B5%85%E8%B0%88BLEU%E8%AF%84%E5%88%86/)'''
    score = 0 
    assert (len(sentences) == len(targets))
    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp 
    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))                                                                                          
    return score


def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)

########
# TODO #
########
# 請在這裡直接 return 0 來取消 Teacher Forcing
# 請在這裡實作 schedule_sampling 的策略
def schedule_sampling():
    return 1


#NOTE: read config.ini
class configurations(object):

    def __init__(self):
        self._config = self._load_config(r'./config.ini')
        self.batch_size = self._config.getint('Model', 'batch_size')
        self.emb_dim = self._config.getint('Model', 'emb_dim')
        self.hid_dim = self._config.getint('Model', 'hid_dim')
        self.n_layers = self._config.getint('Model', 'n_layers')
        self.dropout = self._config.getfloat('Model', 'dropout')
        self.learning_rate = self._config.getfloat('Model', 'learning_rate')
        self.max_output_len = self._config.getint('Model', 'max_output_len')
        self.num_steps = self._config.getint('Model', 'num_steps')
        self.store_steps = self._config.getint('Model', 'store_steps')
        self.summary_steps = self._config.getint('Model', 'summary_steps')
        self.load_model = self._config.getboolean('Path', 'load_model')
        self.store_model_path = self._config.get('Path', 'store_model_path')
        self.load_model_path = self._config.get('Path', 'load_model_path')
        self.data_path = self._config.get('Path', 'data_path')
        self.log_file = self._config.get('Path', 'log_file')
        self.attention = self._config.getboolean('Extra', 'attention')

    def _load_config(self, file) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        if not config.read(file, encoding='utf-8'):
            raise Exception('No config.ini file is read!')
        return config
CONFIG = configurations()
print('config:\n', vars(CONFIG))

#NOTE: logging
def setup_logger(name, log_file =None, level=logging.INFO):
    """ Setup different loggers to record the message with corresponding `level` and `file name` """
    if 'info' in name:
        formatter = logging.Formatter('[%(asctime)s]: %(message)s')
    elif 'error' in name:
        formatter = logging.Formatter('[%(asctime)s] - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    else:
        formatter = logging.Formatter('[%(asctime)s] - %(pathname)s[line:%(lineno)d]: %(message)s')
    if log_file:
        handler = logging.FileHandler(log_file, encoding= 'utf-8')
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
LOGGER = setup_logger(name="main_logger", log_file=CONFIG.log_file)