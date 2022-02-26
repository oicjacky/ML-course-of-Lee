import numpy as np
import torch.nn as nn
import torch.utils.data as data

from preprocess import EN2CNDataset
from utils import (infinite_iter, build_model, save_model,
                    CONFIG, LOGGER)
from training import train
from testing import test


def train_process(config):
    LOGGER.info('[Read data] train and validation data.')
    train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, batch_size=1)
    LOGGER.info('[Build Model] Seq2Seq model with Encoder and Decoder.')
    model, optimizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    LOGGER.info('Start training.')
    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while (total_steps < config.num_steps):
        total_steps += config.summary_steps
        model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, train_dataset)
        avg_train_loss = sum(loss) *5 /config.summary_steps
        train_losses += loss
        LOGGER.info("[Training {}] train loss: {:.3f}, Perplexity: {:.3f}".format(total_steps, avg_train_loss, np.exp(avg_train_loss)))
        
        val_loss, bleu_score, result = test(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
        LOGGER.info("[Validation {}] val loss: {:.3f}, Perplexity: {:.3f}, bleu score: {:.3f}".format(total_steps, val_loss, np.exp(val_loss), bleu_score))

        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            LOGGER.info(f'[Store model] at {total_steps}')
            save_model(model, optimizer, config.store_model_path, total_steps)
            with open(f'{config.store_model_path}/output_{total_steps}.txt', 'w', encoding='utf-8') as f:
                for line in result:
                    print(line, file=f)

    return train_losses, val_losses, bleu_scores

def test_process(config):
    LOGGER.info('[Read data] test data.')
    test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
    test_loader = data.DataLoader(test_dataset, batch_size=1)
    LOGGER.info('[Build Model] Seq2Seq model with Encoder and Decoder.')
    model, optimizer = build_model(config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    # 測試模型
    test_loss, bleu_score, result = test(model, test_loader, loss_function)
    # 儲存結果
    with open(f'./test_output.txt', 'w', encoding='utf-8') as f:
        for line in result:
            print(line, file=f)

    return test_loss, bleu_score


if __name__ == "__main__":

    #NOTE: train
    train_losses, val_losses, bleu_scores = train_process(CONFIG)

    #NOTE: test
    test_loss, bleu_score = test_process(CONFIG)
    LOGGER.info(f'test loss: {test_loss}, bleu_score: {bleu_score}')

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('次數')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.show()