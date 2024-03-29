## [HW4 Kaggle](https://www.kaggle.com/c/ml2020spring-hw4/overview)

## Directory structure

Modified the template code of HW4 colab notebook, and organized the code structure into the following scripts:

```markdown
- utils.py
- word2vector.py
- preprocess.py
    - data.py
- model.py
- main.py
    - train.py
    - test.py
```

## Questions:

- (1%) 請說明你實作的 RNN 的模型架構、word embedding 方法、訓練過程 (learning curve) 和準確率為何？ (盡量是過 public strong baseline 的 model)
    
    此RNN模型主要由多層LSTM加上後面一層linear+sigmoid layer，最後output probability with $\geq$ 0.5=positive; $<$ 0.5=negative.
    <img src="../images/hw4_pic01.PNG" style="vertical-align:middle; margin:0px 30px" width="60%" >

    | Layer            | Detail                                                                                                 |
    | ---------------- | ------------------------------------------------------------------------------------------------------ |
    | Word embedding   | `gensim(4.0.1)` Word2Vec model with vector_size=250, window=5, min_count=5, workers=2, epochs=10, sg=1 |
    | LSTM             | `torch.nn.LSTM` with hidden_size=150, num_layers=2, bidirectional=True                                 |
    | Dropout          | with dropout probability=0.3                                                                           |
    | Linear + Sigmoid | a classifier with the threshold 0.5 to classify 0=negative, 1=positive                                 |
    
    Note that the other parameters are:
    - length of sentence: 30
    - batch size: 32
    - learning rate: 0.001 with exponential decay rate 0.9
    - epoch: 10 with early stopping rule, the count of no-increasing accuracy greater than 2

    | Train   | Validation      | Test(private) | Test(public) |
    | ------- | --------------- | ------------- | ------------ |
    | 0.88902 | 0.82535         | 0.82509       | 0.82636      | concate (min, max, mean) of LSTM's hidden states |
    | ---     | strong baseline | 0.82011       | 0.82171      |
    | ---     | simple baseline | 0.76917       | 0.76978      |
    
    <img src="../images/hw4_pic03.PNG" style="vertical-align:middle; margin:0px 30px" width="80%" >
<!--|0.89026|0.82585         |0.82588       |0.82581      |the same but a little worse, concate (min, max, mean) of LSTM's hidden states  -->
<!--|0.86807|0.82290         |0.82534       |0.82496      | only concate mean of LSTM's hidden states  -->
    

- (2%) 請比較 BOW + DNN 與 RNN 兩種不同 model 對於 "today is a good day, but it is hot" 與 "today is hot, but it is a good day" 這兩句的分數 (過 Sigmoid 後的數值)，並討論造成差異的原因。 

    | Sentence                             | BOW+DNN | RNN (LSTM) |
    | ------------------------------------ | ------- | ---------- |
    | "today is a good day, but it is hot" | 0.42266 | 0.22814    |
    | "today is hot, but it is a good day" | 0.42266 | 0.97169    |

    根據上表，RNN模型將"today is a good day, but it is hot"判斷為負面，而"today is hot, but it is a good day"則為正面。
    然而，使用BOW + DNN兩者機率都是0.42266，代表它無法有效分辨這兩句的差異(因為兩句的bag of word representation一樣)。
    此外，有個tricky是: ","是否放入bag of word? 如果放入，則"day,"和"data"與"hot"和"hot,"的差異，將會使BOW+DNN結果不同(有放入情況下分別為0.56966與0.43803，恰和RNN判斷相反)。
    
<!-- # RNN
"today is a good day, but it is hot" with probabilty 0.22814
"today is hot, but it is a good day" with probabilty 0.97169
# BOW
"today is a good day, but it is hot" with probabilty 0.56966
"today is hot, but it is a good day" with probabilty 0.43803 -->

- (1%) 請敘述你如何 improve performance（preprocess、embedding、架構等等），並解釋為何這些做法可以使模型進步，並列出準確率與 improve 前的差異。（semi-supervised 的部分請在下題回答）

    1. **preprocess、embedding**:
        主要調整輸入句子長度(length of sentence=30)和`gensim.models.Word2Vec`的參數，包含:
        以Skip-gram方式訓練(sg=1)，詞向量維度(vector_size=250)，訓練次數(epoch=10)，其他如window=5, min_count=5都為default value。

        | Model      | Test(private) | Test(public) |
        | ---------- | ------------- | ------------ |
        | sen_len=20 | 0.80669       | 0.80585      |
        | sen_len=30 | 0.81909       | 0.82067      |
        Other parameters: embedding_dim = 250, hidden_dim = 150, bidirectional=False, num_layers = 1, dropout = 0.5, batch_size = 32, epoch = 10, lr = 0.001

    2. **RNN (LSTM)**:
        使用多層`torch.nn.LSTM`並提高hidden state dimension，以捕捉句子前後語氣關係。
        另外，亦使用bidirectional LSTM方式讓模型以前後方向讀取句子(hidden_size=150, num_layers=2, bidirectional=True)。
        最後LSTM的output再過一層Dropout

        | Model                                          | Test(private) | Test(public) |
        | ---------------------------------------------- | ------------- | ------------ |
        | bidirectional=False, num_layers=1, dropout=0.5 | 0.81909       | 0.82067      |
        | bidirectional=True, num_layers=2, dropout=0.3  | 0.82135       | 0.82281      |
        Other parameters: embedding_dim = 250, hidden_dim = 150, batch_size = 32, epoch = 10, lr = 0.001, sen_len=30

        
    3. **架構**:
        一般RNN是取用**最後一層hidden state的output(它代表經萃取word embedding後的產物)**，再過一層linear+sigmoid layer，然後output probability。
        這裡將**所有hidden states(每個時間點)**，進行min, max, mean後組合起來。
        ```python
        # model.py line 43-51
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        if self.concate:
            x = torch.cat([x.min(dim=1).values , x.max(dim=1).values , x.mean(dim=1)] , dim=1)
        else:
            x = x[:, -1, :] 
        x = self.classifier(x)
        ```
        
        此外，使用`torch.optim.lr_scheduler.ExponentialLR` with gamma=0.9調整learning rate。

        | Model                              | Test(private) | Test(public) |
        | ---------------------------------- | ------------- | ------------ |
        | concate=False, ExponentialLR=False | 0.82135       | 0.82281      |
        | concate=True, ExponentialLR=True   | 0.82509       | 0.82636      |
        Other parameters: embedding_dim = 250, hidden_dim = 150, batch_size = 32, epoch = 10, lr = 0.001, sen_len=30, bidirectional=True, num_layers=2, dropout=0.3

    *Reference:*
    1. [*Word2Vec Model官方介紹參數*](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py)
    2. [*Word2Vec的簡易教學與參數調整指南*](https://www.kaggle.com/jerrykuo7727/word2vec)
    3. [*自然語言處理入門- Word2vec小實作*](https://medium.com/pyladies-taiwan/%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86%E5%85%A5%E9%96%80-word2vec%E5%B0%8F%E5%AF%A6%E4%BD%9C-f8832d9677c8)
    4. [*Understanding LSTM*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- (2%) 請描述你的semi-supervised方法是如何標記label，並比較有無semi-supervised training對準確率的影響並試著探討原因（因為 semi-supervise learning 在 labeled training data 數量較少時，比較能夠發揮作用，所以在實作本題時，建議把有 label 的training data從 20 萬筆減少到 2 萬筆以下，在這樣的實驗設定下，比較容易觀察到semi-supervise learning所帶來的幫助）

    #TODO: **Self-training**
