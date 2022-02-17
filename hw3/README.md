## [HW3 Kaggle](https://www.kaggle.com/c/ml2020spring-hw3/overview)

## Directory structure

Modified the template code of HW3 colab notebook, and organized the code structure into the following scripts:

```markdown
- preprocess.py
- model.py
- training.py
- testing.py
- main.py
- logger.py
- config.ini
- submit.py
- final.py
```

## Questions:

- 資料集描述如下：
    | index | category        | 類別               |
    | ----- | --------------- | ------------------ |
    | 0     | Bread           | 麵包               |
    | 1     | Dairy product   | 如起司、牛奶、奶油 |
    | 2     | Dessert         | 甜食               |
    | 3     | Egg             | 蛋                 |
    | 4     | Fried food      | 炸物               |
    | 5     | Meat            | 肉類               |
    | 6     | Noodles/Pasta   | 麵食               |
    | 7     | Rice            | 米飯               |
    | 8     | Seafood         | 海鮮               |
    | 9     | Soup            | 湯                 |
    | 10    | Vegetable/Fruit | 蔬菜水果           |

    其中，
    - Training set: 9866張
    - Validation set: 3430張
    - Testing set: 3347張
    - 表格中準確率以Accuracy為主(即testing set表現)，而Train and Validate則是訓練過程評估參數表現的數值。


1. (1%) 請說明你實作的 CNN 模型，其模型架構、訓練參數量和準確率為何？

    <!-- *NOTE1:*  
    Set epoch=80, batch_size=32, learning_rate=0.001, exponential learning rate decay=0.9,  
    Data augmentation:  
        RandomHorizontalFlip() -> RandomRotation(15)  
    | Model                             | Accuracy | Train and Validate                   |
    | --------------------------------- | -------- | ------------------------------------ |
    | CNN (colab code)                  | 0.726920 | train_acc=0.907764, val_acc=0.728654 | ckpt_CNN_colab.model (Accuracy=0.724514 on colab) |
    | VGG16                             | 0.646730 | train_acc=0.707886, val_acc=0.646486 | ckpt_VGG16_None.model                             |
    | VGG16 (linear bathchnorm)         | 0.695639 | train_acc=0.862457, val_acc=0.683243 | ckpt_VGG16_batchnorm.model                        |
    | VGG16 (linear dropout)            | 0.623159 | train_acc=0.642712, val_acc=0.627027 | ckpt_VGG16_dropout.model                          |
    | VGG16 (linear dropout+bathchnorm) | 0.726576 | train_acc=0.811778, val_acc=0.704865 | ckpt_dropout_bathchnorm_80.model                  |
    
    Increase `VGG16 (linear dropout+bathchnorm)`'s epoch from 80 to 100。  
    看來是沒有用，因為early stop已經被觸發了，準確率也沒有上揚:  
    | Model                             | Accuracy | Train and Validate                   |
    | --------------------------------- | -------- | ------------------------------------ |
    | VGG16 (linear dropout+bathchnorm) | 0.726023 | train_acc=0.819278, val_acc=0.702332 | ckpt_dropout_bathchnorm.model | -->

    
    參考VGG16架構在每一層convolution layer後面加上batch normaliztion layer，  
    最後fully connected layer都加上dropout避免overfitting:  
    <img src="..\images\hw3_VGG16.PNG" style="vertical-align:middle; margin:0px 50px" width="40%">

    | Model                                      | Accuracy                            | Train    |
    | ------------------------------------------ | ----------------------------------- | -------- |
    | VGG16 (linear dropout+batch norm) epoch180 | 0.846728                            | 0.996616 | ckpt_VGG16_dropout_batchnorm_final.model | _info_final_0257.log |
    | ----- strong baseline -----                | 0.79928 (private), 0.79318 (public) | ---      |                                          |
    | ----- simple baseline -----                | 0.70788 (private), 0.71727 (public) | ---      |                                          |


    <!-- *NOTE2:*   -->
    - 一開始，我先設定batch size=32, learning rate=0.001, dropout rate=0.5, No exponential learning rate decay  
        其中Data augmentation使用:  
        ```python
        RandomHorizontalFlip() -> 
        RandomCrop(size=(128, 128)) -> 
        ColorJitter(brightness=0.5, hue=0.3) -> 
        RandomRotation(30) -> 
        Normalize(mean=[0.3438, 0.4511, 0.5551], std=[0.2811, 0.2740, 0.2711])  
        ```
        | Model                                      | Accuracy | Train and Validate                   |
        | ------------------------------------------ | -------- | ------------------------------------ |
        | VGG16 (linear dropout+batch norm) epoch180 | 0.760084 | train_acc=0.964930, val_acc=0.742566 | ckpt_VGG16_dropout_batchnorm_ep180.model | _info_0257.log |
        | *other model performance*                  | ---      | ---                                  |
        | CNN (colab code) epoch80                   | 0.699134 | train_acc=0.938780, val_acc=0.669679 | ---                                      | info_0248.log  |
        | CNN (colab code) epoch150                  | 0.715865 | train_acc=0.966045, val_acc=0.706997 | ckpt_CNN_ep150.model                     | info_0248.log  |
        | VGG16 (linear dropout+batch norm) epoch80  | 0.701225 | train_acc=0.787046, val_acc=0.685131 | ---                                      | _info_0257.log |
        | VGG16 (linear dropout+batch norm) epoch150 | 0.749627 | train_acc=0.957125, val_acc=0.741983 | ckpt_VGG16_dropout_batchnorm_ep150.model | _info_0257.log |
    
        此時，`VGG16 (linear dropout+batch norm)`在epocp 180後打開exponential learning rate decay(gamma=0.9)持續訓練:  
        | Model                                      | Accuracy | Train and Validate                   |
        | ------------------------------------------ | -------- | ------------------------------------ |
        | VGG16 (linear dropout+batch norm) epoch300 | 0.782492 | train_acc=0.994729, val_acc=0.765598 | ckpt_VGG16_dropout_batchnorm_ep300.model | _info_0257.log |
        
        可以看到Accuracy從 **74.96%** 提升到 **78.24%**。


    - 調整好設定參數與訓練方式後，把training和validation合併，用更多資料量去訓練模型，  
        發現epoch 120的testing accuracy已達到 **82.01%**。  
        在epoch來到120後，打開exponential learning rate decay(gamma=0.9)持續訓練:
        | Model                                      | Accuracy | Train    |
        | ------------------------------------------ | -------- | -------- |
        | VGG16 (linear dropout+batch norm) epoch180 | 0.846728 | 0.996616 | ckpt_VGG16_dropout_batchnorm_final.model | _info_final_0257.log |
        | *other model performance*                  | ---      | ---      |
        | VGG16 (linear dropout+batch norm) epoch120 | 0.820137 | 0.956829 | ckpt_VGG16_dropout_batchnorm_final.model | _info_final_0257.log |
        | VGG16 (linear dropout+batch norm) epoch160 | 0.844637 | 0.994209 | ckpt_VGG16_dropout_batchnorm_final.model | _info_final_0257.log |
        
        最終，`VGG16 (linear dropout+batch norm)`在testing accuracy上得到 **84.67%** 準確率。  


2. (1%) 請實作與第一題接近的參數量，但 CNN 深度（CNN 層數）減半的模型，並說明其模型架構、訓練參數量和準確率為何？
   
    <!-- *NOTE1:*  
    | Model            | Accuracy | Train and Validate                   |
    | ---------------- | -------- | ------------------------------------ |
    | CNN (colab code) | 0.726920 | train_acc=0.907764, val_acc=0.728654 | ckpt_CNN_colab.model (Accuracy=0.724514 on colab) |
    | VGG16            | 0.646730 | train_acc=0.707886, val_acc=0.646486 | ckpt_VGG16_None.model                             |
    | VGG16_half       | 0.677622 | train_acc=0.996351, val_acc=0.663265 | ckpt_VGG16_half.model                             | -->

    <!-- *NOTE2:*   -->
    | Model                                           | Accuracy | Train and Validate                   | #params  |
    | ----------------------------------------------- | -------- | ------------------------------------ | -------- |
    | VGG16 (linear dropout+batch norm) epoch150      | 0.749627 | train_acc=0.957125, val_acc=0.741983 | 52619851 | ckpt_VGG16_dropout_batchnorm_ep150.model | _info_0257.log      |
    | VGG16_half (linear dropout+batch norm) epoch150 | 0.745444 | train_acc=0.966045, val_acc=0.738484 | 52321503 | ckpt_VGG_half_dropbatch.model            | _info_half_1514.log |
    
    - 發現相較於原本`VGG16`(深且瘦)的模型，較淺較胖的`VGG16_half`在overfitting狀況較明顯。  
      雖然training accuracy高，但不論validation或testing都比較差。
    - **TODO**: 
      1. 修改`VGG16_half`的減半方式，目前是直接調整CNN層數，而未調整CNN的kernel size。  
         而在參數量差不多情況下，`VGG16_half`在後面fully connected layer參數是比較多而胖的。  
         在`VGG`論文中[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)提到，  
         它使用多個kernel size比較小的convolution filter，取代原本少量但kernal size較大的filter。


3. (1%) 請實作與第一題接近的參數量，簡單的 DNN 模型，同時也說明其模型架構、訓練參數和準確率為何？

    設定batch size=32, learning rate=0.001，一樣在fully connected layer都加上dropout=0.5，  
    | Model        | Accuracy | Train and Validate                   | #params  |
    | ------------ | -------- | ------------------------------------ | -------- |
    | DNN epoch100 | 0.336719 | train_acc=0.301743, val_acc=0.326531 | 50866187 | ckpt_DNN_ep100.model | _info_DNN.log |
    | DNN epoch150 | 0.350463 | train_acc=0.319177, val_acc=0.341108 | 50866187 | ckpt_DNN.model       | _info_DNN.log |


4. (1%) 請說明由 1 ~ 3 題的實驗中你觀察到了什麼？
    1. 原本`VGG16`的模型架構由13層convolution layers和3層fully connected layers(及5層max pooling layers)組成。  
       在此架構上加入Batch Normalization和Dropout的處理，提升資料分配集中性(避免Internal Covariate Shift)，避免深層神經網路Gradient vanishing問題，  
       以及訓練模型overfitting狀況。
    2. 在參數量接近(皆在5千萬量級)情況下，
       1. 相較於淺且胖的神經網路，深且瘦的神經網路比較容易得到更好的表現。原因可能為:
          - 深層神經網路就像將主要任務拆分成多個小任務，每一層都萃取出部分特徵，最後匯聚而成分類重要資訊。
          - 胖的神經網路容易導致overfitting，試想一個神經網路，它一層的神經元個數與資料數相同，代表每一個神經元就帶一筆資料的資訊。  
            這時，training可以輕易達到100%準確率，但是testing卻可能異常差，亦即嚴重overfitting。
       2. 一般的DNN模型在影像分類問題上，無法達到CNN模型的表現。原因主要在於convolution的動作，確實可以  
          抓取出圖片中的特徵，包含輪廓、顏色、圖像位置等資訊。因此，在影像問題上CNN模型優於DNN模型。


5. (1%) 請嘗試 data normalization 及 data augmentation，說明實作方法並且說明實行前後對準確率有什麼樣的影響？
   
    比較使用比較少的data augmentation，及有無使用data normalization:  

    <!-- *NOTE1:*   -->
    - Data augmentation:  
        ```python
        RandomHorizontalFlip() ->
        RandomRotation(15)  
        ```
        設定batch size=32, learning rate=0.001, exponential learning rate decay(gamma=0.9)，  
        先訓練到epoch=80，觀察表現：  
        | Model                             | Accuracy | Train and Validate                   |
        | --------------------------------- | -------- | ------------------------------------ |
        | CNN (colab code)                  | 0.726920 | train_acc=0.907764, val_acc=0.728654 | ckpt_CNN_colab.model (Accuracy=0.724514 on colab) |
        | VGG16                             | 0.646730 | train_acc=0.707886, val_acc=0.646486 | ckpt_VGG16_None.model                             |
        | VGG16 (linear batch norm)         | 0.695639 | train_acc=0.862457, val_acc=0.683243 | ckpt_VGG16_batchnorm.model                        |
        | VGG16 (linear dropout)            | 0.623159 | train_acc=0.642712, val_acc=0.627027 | ckpt_VGG16_dropout.model                          |
        | VGG16 (linear dropout+batch norm) | 0.726576 | train_acc=0.811778, val_acc=0.704865 | ckpt_dropout_bathchnorm_80.model                  |
        
        將`VGG16 (linear dropout+batch norm)`的epoch從80提升到100。  
        看來是沒有用，因為early stop已經被觸發了，準確率反而下降:  
        | Model                             | Accuracy | Train and Validate                   |
        | --------------------------------- | -------- | ------------------------------------ |
        | VGG16 (linear dropout+batch norm) | 0.726023 | train_acc=0.819278, val_acc=0.702332 | ckpt_dropout_bathchnorm.model |  |


    <!-- *NOTE2:*   -->
    - Data augmentation:  
        ```python
        RandomHorizontalFlip() -> 
        RandomCrop(size=(128, 128)) -> 
        ColorJitter(brightness=0.5, hue=0.3) -> 
        RandomRotation(30) -> 
        Normalize(mean=[0.3438, 0.4511, 0.5551], std=[0.2811, 0.2740, 0.2711])  
        ```
        如第一題模型表現，加入更多的data augmentation處理後，可以訓練更多的epoch，  
        | Model                                      | Accuracy | Train and Validate                   |
        | ------------------------------------------ | -------- | ------------------------------------ |
        | CNN (colab code) epoch80                   | 0.699134 | train_acc=0.938780, val_acc=0.669679 | ---                                      | info_0248.log  |
        | CNN (colab code) epoch150                  | 0.715865 | train_acc=0.966045, val_acc=0.706997 | ckpt_CNN_ep150.model                     | info_0248.log  |
        | VGG16 (linear dropout+batch norm) epoch80  | 0.701225 | train_acc=0.787046, val_acc=0.685131 | ---                                      | _info_0257.log |
        | VGG16 (linear dropout+batch norm) epoch150 | 0.749627 | train_acc=0.957125, val_acc=0.741983 | ckpt_VGG16_dropout_batchnorm_ep150.model | _info_0257.log |
        | VGG16 (linear dropout+batch norm) epoch180 | 0.760084 | train_acc=0.964930, val_acc=0.742566 | ckpt_VGG16_dropout_batchnorm_ep180.model | _info_0257.log |

        | Model                                      | Accuracy | Train    |
        | ------------------------------------------ | -------- | -------- |
        | VGG16 (linear dropout+batch norm) epoch180 | 0.846728 | 0.996616 | ckpt_VGG16_dropout_batchnorm_final.model | _info_final_0257.log |
        
        最終模型準確率也會隨之提升，得到 **84.67%** 準確率。  

6. (1%) 觀察答錯的圖片中，哪些 class 彼此間容易用混？[繪出 confusion matrix 分析]

    <img src="..\images\food_confusion_mat_1.png" style="vertical-align:middle; margin:0px 50px" width="75%">
    
    X軸代表模型判斷類別，Y軸代表真實類別。其中，每一列(row)加總等於1，  
    代表對角線數值為**正確判斷比例**，而其他數值為"原本為該列類別，**誤判為該行類別**"。
    - Dairy product準確率70%為最低，有12%被誤判為Dessert，與5%被誤判為Seafood。
    - Dessert準確率79%，但有5%被誤判為Bread。
    - Egg準確率78%，有9%被誤判為Bread，與5%被誤判為Dessert。
    - 其他類別準確率皆至少82%以上，最高為Noodles\Pasta和Soup的95%。