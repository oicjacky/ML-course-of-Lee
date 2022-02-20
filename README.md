<!-- TODO
- More description in start_pytorch of pytorch introduction
- Add ObjectDetection
- Add ImageNetAPP
 -->
# Machine Learning 2020 Spring, Hung-yi Lee

[Syllabus](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)

  - [x] [Requirement](#requirement)
  - [ ] [Homework 2](#homework-2)
  - [x] [Homework 3](#homework-3)
  - [x] [Homework 4](#homework-4)
  - [x] [PyTorch Introduction](#pytorch-introduction)


## Requirement

- Python 3.7.7 and PyTorch:
    ```
    torch==1.7.1+cu110
    torchaudio==0.7.2
    torchvision==0.8.2+cu110
    # On Colab
    torch==1.10.0+cu111
    torchaudio==0.10.0+cu111
    torchvision==0.11.1+cu111
    ```
- More detail in [`requirement.txt`](https://github.com/oicjacky/ML-course-of-Lee/blob/main/requirement.txt)

## Homework 2

- [Report HW2 Classification]()
    
    <u>**Task - Income high or low**</u>

    Problem: Binary classification, to predict the income of an indivisual exceeds 50,000 or not.

    Method: **logistic regression** (linear classifier), **linear dicriminant** (generative model)

    Dataset: This dataset is obtained by removing unnecessary attributes and balancing the ratio between positively and negatively labeled data in the [Census-Income (KDD)](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)) Data Set, which can be found in [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). 
    該數據集包含從美國人口普查局進行的1994年和1995年的當前人口調查中提取的加權普查數據。數據包含41個與人口和就業相關的變量。

    *Reference:*  
    1. [Detail description of HW2 Classification](https://docs.google.com/presentation/d/1dQVeHfIfUUWxMSg58frKBBeg2OD4N7SD0YP3LYMM7AA/edit#slide=id.g7be340f71d_0_186)
    2. [Colab sample code](https://colab.research.google.com/drive/1k6uHcnNbQwuttgSK2ekE5LzpefJeBt6k)


## Homework 3

- [Report HW3 CNN](https://github.com/oicjacky/ML-course-of-Lee/tree/main/hw3)
    
    <u>**Task - Food Classification**</u>

    此資料集[Food-11](https://www.kaggle.com/vermaavi/food11)為網路上蒐集到的食物照片，共有11類:
    
    - Training set: 9866張
    - Validation set: 3430張
    - Testing set: 3347張
    
    | category        | 類別               |
    | --------------- | ------------------ |
    | Bread           | 麵包               |
    | Dairy product   | 如起司、牛奶、奶油 |
    | Dessert         | 甜食               |
    | Egg             | 蛋                 |
    | Fried food      | 炸物               |
    | Meat            | 肉類               |
    | Noodles/Pasta   | 麵食               |
    | Rice            | 米飯               |
    | Seafood         | 海鮮               |
    | Soup            | 湯                 |
    | Vegetable/Fruit | 蔬菜水果           |

    *Reference:*  
    1. [Detail description of HW3 CNN](https://docs.google.com/presentation/d/1_6TJrFs3JGBsJpdRGLK1Fy_EiJlNvLm_lTZ9sjLsaKE/edit#slide=id.p1)
    2. [Kaggle competition](https://reurl.cc/ZO7XpM)
    3. [Colab sample code](https://colab.research.google.com/drive/1rDbT2_7ULwKUMO1oJ7w6oD20K5d75bBW#scrollTo=zhzdomRTOKoJ)


## Homework 4

- [Report HW4 RNN](https://github.com/oicjacky/ML-course-of-Lee/tree/main/hw4)

    <u>Task - Text Sentiment Classification</u>

    資料為Twitter上收集的推文，每則推文都會被標註為正面(1)或負面(0)，如:
    `thanks! i love the color selectors. btw, that's a great way to search and list. (LABEL=1)`、
    `I feel icky, i need a hug. (LABEL=0)`

    除了labeled data以外，我們還額外提供了120萬筆左右的 unlabeled data: 
    - labeled training data ：20萬
    - unlabeled training data：120萬
    - testing data：20萬（10 萬 public，10 萬 private）

    Preprocessing the sentences:
    - 首先建立字典，字典元素為每一個字與其所對應到的index value。
        - e.g. `I have a pen.` → [1, 2, 3, 4];
        `I have an apple.` → [1, 2, 5, 6]  
        where {'I': 1, 'have': 2, 'a': 3, 'pen': 4, 'an': 5, 'apple': 6}

    1. 利用word embedding來代表每一個單字。亦即，用一個向量來表示字(或詞)的意思。
        <!-- <img src="images\Machine-Learning-HW4-RNN_1.PNG" style="vertical-align:middle; margin:0px 50px" width="60%" >
        <img src="images\Machine-Learning-HW4-RNN_2.PNG" style="vertical-align:middle; margin:0px 50px" width="60%" >
        <img src="images\Machine-Learning-HW4-RNN_4.PNG" style="vertical-align:middle; margin:0px 50px" width="60%" > -->
    2. 利用bag of words (BOW)方式得到代表該句子的vector
        <!-- <img src="images\Machine-Learning-HW4-RNN_3.PNG" style="vertical-align:middle; margin:0px 50px" width="60%" > -->

    Semi-supervised Learning:
    - 使用unlabeled data協助模型訓練，如Self-Training
        <!-- <img src="images\Machine-Learning-HW4-RNN_5.PNG" style="vertical-align:middle; margin:0px 50px" width="60%" > -->
    
    *Reference:*  
    1. [Detail description of HW4 RNN](https://docs.google.com/presentation/d/1W5-D0hqchrkVgQxwNLBDlydamCHx5yetzmwbUiksBAA/edit#slide=id.g7cd4f194f5_2_45)  
    2. [Colab sample code](https://colab.research.google.com/drive/16d1Xox0OW-VNuxDn1pvy2UXFIPfieCb9)


## PyTorch Introduction

- [Start PyTorch](https://github.com/oicjacky/ML-course-of-Lee/tree/main/start_pytorch)
    ```markdown
    - start_Pytorch.py
    - start_dataloader.py
    - start_buildnn.py
    - start_autograd.py
    - start_optimizer.py
    ```
    *Reference:*  
    1. [PyTorch documentation(ver. 1.10 stable release)](https://pytorch.org/docs/1.10/)
    2. [Cheat sheet organized by Chen](https://hackmd.io/@rh0jTfFDTO6SteMDq91tgg/HkDRHKLrU)