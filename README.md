## [Machine Learning HW2 Classification](https://colab.research.google.com/drive/1k6uHcnNbQwuttgSK2ekE5LzpefJeBt6k)
#### **Task - Income high or low**
Problem: Binary classification, to predict the income of an indivisual exceeds 50,000 or not.

Method: **logistic regression** (linear classifier), **linear dicriminant** (generative model)

Dataset: This dataset is obtained by removing unnecessary attributes and balancing the ratio between positively and negatively labeled data in the [**Census-Income (KDD)**](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)) Data Set, which can be found in [**UCI Machine Learning Repository**](https://archive.ics.uci.edu/ml/index.php). 
該數據集包含從美國人口普查局進行的1994年和1995年的當前人口調查中提取的加權普查數據。數據包含41個與人口和就業相關的變量。


----
## [Machine Learning HW3 CNN](https://docs.google.com/presentation/d/1_6TJrFs3JGBsJpdRGLK1Fy_EiJlNvLm_lTZ9sjLsaKE/edit#slide=id.p1)

#### **Task - Food Classification**
此次資料集為網路上蒐集到的食物照片，共有11類:
Bread(麵包), Dairy product(如起司、牛奶、奶油), Dessert(甜食), Egg(蛋), Fried food(炸物), Meat(肉類), Noodles/Pasta(麵食), Rice(米飯), Seafood(海鮮), Soup(湯), and Vegetable/Fruit(蔬菜水果).
Training set: 9866張
Validation set: 3430張
Testing set: 3347張

*Dataset Link: https://reurl.cc/3DLavL*
*Kaggle competition: https://reurl.cc/ZO7XpM*
*Colab sample code: https://colab.research.google.com/drive/1rDbT2_7ULwKUMO1oJ7w6oD20K5d75bBW#scrollTo=zhzdomRTOKoJ*


----
## [PyTorch Introduction](https://www.youtube.com/watch?v=kQeezFrNoOg&feature=youtu.be)

*[PyTorch 官方文檔（ver 1.2.0）](https://pytorch.org/docs/1.2.0/)*
*Cheat sheet: https://hackmd.io/@rh0jTfFDTO6SteMDq91tgg/HkDRHKLrU*
*Colab sample code: https://colab.research.google.com/drive/1CmkRJ5R41D2QIZXQLszMt68TWUrGd-yf*

- 微分梯度計算automatic differentiation
- 常用架構與函數PyTorch common functions in deep learning
- Data Processing with PyTorch `DataSet`
- Mixed Presision Training in PyTorch


----
## [Machine Learning HW4 RNN](https://docs.google.com/presentation/d/1W5-D0hqchrkVgQxwNLBDlydamCHx5yetzmwbUiksBAA/edit#slide=id.g7cd4f194f5_2_45)

#### **Task - Text Sentiment Classification**
資料為Twitter上收集的推文，每則推文都會被標註為正面(1)或負面(0)，如:
`thanks! i love the color selectors. btw, that's a great way to search and list. (LABEL=1)`、
`I feel icky, i need a hug. (LABEL=0)`

除了labeled data以外，我們還額外提供了120萬筆左右的 unlabeled data: 
- labeled training data ：20萬
- unlabeled training data：120萬
- testing data：20萬（10 萬 public，10 萬 private）

##### Preprocessing the sentences
- 首先建立字典，字典元素為每一個字與其所對應到的index value。
    - example, `i have a pen.` → [1, 2, 3, 4]
      `i have an apple.` → [1, 2, 5, 6]
    **{'i': 1, 'have': 2, 'a': 3, 'pen': 4, ...}**

1. 利用word embedding來代表每一個單字。亦即，用一個向量來表示字(或詞)的意思。
    <img src="C:\Users\user\Pyvirtualenv\course_CNN\images\Machine-Learning-HW4-RNN_1.PNG" style="vertical-align:middle; margin:0px 50px" width="80%" >
    <img src="C:\Users\user\Pyvirtualenv\course_CNN\images\Machine-Learning-HW4-RNN_2.PNG" style="vertical-align:middle; margin:0px 50px" width="80%" >
    <img src="C:\Users\user\Pyvirtualenv\course_CNN\images\Machine-Learning-HW4-RNN_4.PNG" style="vertical-align:middle; margin:0px 50px" width="80%" >
2. 利用bag of words (BOW)方式得到代表該句子的vector
    <img src="C:\Users\user\Pyvirtualenv\course_CNN\images\Machine-Learning-HW4-RNN_3.PNG" style="vertical-align:middle; margin:0px 50px" width="80%" >


##### Semi-supervised Learning
<img src="C:\Users\user\Pyvirtualenv\course_CNN\images\Machine-Learning-HW4-RNN_5.PNG" style="vertical-align:middle; margin:0px 50px" width="80%" >


##### See also:
[自然語言處理入門- Word2vec小實作](https://medium.com/pyladies-taiwan/%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86%E5%85%A5%E9%96%80-word2vec%E5%B0%8F%E5%AF%A6%E4%BD%9C-f8832d9677c8)


----
