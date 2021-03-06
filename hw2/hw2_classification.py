"""
Problem
=======
Binary classification, to predict the income of an indivisual
exceeds 50,000 or not.

case(i). Training of Logistic regression
========================================
Mini-batch gradient descent is used here, in which training data
are split into several mini-batches and each batch is feed into
the model sequentially for losses and gradients computation.
Weights and bias are updated on a mini-batch basis.

Once we have gone through the whole training set, the data have to
be re-shuffled and mini-batch gradient desent has to be run on
it again. We repeat such process until max number of iterations
is reached.

我們使用小批次梯度下降法來訓練。訓練資料被分為許多小批次，針對每一個小批次，我們分別計算其梯度以及損失，並根據該批次來更新模型的參數。當一次迴圈完成，也就是整個訓練集的所有小批次都被使用過一次以後，我們將所有訓練資料打散並且重新分成新的小批次，進行下一個迴圈，直到事先設定的迴圈數量達成為止。

case(ii). LDA with closed-form sample mean and covariance matrix
================================================================
See more in `LinearDicriminantAnalysis.__doc__`
"""
import os
import numpy as np
from typing import Tuple


class PrepareData:

    def __init__(self, file_path = r"E:/Download/dataset/data/") -> None:
        self.file_path = file_path


    def _load_data_from_disk(self):
        """Load and parse csv files to np.array.  
        where `file_path` is data path with default: "E:/Download/dataset/data/"

        該數據集包含從美國人口普查局進行的1994年和1995年的當前人口調查中提取的加權普查數據。數據包含41個與人口和就業相關的變量
        """
        with open(os.path.join(self.file_path, "X_train")) as f:
            next(f) # first line is columns info
            X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

        with open(os.path.join(self.file_path, "Y_train")) as f:
            next(f)
            Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)

        with open(os.path.join(self.file_path, "X_test")) as f:
            next(f)
            X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
        return X_train, Y_train, X_test


    def _normalize(self, X, train = True, specified_column = None, X_mean = None, X_std = None):
        """Normalizes specific columns of X.
        The mean and standard variance of training data will be reused when processing testing data.
        
        Arguments:
            X: data to be processed
            train: 'True' when processing training data, 'False' for testing data
            specific_column: indexes of the columns that will be normalized. If 'None', all columns
                will be normalized.
            X_mean: mean value of training data, used when train = 'False'
            X_std: standard deviation of training data, used when train = 'False'
        Outputs:
            X: normalized data
            X_mean: computed mean value of training data
            X_std: computed standard deviation of training data"""
        epslion = 1e-8
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        if train:
            X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
            X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

        X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + epslion)
        
        return X, X_mean, X_std


    def _train_dev_split(self, X, Y, dev_ratio = 0.25):
        """ splits data into training and development (validation) set. """
        train_size = int(len(X) * (1 - dev_ratio))
        return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


    def loading_entry(self):

        X_train, Y_train, X_test = self._load_data_from_disk()
        
        # Normalize training and testing data
        X_train, X_mean, X_std = self._normalize(X_train, train = True)
        X_test, _, _ = self._normalize(X_test, train = False,
                                       X_mean = X_mean, X_std = X_std)

        # Split data into training set and development set
        X_train, Y_train, X_dev, Y_dev = self._train_dev_split(X_train, Y_train, dev_ratio = 0.1)

        print('Size of training set: {}'.format(X_train.shape[0]))
        print('Size of development set: {}'.format(X_dev.shape[0]))
        print('Size of testing set: {}'.format(X_test.shape[0]))
        print('Dimension of data: {}'.format(X_train.shape[1]))
        return X_train, Y_train, X_dev, Y_dev, X_test


class UsageFunction:
    """Some functions that will be repeatedly used when
    iteratively updating the parameters."""    
    
    def _shuffle(self, X, Y):
        """Shuffles two equal-length list/array, X and Y, together. """
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        return X[randomize], Y[randomize]

    def _sigmoid(self, z, epsilon = 1e-8):
        """To avoid overflow, min/max output value. """
        return np.clip( 1.0 / (1 + np.exp(-z)), epsilon, 1 - epsilon )

    def _f(self, X, w, b):
        """Logistic regression function, parameterized by w and b.
        
        Arguements:
            X: input data, shape = [batch_size, data_dimension]
            w: weight vector, shape = [data_dimension, ]
            b: bias, scalar
        Output:
            predicted probability of each row of X being positively labeled, shape = [batch_size, ]"""
        return self._sigmoid(X @ w + b)#_sigmoid(np.matmul(X, w) + b)

    def _predict(self, X, w, b):
        """Returns a truth value prediction for each row of X 
        by rounding the result of logistic regression function."""
        return np.round(self._f(X, w, b)).astype(np.int16)

    def _accuracy(self, Y_pred, Y_label):
        """Calculates prediction accuracy"""
        return 1 - np.mean(np.abs( Y_pred - Y_label ))

    def _cross_entropy_loss(self, y_pred, Y_label):
        """Computes the cross entropy.
        
        Arguements:
            y_pred: probabilistic predictions, float vector
            Y_label: ground truth labels, bool vector
        Output:
            cross entropy, scalar"""
        return -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log((1 - y_pred)))

    def _gradient(self, X, Y_label, w, b):
        """Computes the gradient of cross entropy loss with respect to weight w and bias b."""
        y_pred = self._f(X, w, b)
        pred_error = Y_label - y_pred
        w_grad = -np.sum(pred_error * X.T, 1)
        b_grad = -np.sum(pred_error)
        return w_grad, b_grad


class LogisticRegression(UsageFunction):

    def __init__(self, w, b,
                max_iter = 10,
                batch_size = 8,
                learning_rate = 0.2,
                train_loss = [],
                dev_loss = [],
                train_acc = [],
                dev_acc = [],
                ) -> None:
        # Zero initialization for weights ans bias
        self.w = w
        self.b = b
        # Some parameters for training    
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Keep the loss and accuracy at every iteration for plotting
        self.train_loss = train_loss
        self.dev_loss = dev_loss
        self.train_acc = train_acc
        self.dev_acc = dev_acc


    def gradient_descent(self, X_Y_train_dev: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                        ) -> Tuple[np.ndarray, np.ndarray]:
        X_train, Y_train, X_dev, Y_dev = X_Y_train_dev # output of PrepareData.loading_entry()
        train_size = X_train.shape[0]
        dev_size = X_dev.shape[0]

        # Calcuate the number of parameter updates
        step = 1

        # Mini-batch training
        mini_bach = int(np.floor(train_size / self.batch_size))
        
        # Iterative training
        for epoch in range(self.max_iter):
            # Random shuffle at the begging of each epoch
            X_train, Y_train = self._shuffle(X_train, Y_train)
            
            for idx in range(mini_bach):
                print(f"Epoch: {epoch} [{idx}/{mini_bach} with batch size {self.batch_size}]", end= '\r')
                mini_bach_index = list(range(idx*self.batch_size, (idx+1)*self.batch_size))
                X = X_train[mini_bach_index]
                Y = Y_train[mini_bach_index]

                # Compute the gradient
                w_grad, b_grad = self._gradient(X, Y, self.w, self.b)
                    
                # gradient descent update
                # learning rate decay with time
                self.w = self.w - self.learning_rate/np.sqrt(step) * w_grad
                self.b = self.b - self.learning_rate/np.sqrt(step) * b_grad

                step = step + 1
                    
            # Compute loss and accuracy of training set and development set
            y_train_pred = self._f(X_train, self.w, self.b)
            Y_train_pred = np.round(y_train_pred)
            self.train_acc.append(self._accuracy(Y_train_pred, Y_train))
            self.train_loss.append(self._cross_entropy_loss(y_train_pred, Y_train) / train_size)

            y_dev_pred = self._f(X_dev, self.w, self.b)
            Y_dev_pred = np.round(y_dev_pred)
            self.dev_acc.append(self._accuracy(Y_dev_pred, Y_dev))
            self.dev_loss.append(self._cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
        print(f"Complete {epoch+1} epochs, each {mini_bach} mini-batch with batch size {self.batch_size}")
        print('Training loss: {}'.format(self.train_loss[-1]))
        print('Development loss: {}'.format(self.dev_loss[-1]))
        print('Training accuracy: {}'.format(self.train_acc[-1]))
        print('Development accuracy: {}'.format(self.dev_acc[-1]))
        return self


class LinearDicriminantAnalysis:
    """LDA approaches the problem by assuming that the conditional probability
    density functions P(X|Y=0) and P(X|Y=1) are both the normal distribution with
    mean and covariance (mean_0, cov_0), (mean_1, cov_1). Moreover, it
    makes the additional simplifying homoscedasticity assumption (i.e. that the class covariances are identical)
    
    Reference: [LDA in wiki](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
    """
    @staticmethod
    def LDA_generative_model(X_train, Y_train, feature_dim):
        # Compute in-class mean
        X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
        X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])
        mean_0 = np.mean(X_train_0, axis = 0)
        mean_1 = np.mean(X_train_1, axis = 0)  

        # Compute in-class covariance
        cov_0 = np.zeros((feature_dim, feature_dim))
        cov_1 = np.zeros((feature_dim, feature_dim))
        for x in X_train_0:
            cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
        for x in X_train_1:
            cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

        # Shared covariance is taken as a weighted average of individual in-class covariance.
        cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

        # Compute inverse of covariance matrix.
        # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
        # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
        u, s, v = np.linalg.svd(cov, full_matrices=False)
        inv = np.matmul(v.T * 1 / s, u.T)

        # Directly compute weights and bias
        w = np.dot(inv, mean_0 - mean_1)
        b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
            + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0]) 

        # Compute accuracy on training set
        Y_train_pred = 1 - UsageFunction()._predict(X_train, w, b) # since it predict P(Y=0|X) with ">0.5" = 1, otherwise = 0.
        print('Training accuracy: {}'.format(UsageFunction()._accuracy(Y_train_pred, Y_train)))
        return w, b


# Print out the most significant weights
def print_significant_weight(w, file_path):
    ind = np.argsort(np.abs(w))[::-1]
    with open(os.path.join(file_path, 'X_test')) as f:
        content = f.readline().strip('\n').split(',')
    features = np.array(content)
    for i in ind[0:10]:
        print(features[i], w[i])


if __name__ == "__main__":
    
    np.random.seed(0) 

    # Preparing data
    preparedata = PrepareData(file_path=r"E:/Download/dataset/data/")
    X_train, Y_train, X_dev, Y_dev, X_test = preparedata.loading_entry()
    feature_dim = X_train.shape[1]

    # Logistic Regression with gradient descent
    logisticregression = LogisticRegression(w = np.zeros((feature_dim,)),
                                            b = np.zeros((1,)))
    logisticregression.gradient_descent(X_Y_train_dev = (X_train, Y_train, X_dev, Y_dev))

    # print("attributes", [ele for ele in dir(logisticregression) if not '__' in ele])
    w = logisticregression.w
    b = logisticregression.b
    predictions = UsageFunction()._predict(X_test, w, b)
    print_significant_weight(w, preparedata.file_path)

    # Plotting loss and accuracy curve
    import matplotlib.pyplot as plt
    plt.plot(logisticregression.train_loss)
    plt.plot(logisticregression.dev_loss)
    plt.title("Loss")
    plt.legend(['train', 'dev'])
    # plt.savefig('loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(logisticregression.train_acc)
    plt.plot(logisticregression.dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    # plt.savefig('acc.png')
    plt.show()


    # LDA with closed-form sample mean and covariance matrix
    print("Start LDA...")
    w_lda, b_lda = LinearDicriminantAnalysis.LDA_generative_model(X_train, Y_train, feature_dim)
    predictions_lda = 1 - UsageFunction()._predict(X_test, w_lda, b_lda) # 1 - (mark of Y=0)
    print_significant_weight(w_lda, preparedata.file_path)


    print("All done.")