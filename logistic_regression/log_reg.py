import numpy as np


class LogisticRegression:
    """
    Implement logistic regression with SGD.
    """
    def __init__(self, lr=0.1):
        """
        :param lr: learning rate
        """
        self._lr = lr

        self._x = None
        self._y = None

    def loss_function(self):
        """
        TODO: calculate the loss function
        """
        if self._x is None or self._y is None:
            raise ValueError('All methods can be called after fit method is called.')
           
        loss = np.sum(self._y * np.log(self.sigmoid(self._x)) + (1 - self._y) * np.log(1 - self.sigmoid(self._x)))

        return -loss

    def gradient(self, x, y):
        """
        Compute the gradient of the loss function.
        """
        x_vector = x.reshape(-1, 1).T
        x = np.insert(1, 1, x)
        return -(y - self.sigmoid(x_vector)[0]) * x

    def sigmoid(self, x):
        """
        :param x: feature matrix.
        :returns: sigmoid vector of all features.
        TODO: Add ones to feature matrix and compute sigmoid.
        """
        if self._weights is None:
            raise ValueError('All methods can be called after fit method is called.')

        ones = np.ones(shape=(x.shape[0],)).reshape(-1, 1)
        x = np.append(ones, x, axis=1)

        return 1 / (1 + np.exp(-x @ self._weights))

    def fit(self, x, y):
        """
        TODO: normalize the data and fit the logistic regression.
        :param x: features matrix
        :param y: labels
        :returns: None if can't fit, weights, if fitted.
        """
        self._x = x
        self._y = y
        self._weights = np.zeros(self._x.shape[1] + 1) 

        loss = self.loss_function()
        while loss > 282:
            i = np.random.randint(0, self._x.shape[0] - 1)
            gradient = self.gradient(self._x[i], self._y[i])
            self._weights -= self._lr * gradient
            loss = self.loss_function()

        return self._weights

    def predict(self, x, threshold=.5):
        """
        Predict which class is each data in x
        :param x: features matrix
        """
        return np.where(self.predict_proba(x) >= threshold, 1, 0)

    def predict_proba(self, x):
        """
        Predict the probability, that x is of class 1.
        """
        return self.sigmoid(x)

