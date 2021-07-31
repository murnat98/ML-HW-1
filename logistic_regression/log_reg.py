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

    def gradient(self, x, y):
        """
        TODO: Compute the gradient of the loss function.
        """

    def sigmoid(self, x):
        """
        :param x: feature matrix.
        :returns: sigmoid vector of all features.
        TODO: Add ones to feature matrix and compute sigmoid.
        """
        if self._weights is None:
            raise ValueError('All methods can be called after fit method is called.')

    def fit(self, x, y):
        """
        TODO: normalize the data and fit the logistic regression.
        :param x: features matrix
        :param y: labels
        :returns: None if can't fit, weights, if fitted.
        """
        self._weights = None  # TODO: initialize weights here
        self._x = x
        self._y = y

        # TODO: SGD here

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

