class LinearRegression:
    def __init__(self, method='sgd', lr=0.1):
        """
        :param method: 'sgd' for SGD, 'gd' for GD, 'analytic' for analytic solution
        "param lr: learning rate
        """
        self._lr = lr

        if method not in ('analytic', 'gd', 'sgd'):
            raise ValueError('method can be only "gd", "sgd" or "analytic"')

        self._method = method
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
        Calculate the gradient of the loss function.
        If x is a vector, calculate only for this data (for SGD), else for whole dataset (for GD)
        """

    def fit(self, x, y):
        """
        TODO: normalize the data and fit the linear regression.
        :param x: features matrix
        :param y: labels
        :returns: None if can't fit, weights, if fitted.
        """
        self._weights = None  # TODO: initialize weights here
        self._x = x
        self._y = y

        if self._method == 'sgd':
            pass
        elif self._method == 'gd':
            pass
        elif self._method == 'analytic':
            pass
        else:
            raise ValueError('method can be only "gd", "sgd" or "analytic"')

        return self._weights

    def predict(self, x):
        """
        TODO: Calculate the predictions for each data in features matrix.
        :param x: features matrix
        """

