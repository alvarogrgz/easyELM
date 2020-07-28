"""
Extreme Learning Machine implementation
"""

# Authors: Álvaro García González <alvarogrgz@gmail.com>
# License: MIT

import numpy as np
from scipy.linalg import pinv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import r2_score


class _BaseELM:
    """Base class for ELM classification and regression.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(self, hidden_layer_neurons=100, activation='sigmoid', random_state=None):
        if hidden_layer_neurons < 1:
            raise ValueError("hiddenSize must be greater than 0")
        self.hidden_layer_neurons = hidden_layer_neurons
        self.activationName = activation
        self.activation = self._activationFunction(self.activationName)
        self.random_state = random_state
        self.isFitted = False

    def _identity(self, Z):
        return Z

    def _sigmoid(self, Z):
        # Avoid overflow
        Z = np.clip(Z, -709, 709)
        return 1 / (1 + np.exp(-Z))

    def _relu(self, Z):
        return np.maximum(Z, 0, Z)

    def _tanh(self, Z):
        return np.tanh(Z)

    def _activationFunction(self, functionName):
        if functionName == 'sigmoid':
            return self._sigmoid
        if functionName == 'relu':
            return self._relu
        if functionName == 'tanh':
            return self._tanh
        if functionName == 'identity':
            return self._identity
        else:
            raise ValueError("Not a valid activation function")

    def _hidden_layer_output(self, x):
        A = np.dot(x, self.weight) + self.bias
        A = self.activation(A)
        return A

    def _check_is_fitted(self):
        if self.isFitted:
            return self
        else:
            raise Exception(
                "This model is not fitted yet. Call 'fit' with appropriate arguments before using this model.")

    def _fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_samples, self.n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        # Set the Random seed
        np.random.seed(self.random_state)
        # Initialize weights
        self.weight = np.random.normal(size=[self.n_features, self.hidden_layer_neurons])
        # Initialize bias
        self.bias = np.random.normal(size=self.hidden_layer_neurons)
        # Calculate hidden layer output matrix (Hinit)
        H = self._hidden_layer_output(X)
        # Calculate the Moore-Penrose pseudoinverse matriks
        H_moore_penrose = pinv2(H)
        # Calculate the output weight matrix beta
        self.beta = np.dot(H_moore_penrose, y)
        self.isFitted = True
        return self

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : returns a trained ELM model.
        """
        return self._fit(X, y)

    def _predict(self, X):
        """Predict using the trained model
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The decision function of the samples for each class in the model.
        """
        y_pred = self._hidden_layer_output(X)
        y_pred = np.dot(y_pred, self.beta)
        return y_pred

    def get_params(self):
        """
            Get parameters for this model.
            Returns
            -------
            params : mapping of string to any
                Parameter names mapped to their values.
            """
        out = {
            'hidden_layer_neurons': self.hidden_layer_neurons,
            'activation': self.activationName,
            'random_state': self.random_state
        }
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if key == 'hidden_layer_neurons':
                self.hidden_layer_neurons = value
            elif key == 'activation':
                self.activationName = value
                self.activation = self._activationFunction(value)
            elif key == 'random_state':
                self.random_state = value
            self.isFitted = False

        return self


class ELMRegressor(_BaseELM):
    """Extreme Learning machine regressor.
    Parameters
    ----------
    hidden_layer_neurons : int, default=100
        The number of neurons in the hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='sigmoid'
        Activation function for the hidden layer.
        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x
        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias.
        Pass an int for reproducible results across multiple function calls.

    References
    ----------
    G. Bin Huang, Q. Y. Zhu, and C. K. Siew,
    “Extreme learning machine: Theory and applications,”
    Neurocomputing, vol. 70, no. 1–3, pp. 489–501, 2006, doi: 10.1016/j.neucom.2005.12.126
    """

    def __init__(self, hidden_layer_neurons=100, activation='sigmoid', random_state=None):
        super().__init__(hidden_layer_neurons=hidden_layer_neurons, activation=activation, random_state=random_state)

    def predict(self, X):
        """Predict using the extreme learning machine model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        self._check_is_fitted()
        y_pred = self._predict(X)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The test data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class ELMCLassifier(_BaseELM):
    """Extreme Learning machine classifier.
    Parameters
    ----------
    hidden_layer_neurons : int, default=100
        The number of neurons in the hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='sigmoid'
        Activation function for the hidden layer.
        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x
        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias.
        Pass an int for reproducible results across multiple function calls.

    ----------
    G. Bin Huang, Q. Y. Zhu, and C. K. Siew,
    “Extreme learning machine: Theory and applications,”
    Neurocomputing, vol. 70, no. 1–3, pp. 489–501, 2006, doi: 10.1016/j.neucom.2005.12.126
    """

    def __init__(self, hidden_layer_neurons=100, activation='sigmoid', random_state=None):
        super().__init__(hidden_layer_neurons=hidden_layer_neurons, activation=activation, random_state=random_state)
        self._label_binarizer = LabelBinarizer()

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            Test samples.
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : returns a trained ELM model.
        """
        y = self._label_binarizer.fit_transform(y)
        return self._fit(X, y)

    def predict(self, X):
        """Predict using the extreme learning machine classifier
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : ndarray, shape (n_samp  les,) or (n_samples, n_classes)
            The predicted classes.
        """
        self._check_is_fitted()
        y_pred = self._predict(X)

        return self._label_binarizer.inverse_transform(y_pred)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def predict_log_proba(self, X):
        """Return the log of probability estimates.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        log_y_prob : ndarray of shape (n_samples, n_classes)
            The predicted log-probability of the sample for each class
            in the model. Equivalent to log(predict_proba(X))
        """
        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)

    def predict_proba(self, X):
        """Probability estimates.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model
        """
        self._check_is_fitted()
        y_pred = self._predict(X)

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred
