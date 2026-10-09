import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    m, n = X.shape
    W = np.zeros(n)
    b = 0

    for i in range(steps):
        y_pred = _sigmoid(X.dot(W) + b)

        dw = (X.T.dot(y_pred - y))/m
        db = np.mean(y_pred - y)

        W -= lr * dw
        b -= lr * db

    return W, b