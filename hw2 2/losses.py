import numpy as np
import abc


class Loss(abc.ABC):
    @abc.abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the loss between the predicted and true values.

        Parameters:
            y_pred (`np.ndarray`):
                The predicted values, typically output from a model. The shape
                of `y_pred` is (`batch_size`, `input_dim`).
            y_true (`np.ndarray`):
                The true/target values corresponding to the predictions. Note
                that the shapes of `y_pred` and `y_true` might not match, but
                they will have the same number of samples, i.e. `batch_size`.

        Returns:
            loss (`np.ndarray`):
                The loss value for each sample in the input. The shape of the
                output is (`batch_size`,).
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. the predicted values.

        Parameters:
            grad (`np.ndarray`):
                The gradient of the final loss with respect to the output of
                `self.forward`. This value typically reflects a scaling factor
                caused by averaging the loss, and can also be used to reverse
                the gradient direction to achieve gradient ascent.

        Returns:
            grad (`np.ndarray`):
                The gradient of the loss with respect to the `y_pred`.
        """
        return NotImplemented


class MeanSquaredError(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_true = y_true.reshape(y_true.shape[0], -1)
        self.cache = (y_pred, y_true)
        loss = np.mean((y_pred - y_true) ** 2, axis=1)
        return loss

    def backward(self, grad: np.ndarray) -> np.ndarray:
        y_pred, y_true = self.cache
        N = y_pred.shape[1]
        return grad[:, None] * 2 / N * (y_pred - y_true)


class CrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the cross-entropy loss between the predicted probabilities and
        the true labels.

        Parameters:
            y_pred (`np.ndarray`):
                A 2D array of predicted probabilities, where the first axis
                corresponds to the number of samples and the second axis
                corresponds to the number of classes.
            y_true (`np.ndarray`):
                A 1D array of true labels, where each label is an integer in
                the range [0, num_classes).
        """
        # y_pred: (N, C) probabilities; y_true: (N,)
        # numerical stability: clip probabilities
        y_pred = y_pred.astype(np.float32,copy=False)
        eps = np.finfo(y_pred.dtype).tiny
        y_pred = np.clip(y_pred, eps, 1.0)
        self.cache = (y_pred, y_true)
        # pick probabilities at true labels
        p_true = y_pred[np.arange(y_pred.shape[0]), y_true]
        return -np.log(p_true)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        y_pred, y_true = self.cache
        N, C = y_pred.shape
        # dL/dy_pred for cross-entropy w.r.t probabilities: -1/p_true for true class, 0 otherwise
        grad_logits = np.zeros_like(y_pred)
        grad_logits[np.arange(N), y_true] = -1.0 / y_pred[np.arange(N), y_true]
        # chain with incoming grad per-sample
        return grad[:, None] * grad_logits
