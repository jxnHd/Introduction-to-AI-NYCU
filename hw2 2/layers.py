import abc
from typing import List

import numpy as np


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propogate through the layer.

        Parameters:
            x (`np.ndarray`):
                input tensor with shape `(batch_size, input_dim)`.

        Returns:
            out (`np.ndarray`):
                output tensor of the layer with shape
                `(batch_size, output_dim)`.
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propogate through the layer.

        Calculate the gradient of the loss with respect to the parameters of
        this layer and return the gradient with respect to the input of the
        layer.

        Parameters:
            grad (`np.ndarray`):
                gradient with respect to the output of the layer.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of the layer.
        """
        return NotImplemented

    def params(self) -> List[np.ndarray]:
        """Return the parameters of the layer.

        The order of parameters should be the same as the order of gradients
        returned by `self.grads()`.
        """
        return []

    def grads(self) -> List[np.ndarray]:
        """Return the gradients of the layer.

        The order of gradients should be the same as the order of parameters
        returned by `self.params()`.
        """
        return []


class Sequential(Layer):
    """A sequential model that stacks layers in a sequential manner.

    Parameters:
        layers (`List[Layer]`):
            list of layers to be added to the model.
    """
    def __init__(self, *layers: List[Layer]):
        self.layers = []
        self.append(*layers)

    def append(self, *layers: List[Layer]):
        """Add layers to the model"""
        self.layers.extend(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sequentially forward propogation through layers.

        Parameters:
            x (`np.ndarray`):
                input tensor.

        Returns:
            out (`np.ndarray`):
                output tensor of the last layer.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Sequentially backward propogation through layers.

        Parameters:
            grad (`np.ndarray`):
                gradient of loss function with respect to the output of forward
                propogation.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of forward propogation.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> List[np.ndarray]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.params())
        return parameters

    def grads(self) -> List[np.ndarray]:
        gradients = []
        for layer in self.layers:
            gradients.extend(layer.grads())
        return gradients


class MatMul(Layer):
    """Matrix multiplication layer.

    Parameters:
        W (`np.ndarray`):
            weight matrix with shape `(input_dim, output_dim)`.
    """
    def __init__(self, W: np.ndarray):
        self.W = W.astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32,copy=False)
        self.cache_x = x
        return x @ self.W

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache_x
        # dL/dW = x^T @ grad
        self._grad_W = x.T @ grad
        espW = np.finfo(self._grad_W.dtype).tiny
        self._grad_W = self._grad_W + (self._grad_W == 0) * espW
        out = grad @ self.W.T
        eps = np.finfo(out.dtype).tiny
        # dL/dx = grad @ W^T
        return out + (out == 0) * eps

    def params(self) -> List[np.ndarray]:
        return [self.W]

    def grads(self) -> List[np.ndarray]:
        return [getattr(self, "_grad_W", np.zeros_like(self.W))]


class Bias(Layer):
    """Bias layer.

    Parameters:
        b (`np.ndarray`):
            bias vector with shape `(output_dim,)`.
    """
    def __init__(self, b: np.ndarray):
        self.b = b.astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32,copy=False)
        self.cache_shape = x.shape
        return x + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # dL/db = sum over batch
        self._grad_b = np.sum(grad, axis=0)
        epsb = np.finfo(self._grad_b.dtype).tiny
        self._grad_b = self._grad_b + (self._grad_b == 0) * epsb
        # gradient w.r.t input
        eps = np.finfo(grad.dtype).tiny
        return grad + (grad == 0) * eps
    
    def params(self) -> List[np.ndarray]:
        return [self.b]

    def grads(self) -> List[np.ndarray]:
        return [getattr(self, "_grad_b", np.zeros_like(self.b))]


class ReLU(Layer):
    """Rectified Linear Unit."""

    def forward(self, x):
        self.cache = x
        return np.clip(x, 0, None)

    def backward(self, grad):
        x = self.cache
        out = np.where(x > 0, grad, 0)
        eps = np.finfo(out.dtype).tiny
        return out + (out == 0) * eps

class Softmax(Layer):
    def forward(self, x):
        # numerically stable softmax
        x = x.astype(np.float32,copy=False)
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shift)
        sums = np.sum(exp_x, axis=1, keepdims=True)
        self.cache = exp_x / sums
        return self.cache

    def backward(self, grad):
        s = self.cache
        # For each sample i: J^T g = s * (g - (g·s))
        dot = np.sum(grad * s, axis=1, keepdims=True)
        out = s * (grad - dot)
        eps = np.finfo(out.dtype).tiny
        return out + (out == 0) * eps