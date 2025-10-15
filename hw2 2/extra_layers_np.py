import numpy as np
from typing import List, Tuple

# 使用現有 Layer/Sequential 介面，但不修改原檔
from layers import Layer


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def im2col_indices(x: np.ndarray, kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """Convert NCHW image batch to column matrix for 2D conv/pool.

    Returns (cols, out_h, out_w)
    - cols: shape (N*out_h*out_w, C*KH*KW)
    """
    N, C, H, W = x.shape
    KH, KW = _pair(kernel_size)
    SH, SW = _pair(stride)
    PH, PW = _pair(padding)

    H_p = H + 2 * PH
    W_p = W + 2 * PW
    out_h = (H_p - KH) // SH + 1
    out_w = (W_p - KW) // SW + 1

    x_padded = np.pad(x, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode="constant")

    cols = np.empty((N * out_h * out_w, C * KH * KW), dtype=x.dtype)

    row = 0
    for i in range(out_h):
        i_start = i * SH
        i_end = i_start + KH
        for j in range(out_w):
            j_start = j * SW
            j_end = j_start + KW
            window = x_padded[:, :, i_start:i_end, j_start:j_end]  # (N, C, KH, KW)
            cols[row * N:(row + 1) * N] = window.reshape(N, -1)
            row += 1

    return cols, out_h, out_w


def col2im_indices(cols: np.ndarray, x_shape: Tuple[int, int, int, int], kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int], out_h: int, out_w: int) -> np.ndarray:
    """Inverse of im2col_indices.

    cols: (N*out_h*out_w, C*KH*KW)
    returns: (N, C, H, W)
    """
    N, C, H, W = x_shape
    KH, KW = _pair(kernel_size)
    SH, SW = _pair(stride)
    PH, PW = _pair(padding)

    H_p = H + 2 * PH
    W_p = W + 2 * PW

    x_padded = np.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    row = 0
    for i in range(out_h):
        i_start = i * SH
        i_end = i_start + KH
        for j in range(out_w):
            j_start = j * SW
            j_end = j_start + KW
            window = cols[row * N:(row + 1) * N].reshape(N, C, KH, KW)
            x_padded[:, :, i_start:i_end, j_start:j_end] += window
            row += 1

    if PH == 0 and PW == 0:
        return x_padded
    return x_padded[:, :, PH:H_p - PH, PW:W_p - PW]


class Reshape(Layer):
    """重塑張量形狀（僅改變形狀，不含參數）。

    new_shape: 目標形狀（不含 batch 維度）。
    輸入: (N, D...) -> 輸出: (N, *new_shape)
    """
    def __init__(self, new_shape: Tuple[int, ...]):
        self.new_shape = tuple(new_shape)
        self._cache_in_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        self._cache_in_shape = x.shape
        N = x.shape[0]
        return x.reshape((N,) + self.new_shape)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self._cache_in_shape)


class Flatten(Layer):
    """展平除了 batch 維度以外的所有維度。"""
    def __init__(self):
        self._cache_in_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        self._cache_in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self._cache_in_shape)


class Conv2D(Layer):
    """簡單 2D 卷積層（NCHW），使用 im2col 實作。

    參數:
      W: (C_out, C_in, KH, KW)
      b: (C_out,)
      stride: int 或 (sh, sw)
      padding: int 或 (ph, pw) － zero padding
    輸入: (N, C_in, H, W)
    輸出: (N, C_out, OH, OW)
    """
    def __init__(self, W: np.ndarray, b: np.ndarray, stride: int | Tuple[int, int] = 1, padding: int | Tuple[int, int] = 0):
        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        N, C_in, H, W = x.shape
        C_out, C_in_w, KH, KW = self.W.shape
        assert C_in == C_in_w, "Conv2D: input channels mismatch with weight"

        X_col, out_h, out_w = im2col_indices(x, (KH, KW), self.stride, self.padding)
        W_col = self.W.reshape(C_out, -1).T  # (C_in*KH*KW, C_out)

        out_col = X_col @ W_col + self.b
        out = out_col.reshape(out_h * out_w, N, C_out).transpose(1, 2, 0)
        out = out.reshape(N, C_out, out_h, out_w)

        self._cache = (x.shape, X_col, W_col, out_h, out_w)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        (x_shape, X_col, W_col, out_h, out_w) = self._cache
        N, C_in, H, W = x_shape
        C_out = W_col.shape[1]
        KH_KW_Cin = W_col.shape[0]

        grad_col = grad.reshape(N, C_out, out_h * out_w).transpose(2, 0, 1).reshape(out_h * out_w * N, C_out)

        # dW
        dW_col = X_col.T @ grad_col  # (C_in*KH*KW, C_out)
        dW = dW_col.T.reshape(self.W.shape)

        # db
        db = np.sum(grad_col, axis=0)

        # dX
        dX_col = grad_col @ W_col.T  # (N*out_h*out_w, C_in*KH*KW)
        dX = col2im_indices(dX_col, x_shape, self.W.shape[2:], self.stride, self.padding, out_h, out_w)

        self._grad_W = dW
        self._grad_b = db
        return dX

    def params(self) -> List[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> List[np.ndarray]:
        return [getattr(self, "_grad_W", np.zeros_like(self.W)), getattr(self, "_grad_b", np.zeros_like(self.b))]


class MaxPool2D(Layer):
    """最大池化（NCHW）。"""
    def __init__(self, kernel_size: int | Tuple[int, int] = 2, stride: int | Tuple[int, int] | None = None):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride

        # 對每個通道獨立池化：視為 (N*C, 1, H, W)
        x_ = x.reshape(N * C, 1, H, W)
        X_col, out_h, out_w = im2col_indices(x_, (KH, KW), (SH, SW), (0, 0))
        # 在窗口維度上取最大值
        max_idx = np.argmax(X_col, axis=1)
        out_flat = X_col[np.arange(X_col.shape[0]), max_idx]
        out = out_flat.reshape(out_h * out_w, N, C).transpose(1, 2, 0).reshape(N, C, out_h, out_w)

        self._cache = (x.shape, X_col, max_idx, out_h, out_w)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        (x_shape, X_col, max_idx, out_h, out_w) = self._cache
        N, C, H, W = x_shape
        KH, KW = self.kernel_size
        SH, SW = self.stride

        grad_flat = grad.reshape(N, C, out_h * out_w).transpose(2, 0, 1).reshape(out_h * out_w * N * C)

        dX_col = np.zeros_like(X_col)
        dX_col[np.arange(X_col.shape[0]), max_idx] = grad_flat

        dX = col2im_indices(dX_col, (N * C, 1, H, W), (KH, KW), (SH, SW), (0, 0), out_h, out_w)
        dX = dX.reshape(N, C, H, W)
        return dX


class SimpleRNN(Layer):
    """簡單 RNN（tanh）。

    輸入: (N, T, D)
    輸出: (N, H) － 最末時間步的隱狀態
    參數:
      W_xh: (D, H)
      W_hh: (H, H)
      b_h: (H,)
    """
    def __init__(self, W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray, h0: np.ndarray | None = None):
        self.W_xh = W_xh.astype(np.float32)
        self.W_hh = W_hh.astype(np.float32)
        self.b_h = b_h.astype(np.float32)
        self.h0 = None if h0 is None else h0.astype(np.float32)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        N, T, D = x.shape
        H = self.W_hh.shape[0]
        hs = np.zeros((T + 1, N, H), dtype=np.float32)
        if self.h0 is not None:
            hs[0] = self.h0
        for t in range(T):
            hs[t + 1] = np.tanh(x[:, t] @ self.W_xh + hs[t] @ self.W_hh + self.b_h)
        self._cache = (x, hs[1:], hs[:-1])  # (x, h[1..T], h[0..T-1])
        return hs[T]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x, h, h_prev = self._cache
        N, T, D = x.shape
        H = h.shape[2]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dx = np.zeros_like(x)
        dh_next = grad  # from output (N, H)

        for t in reversed(range(T)):
            # derivative of tanh
            dt = (1.0 - h[t] ** 2) * dh_next  # (N, H)
            dW_xh += x[:, t].T @ dt  # (D, H)
            dW_hh += h_prev[t].T @ dt  # (H, H)
            db_h += np.sum(dt, axis=0)
            dx[:, t] = dt @ self.W_xh.T
            dh_next = dt @ self.W_hh.T

        self._grad_W_xh = dW_xh
        self._grad_W_hh = dW_hh
        self._grad_b_h = db_h
        return dx

    def params(self) -> List[np.ndarray]:
        return [self.W_xh, self.W_hh, self.b_h]

    def grads(self) -> List[np.ndarray]:
        return [
            getattr(self, "_grad_W_xh", np.zeros_like(self.W_xh)),
            getattr(self, "_grad_W_hh", np.zeros_like(self.W_hh)),
            getattr(self, "_grad_b_h", np.zeros_like(self.b_h)),
        ]


