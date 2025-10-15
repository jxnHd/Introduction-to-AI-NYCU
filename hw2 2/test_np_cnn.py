import argparse
import os
import pickle

import numpy as np
import tqdm

from layers import Sequential, ReLU, Softmax, MatMul, Bias
from losses import CrossEntropy
from extra_layers_np import Conv2D, MaxPool2D, Flatten, Reshape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n', default=50, type=int, help='number of propagations')
    parser.add_argument('--input_shape', default='(1,28,28)', type=str, help='CHW for single sample')
    parser.add_argument('--output_shape', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--output_dir', default='./grads_np_cnn', type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # parse input shape string like "(1,28,28)"
    C, H, W = eval(args.input_shape)

    # model: reshape flat -> NCHW -> Conv -> ReLU -> Pool -> Flatten -> FC -> Softmax
    model = Sequential(
        Reshape((C, H, W)),
        Conv2D(
            W=np.random.randn(8, C, 3, 3).astype(np.float32) * 0.1,
            b=np.random.randn(8).astype(np.float32) * 0.1,
            stride=1,
            padding=1,
        ),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Conv2D(
            W=np.random.randn(16, 8, 3, 3).astype(np.float32) * 0.1,
            b=np.random.randn(16).astype(np.float32) * 0.1,
            stride=1,
            padding=1,
        ),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Flatten(),
        MatMul(np.random.randn((H // 4) * (W // 4) * 16, 64).astype(np.float32) * 0.1),
        Bias(np.random.randn(64).astype(np.float32) * 0.1),
        ReLU(),
        MatMul(np.random.randn(64, args.output_shape).astype(np.float32) * 0.1),
        Bias(np.random.randn(args.output_shape).astype(np.float32) * 0.1),
        Softmax(),
    )

    loss_fn = CrossEntropy()

    os.makedirs(args.output_dir, exist_ok=True)

    with tqdm.trange(args.n) as pbar:
        for iter in pbar:
            # generate random flat input (N, C*H*W)
            x = np.random.rand(args.batch_size, C * H * W).astype(np.float32)
            y_true = np.random.randint(0, args.output_shape, args.batch_size)

            y_pred = model.forward(x)
            loss = loss_fn.forward(y_pred, y_true)
            loss_mean = np.sum(loss) / args.batch_size
            pbar.write(f"iter: {iter:02d}, loss: {loss_mean:.5f}")

            grad = np.ones_like(loss) / args.batch_size
            model.backward(loss_fn.backward(grad))

            with open(f"{args.output_dir}/iter{iter:02d}.pkl", "wb") as f:
                pickle.dump(model.grads(), f)


if __name__ == "__main__":
    main()


