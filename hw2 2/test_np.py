import argparse
import os
import pickle

import numpy as np
import tqdm

from losses import CrossEntropy
from layers import Sequential, ReLU, Softmax, MatMul, Bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n', default=100, type=int, help='number of propagations')
    parser.add_argument('--input_shape', default=28 * 28, type=int)
    parser.add_argument('--output_shape', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--output_dir', default='./grads_np', type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # model initialization
    model = Sequential()
    model.append(MatMul(np.random.randn(args.input_shape, 128) * 0.1))
    model.append(Bias(np.random.randn(128) * 0.1))
    model.append(ReLU())
    model.append(MatMul(np.random.randn(128, 128) * 0.1))
    model.append(Bias(np.random.randn(128) * 0.1))
    model.append(ReLU())
    model.append(MatMul(np.random.randn(128, args.output_shape) * 0.1))
    model.append(Bias(np.random.randn(args.output_shape) * 0.1))
    model.append(Softmax())

    # loss function
    loss_fn = CrossEntropy()

    # create directory to save gradients
    os.makedirs(args.output_dir, exist_ok=True)

    # test backpropagation
    with tqdm.trange(args.n) as pbar:
        for iter in pbar:
            # generate random data
            x = np.random.rand(args.batch_size, args.input_shape)
            y_true = np.random.randint(0, args.output_shape, args.batch_size)
            # forward
            y_pred = model.forward(x)
            loss = loss_fn.forward(y_pred, y_true)
            loss_mean = np.sum(loss) / args.batch_size
            pbar.write(f"iter: {iter:02d}, loss: {loss_mean:.5f}")

            # backward
            grad = np.ones_like(loss) / args.batch_size
            model.backward(loss_fn.backward(grad))

            # save gradients
            with open(f"{args.output_dir}/iter{iter:02d}.pkl", "wb") as f:
                pickle.dump(model.grads(), f)


if __name__ == "__main__":
    main()
