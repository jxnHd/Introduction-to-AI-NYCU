import argparse
import os
import pickle

import numpy as np
import tqdm

from layers import Sequential, ReLU, Softmax, MatMul, Bias
from losses import CrossEntropy
from extra_layers_np import SimpleRNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n', default=50, type=int, help='number of propagations')
    parser.add_argument('--timesteps', default=10, type=int)
    parser.add_argument('--input_dim', default=16, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--output_shape', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--output_dir', default='./grads_np_rnn', type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # RNN -> FC -> Softmax
    W_xh = (np.random.randn(args.input_dim, args.hidden_dim) * 0.1).astype(np.float32)
    W_hh = (np.random.randn(args.hidden_dim, args.hidden_dim) * 0.1).astype(np.float32)
    b_h = (np.random.randn(args.hidden_dim) * 0.1).astype(np.float32)

    model = Sequential(
        SimpleRNN(W_xh, W_hh, b_h),
        MatMul(np.random.randn(args.hidden_dim, 64).astype(np.float32) * 0.1),
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
            x = np.random.rand(args.batch_size, args.timesteps, args.input_dim).astype(np.float32)
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


