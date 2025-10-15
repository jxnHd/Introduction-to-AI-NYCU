import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import tqdm


class MatMul(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = nn.Parameter(torch.tensor(W, dtype=torch.float32))

    def forward(self, x):
        return x @ self.W


class Bias(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        return x + self.b


class ReLU(nn.Module):
    def forward(self, x):
        return torch.clip(x, 0, None)


class Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class CrossEntropy(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.nn.functional.nll_loss(
            torch.log(y_pred), y_true, reduction='none')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n', default=100, type=int, help='number of propagations')
    parser.add_argument('--input_shape', default=28 * 28, type=int)
    parser.add_argument('--output_shape', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--output_dir', default='./grads_pt', type=str)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)

    # model initialization
    model = nn.Sequential()
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
            x = torch.tensor(x, dtype=torch.float32)
            y_true = np.random.randint(0, args.output_shape, args.batch_size)
            y_true = torch.tensor(y_true, dtype=torch.int64)
            # forward
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            loss_mean = torch.sum(loss) / args.batch_size
            pbar.write(f"iter: {iter:02d}, loss: {loss_mean.item():.5f}")

            # backward
            model.zero_grad()
            loss_mean.backward()

            # collect gradients
            grads = []
            for param in model.parameters():
                grads.append(param.grad.numpy())
            # save gradients
            with open(f"{args.output_dir}/iter{iter:02d}.pkl", "wb") as f:
                pickle.dump(grads, f)


if __name__ == "__main__":
    main()
