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
        return torch.clamp(x, min=0)


class Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class CrossEntropy(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.nn.functional.nll_loss(torch.log(y_pred), y_true, reduction='none')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n', default=50, type=int, help='number of propagations')
    parser.add_argument('--input_shape', default='(1,28,28)', type=str, help='CHW for single sample')
    parser.add_argument('--output_shape', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--output_dir', default='./grads_pt_cnn', type=str)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # parse input shape
    C, H, W = eval(args.input_shape)

    # Build model mirroring numpy version order & shapes
    model = nn.Sequential()
    # Reshape flat -> NCHW is done outside as view
    conv1_W = np.random.randn(8, C, 3, 3).astype(np.float32) * 0.1
    conv1_b = np.random.randn(8).astype(np.float32) * 0.1
    conv2_W = np.random.randn(16, 8, 3, 3).astype(np.float32) * 0.1
    conv2_b = np.random.randn(16).astype(np.float32) * 0.1

    # Create layers
    conv1 = nn.Conv2d(C, 8, kernel_size=3, stride=1, padding=1, bias=True)
    with torch.no_grad():
        conv1.weight.copy_(torch.tensor(conv1_W))
        conv1.bias.copy_(torch.tensor(conv1_b))
    conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True)
    with torch.no_grad():
        conv2.weight.copy_(torch.tensor(conv2_W))
        conv2.bias.copy_(torch.tensor(conv2_b))

    model.append(conv1)
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d(kernel_size=2, stride=2))
    model.append(conv2)
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d(kernel_size=2, stride=2))
    model.append(nn.Flatten())

    # FC layers
    fc1_W = np.random.randn((H // 4) * (W // 4) * 16, 64).astype(np.float32) * 0.1
    fc1_b = np.random.randn(64).astype(np.float32) * 0.1
    fc2_W = np.random.randn(64, args.output_shape).astype(np.float32) * 0.1
    fc2_b = np.random.randn(args.output_shape).astype(np.float32) * 0.1

    model.append(MatMul(fc1_W))
    model.append(Bias(fc1_b))
    model.append(ReLU())
    model.append(MatMul(fc2_W))
    model.append(Bias(fc2_b))
    model.append(Softmax())

    loss_fn = CrossEntropy()

    os.makedirs(args.output_dir, exist_ok=True)

    with tqdm.trange(args.n) as pbar:
        for iter in pbar:
            x = np.random.rand(args.batch_size, C * H * W).astype(np.float32)
            y_true = np.random.randint(0, args.output_shape, args.batch_size)
            x = torch.tensor(x, dtype=torch.float32)
            y_true = torch.tensor(y_true, dtype=torch.int64)

            # Reshape flat -> NCHW
            x = x.view(args.batch_size, C, H, W)

            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            loss_mean = torch.sum(loss) / args.batch_size
            pbar.write(f"iter: {iter:02d}, loss: {loss_mean.item():.5f}")

            model.zero_grad()
            loss_mean.backward()

            grads = []
            for param in model.parameters():
                grads.append(param.grad.detach().cpu().numpy())
            with open(f"{args.output_dir}/iter{iter:02d}.pkl", "wb") as f:
                pickle.dump(grads, f)


if __name__ == "__main__":
    main()


