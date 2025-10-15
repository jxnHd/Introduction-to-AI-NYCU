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
    parser.add_argument('--timesteps', default=10, type=int)
    parser.add_argument('--input_dim', default=16, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--output_shape', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--output_dir', default='./grads_pt_rnn', type=str)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build model mirroring numpy version order & shapes
    # SimpleRNN with tanh returning last hidden state
    rnn = nn.RNN(input_size=args.input_dim, hidden_size=args.hidden_dim, nonlinearity='tanh', batch_first=True)

    # Initialize RNN weights to match parameter naming
    W_xh = (np.random.randn(args.input_dim, args.hidden_dim) * 0.1).astype(np.float32)
    W_hh = (np.random.randn(args.hidden_dim, args.hidden_dim) * 0.1).astype(np.float32)
    b_h = (np.random.randn(args.hidden_dim) * 0.1).astype(np.float32)

    with torch.no_grad():
        # PyTorch RNN uses weight_ih_l0 (H*I) and weight_hh_l0 (H*H) in row-major
        rnn.weight_ih_l0.copy_(torch.tensor(W_xh.T))  # (H, I)
        rnn.weight_hh_l0.copy_(torch.tensor(W_hh))    # (H, H)
        # bias_ih + bias_hh; we split b across them equally for equivalence
        rnn.bias_ih_l0.copy_(torch.tensor(b_h))
        rnn.bias_hh_l0.copy_(torch.zeros_like(rnn.bias_hh_l0))

    model = nn.Sequential()
    model.append(rnn)
    model.append(nn.Flatten(start_dim=1, end_dim=1))  # keep last hidden (N, H)

    fc1_W = np.random.randn(args.hidden_dim, 64).astype(np.float32) * 0.1
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
            x = np.random.rand(args.batch_size, args.timesteps, args.input_dim).astype(np.float32)
            y_true = np.random.randint(0, args.output_shape, args.batch_size)
            x = torch.tensor(x, dtype=torch.float32)
            y_true = torch.tensor(y_true, dtype=torch.int64)

            y_seq, h_last = model[0](x)  # RNN layer
            # emulate Sequential forward behavior: replace RNN with last hidden state
            y = h_last.squeeze(0)  # (N, H)
            # pass through the rest of the sequential stack
            for layer in list(model.children())[1:]:
                y = layer(y)

            loss = loss_fn(y, y_true)
            loss_mean = torch.sum(loss) / args.batch_size
            pbar.write(f"iter: {iter:02d}, loss: {loss_mean.item():.5f}")

            # Manual backward through the composed graph
            for m in model.modules():
                if hasattr(m, 'zero_grad'):
                    m.zero_grad()  # safe for all
            loss_mean.backward()

            grads = []
            # collect parameters in the same order we added them
            for param in model.parameters():
                grads.append(param.grad.detach().cpu().numpy())
            with open(f"{args.output_dir}/iter{iter:02d}.pkl", "wb") as f:
                pickle.dump(grads, f)


if __name__ == "__main__":
    main()


