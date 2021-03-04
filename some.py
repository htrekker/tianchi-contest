import argparse
import torch


def parse_arg():
    parser = argparse.ArgumentParser('Input those hyper-parameters.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=32)
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epoch', type=int)

    return parser.parse_args()


# print(parse_arg().batch_size)

parser = argparse.ArgumentParser('Input those hyper-parameters.')
parser.add_argument('--cuda', type=int, required=False, default=-1)

print(parser.parse_args().cuda)
print(vars(parser.parse_args()))

print(torch.device('cuda:0'))
