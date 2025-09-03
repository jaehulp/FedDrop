import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--clients', default=50, type=int)
parser.add_argument('--split', default='iid', type=str)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--seed', default=999, type=int)


args, _ = parser.parse_known_args()

print(args.dataset)