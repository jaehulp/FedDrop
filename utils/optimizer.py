import torch.optim as optim

def load_optimizer(args, model):
    
    if args.optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer_name == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=args.beta, weight_decay=args.weight_decay)
    
    return optimizer
