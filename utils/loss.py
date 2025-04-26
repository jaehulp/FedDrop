import torch
import torch.nn as nn

def get_loss(args):

    loss_name = args.loss_name

    if loss_name == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss(label_smoothing=args.get('label_smoothing'))
    else:
        raise NotImplementedError('Not Implemented Loss. Check ./utils/loss.py')

    return loss
    