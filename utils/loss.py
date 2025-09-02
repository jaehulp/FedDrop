import torch
import torch.nn as nn

class matryoshka_loss(nn.Module):

    def __init__(self, **kwargs):
        super(matryoshka_loss, self).__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(**kwargs)


    def forward(self, output, target):

        loss1 = self.criterion(output, target)
        loss2 = self.criterion(output, target)

        weighted_loss = (1-self.alpha) * loss1 + self.alpha * loss2

        return weighted_loss


def get_loss(args):

    loss_name = args.loss_name

    if loss_name == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss(label_smoothing=args.get('label_smoothing'))
    else:
        raise NotImplementedError('Not Implemented Loss. Check ./utils/loss.py')

    return loss
    