from models.cnn import CNN2
from models.resnet import ResNet
from models.mlp import MLP

def load_model(args):

    model_name = args.model_name

    if model_name == 'CNN2':
        model = CNN2(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
    elif model_name == 'ResNet':
        model = ResNet(args.block, args.num_blocks, num_classes=args.num_classes)
    elif model_name == "MLP":
        model = MLP(
            input_dim=args.input_dim,
            output_dim=args.output_dim
        )
    else:
        raise NotImplementedError('Not Implemented Model name. Check ./utils/model.py')

    return model