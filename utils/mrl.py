import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Original code github. @RAIVNLab
https://github.com/RAIVNLab/MRL
"""


class TopKOutputWrapper(nn.Module):
    def __init__(self, model, classifier_layer_name, k=100, alpha=0.5):
        """
        model: pretrained backbone + classifier model
        classifier_layer_name: str, name of the classifier attribute in the model
        k: number of top-k features
        alpha: weight for original output
        """
        super().__init__()
        self.model = model
        self.k = k
        self.alpha = alpha
        self.classifier_layer_name = classifier_layer_name
        
        # Extract classifier layer
        self.classifier = getattr(self.model, self.classifier_layer_name)

        # Placeholder for features
        self.features = None

    def set_device(self, device):
        self.device = device

        def hook(module, input, output):
            # ensure feature is on correct device explicitly
            self.features = input[0].to(self.device)

        self.handle = self.classifier.register_forward_hook(hook)

    def forward(self, x):
        # Compute original output (this populates self.features via hook)
        x = x.to(self.device)
        original_output = self.model(x)

        if self.training:
            # Clone the features
            topk_features = self.features.clone()

            # Select top-k features per sample
            topk_vals, topk_indices = torch.topk(topk_features, self.k, dim=1)
            mask = torch.zeros_like(topk_features, device=self.device)
            mask.scatter_(1, topk_indices, 1.0)
            topk_features = topk_features * mask

            # Compute top-k output by directly applying classifier
            topk_output = self.classifier(topk_features)

            # Weighted sum of outputs
            combined_output = (1-self.alpha) * original_output + self.alpha * topk_output
            return combined_output
        else:
            # Inference: return only original output
            return original_output

    def unwrap(self):
        self.handle.remove()
        return self.model

from collections import Counter

class PrefixTopKOutputWrapper(TopKOutputWrapper):
    def __init__(self, model, classifier_layer_name, k=100, alpha=0.5):
        """
        model: pretrained backbone + classifier model
        classifier_layer_name: str, name of the classifier attribute in the model
        k: number of top-k features
        alpha: weight for original output
        """
        super().__init__(model, classifier_layer_name, k=k, alpha=alpha)

    def set_topk_indices(self, train_loader):
        self.model.eval()
        feature_counts = torch.zeros(512, device=self.device)

        for i, (xs, ys) in enumerate(train_loader):
            xs = xs.to(self.device)
            _  = self.model(xs)

            features = self.features

            topk_indices = torch.topk(features, self.k, dim=1).indices  # [B, k]

            for i in range(topk_indices.shape[0]):
                feature_counts.scatter_add_(
                    0,
                    topk_indices[i],
                    torch.ones_like(topk_indices[i], dtype=feature_counts.dtype)
                )

        _, global_topk_indices = torch.topk(feature_counts, self.k)
        
        mask = torch.zeros_like(feature_counts, device=self.device)
        mask.scatter_(0, global_topk_indices, 1.0)
        mask = mask.unsqueeze(dim=0)
        self.mask = mask

    def forward(self, x):
        x = x.to(self.device)
        original_output = self.model(x)

        if self.training:
            topk_features = self.features.clone()
            topk_features = topk_features * self.mask

            topk_output = self.classifier(topk_features)

            combined_output = (1-self.alpha) * original_output + self.alpha * topk_output
            return combined_output
        else:
            return original_output

class RandPrefixTopKOutputWrapper(TopKOutputWrapper):
    def __init__(self, model, classifier_layer_name, k=100, alpha=0.5):
        """
        model: pretrained backbone + classifier model
        classifier_layer_name: str, name of the classifier attribute in the model
        k: number of top-k features
        alpha: weight for original output
        """
        super().__init__(model, classifier_layer_name, k=k, alpha=alpha)

    def set_topk_indices(self, train_loader):
        self.model.eval()
        feature_counts = torch.zeros(512, device=self.device)

        for i, (xs, ys) in enumerate(train_loader):
            xs = xs.to(self.device)
            _  = self.model(xs)

            features = self.features

            topk_indices = torch.topk(features, self.k, dim=1).indices  # [B, k]

            for i in range(topk_indices.shape[0]):
                feature_counts.scatter_add_(
                    0,
                    topk_indices[i],
                    torch.ones_like(topk_indices[i], dtype=feature_counts.dtype)
                )

        _, global_topk_indices = torch.topk(feature_counts, self.k)
        
        mask = torch.zeros_like(feature_counts, device=self.device)
        mask.scatter_(0, global_topk_indices, 1.0)
        mask = mask.unsqueeze(dim=0)
        self.mask = mask

    def forward(self, x):
        x = x.to(self.device)
        original_output = self.model(x)

        if self.training:
            topk_features = self.features.clone()
            topk_features = topk_features * self.mask

            topk_output = self.classifier(topk_features)

            combined_output = self.alpha * original_output + (1 - self.alpha) * topk_output
            return combined_output
        else:
            return original_output