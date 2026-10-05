import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def build_model(pretrained=True):
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 101)
    return model


def expected_age(logits):
    return (logits.softmax(-1) * torch.arange(101, device=logits.device)).sum(-1)


def gaussian_targets(ages, sigma=2.0):
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    indices = torch.arange(101, device=ages.device, dtype=torch.float32)
    return (-0.5 * ((indices - ages.float().unsqueeze(-1)) / sigma).square()).softmax(-1)


def dldl_loss(logits, ages, sigma=2.0, kl_weight=1.0, l1_weight=1.0):
    # Explicitly KL(target || prediction), matching the execution guide.
    logits = logits.float()
    kl = F.kl_div(logits.log_softmax(-1), gaussian_targets(ages, sigma), reduction='batchmean')
    l1 = F.l1_loss(expected_age(logits), ages.float())
    return kl_weight * kl + l1_weight * l1


class AgeOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('indices', torch.arange(101, dtype=torch.float32))

    def forward(self, images):
        return (self.model(images).softmax(-1) * self.indices).sum(-1)
