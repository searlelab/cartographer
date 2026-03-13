import torch
import torch.nn as nn

from chronologer_unimod_distributions import return_distribution_dict
from chronologer_unimod_settings import epsilon, train_fdr


def generate_outlier_mask(array, dist_family, fdr):
    dist_dict = return_distribution_dict()
    assert dist_family in dist_dict, 'UNDEFINED DISTRIBUTION ' + dist_family
    dist = dist_dict[dist_family](data=array)
    threshold = dist.ppf(1.0 - fdr)
    return array < threshold


def fdr_masked_mean(array, dist_family, fdr, eps=epsilon):
    outlier_mask = generate_outlier_mask(array, dist_family, fdr)
    masked_mean = (torch.sum(array * outlier_mask) + eps) / (torch.sum(outlier_mask) + eps)
    return masked_mean


class LogL_Loss(nn.Module):
    def __init__(self, n_sources=1, family='laplace', fdr=train_fdr):
        super().__init__()
        self.source_scale = nn.Linear(n_sources, 1, bias=False)
        nn.init.constant_(self.source_scale.weight, 10.0)
        self.family = family
        self.dist = return_distribution_dict()[family]
        self.fdr = fdr

    def forward(self, pred, true, source, eps=epsilon):
        scale = self.source_scale(source).clamp(eps)
        dist = self.dist(center=true, scale=scale)
        logL_loss = -1 * dist.logL(pred)
        return fdr_masked_mean(logL_loss, self.family, self.fdr, eps=eps)

