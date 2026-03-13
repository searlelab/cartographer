import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import train_fdr
from loss_functions import generate_outlier_mask


class CCS_HuberLoss( nn.Module ):
    def __init__( self, delta=1.0, fdr=train_fdr ):
        super().__init__()
        self.delta = float( delta )
        self.fdr = float( fdr )

    def forward( self, pred, true, weights, eps=1e-7 ):
        if pred.ndim == 1:
            pred = pred.unsqueeze( 1 )
        if true.ndim == 1:
            true = true.unsqueeze( 1 )

        per_element = F.huber_loss( pred, true, reduction='none', delta=self.delta )
        per_sample = torch.mean( per_element, dim=1 )
        abs_error = torch.mean( torch.abs( pred - true ), dim=1 )

        w = weights.reshape( -1 )
        outlier_mask = generate_outlier_mask( abs_error, 'laplace', self.fdr )
        masked_w = w * outlier_mask.to( dtype=w.dtype )
        if float( torch.sum( masked_w ).detach().item() ) <= eps:
            masked_w = w

        return ( torch.sum( per_sample * masked_w ) + eps ) / ( torch.sum( masked_w ) + eps )
