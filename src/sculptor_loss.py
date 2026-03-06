import torch
import torch.nn as nn
import torch.nn.functional as F


class CCS_HuberLoss( nn.Module ):
    def __init__( self, delta=1.0 ):
        super().__init__()
        self.delta = float( delta )

    def forward( self, pred, true, weights, eps=1e-7 ):
        if pred.ndim == 1:
            pred = pred.unsqueeze( 1 )
        if true.ndim == 1:
            true = true.unsqueeze( 1 )

        per_element = F.huber_loss( pred, true, reduction='none', delta=self.delta )
        per_sample = torch.mean( per_element, dim=1 )

        w = weights.reshape( -1 )
        return ( torch.sum( per_sample * w ) + eps ) / ( torch.sum( w ) + eps )
