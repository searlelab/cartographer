import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import epsilon, train_fdr
from electrician_settings import charge_dist_len
from loss_functions import generate_outlier_mask, l2_norm, neg_logit
from scout_loader import SCOUT_CCS_OFFSET, SCOUT_CHARGE_DIST_OFFSET, SCOUT_IRT_OFFSET
from scout_settings import ms2_vector_len


def _masked_batch_mean( values, dist_name, fdr, eps=epsilon ):
    outlier_mask = generate_outlier_mask( values, dist_name, fdr )
    return ( torch.sum( values * outlier_mask ) + eps ) / ( torch.sum( outlier_mask ) + eps )


class ScoutMultiTaskLoss( nn.Module ):
    def __init__( self, huber_delta=1.0, fdr=train_fdr ):
        super().__init__()
        self.huber_delta = float( huber_delta )
        self.fdr = float( fdr )

    def _ms2_loss( self, pred, true ):
        pred_norm = l2_norm( pred, eps=epsilon )
        true_norm = l2_norm( true, eps=epsilon )
        cosine = torch.sum( pred_norm * true_norm, dim=1 )
        score = neg_logit( cosine, eps=epsilon )
        return _masked_batch_mean( score, 'gumbel', self.fdr )

    def _scalar_loss( self, pred, true ):
        pred = pred.reshape( -1 )
        true = true.reshape( -1 )
        per_sample = F.huber_loss( pred, true, reduction='none', delta=self.huber_delta )
        abs_error = torch.abs( pred - true )
        outlier_mask = generate_outlier_mask( abs_error, 'laplace', self.fdr )
        return ( torch.sum( per_sample * outlier_mask ) + epsilon ) / ( torch.sum( outlier_mask ) + epsilon )

    def _charge_ce_loss( self, pred, true ):
        log_pred = torch.log( pred.clamp( epsilon ) )
        ce_per_sample = -torch.sum( true * log_pred, dim=1 )
        return _masked_batch_mean( ce_per_sample, 'gumbel', self.fdr )

    def forward( self, pred, target_bundle, mask_bundle ):
        losses = []
        mask_ms2 = mask_bundle[ :, 0 ] > 0.5
        mask_irt = mask_bundle[ :, 1 ] > 0.5
        mask_ccs = mask_bundle[ :, 2 ] > 0.5
        mask_charge = mask_bundle[ :, 3 ] > 0.5

        if torch.any( mask_ms2 ):
            ms2_true = target_bundle[ mask_ms2, :ms2_vector_len ]
            ms2_pred = pred[ 'ms2' ][ mask_ms2 ]
            losses.append( self._ms2_loss( ms2_pred, ms2_true ) )

        if torch.any( mask_irt ):
            irt_true = target_bundle[ mask_irt, SCOUT_IRT_OFFSET ]
            irt_pred = pred[ 'irt' ][ mask_irt, 0 ]
            losses.append( self._scalar_loss( irt_pred, irt_true ) )

        if torch.any( mask_ccs ):
            ccs_true = target_bundle[ mask_ccs, SCOUT_CCS_OFFSET ]
            ccs_pred = pred[ 'ccs' ][ mask_ccs, 0 ]
            losses.append( self._scalar_loss( ccs_pred, ccs_true ) )

        if torch.any( mask_charge ):
            charge_true = target_bundle[ mask_charge, SCOUT_CHARGE_DIST_OFFSET : SCOUT_CHARGE_DIST_OFFSET + charge_dist_len ]
            charge_pred = pred[ 'charge_dist' ][ mask_charge ]
            losses.append( self._charge_ce_loss( charge_pred, charge_true ) )

        if len( losses ) == 0:
            return torch.zeros( (), dtype=target_bundle.dtype, device=target_bundle.device, requires_grad=True )

        return torch.stack( losses ).mean()
