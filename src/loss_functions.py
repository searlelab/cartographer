import numpy as np
import torch
import torch.nn as nn

from constants import epsilon, train_fdr
from distributions import return_distribution_dict


def generate_outlier_mask( array, distribution, fdr, ):
    dist_dict = return_distribution_dict()
    assert distribution in dist_dict, 'UNDEFINED DISTRIBUTION ' + distribution
    dist = dist_dict[ distribution ]( array )
    threshold = dist.ppf( 1.0 - fdr, )
    return array < threshold


def l2_norm( vector, eps=epsilon, ):
    sum_sqs = torch.sum( vector**2, dim=(1,-1), keepdims=True, )
    denom = torch.sqrt( torch.max( sum_sqs, eps, ) )
    #print( vector.shape, sum_sqs.shape, denom.shape, )
    return vector / denom

def neg_logit( array, eps=epsilon, ):
    return torch.log( (1 - array).clamp(eps,) ) - torch.log( array.clamp(eps,) ) 


class RT_masked_negLogL( nn.Module ):
    def __init__( self, n_sources, fdr=train_fdr, ):
        super().__init__()
        self.source_b = nn.Linear( n_sources, 1, bias=False, )
        nn.init.constant_( self.source_b.weight, 10.0, )
        self.fdr = fdr
    
    def forward( self, pred, true, source, eps=epsilon, ):
        abs_error = torch.abs( pred - true )
        exp_abs_error = self.source_b( source ).clamp( eps, )
        norm_abs_error = abs_error / exp_abs_error
        neg_logL = norm_abs_error + torch.log( 2 * exp_abs_error )
        
        outlier_mask = generate_outlier_mask( norm_abs_error, 'laplace', self.fdr, )
        mean_neg_logL = ( torch.sum( neg_logL * outlier_mask ) + eps ) /\
                        ( torch.sum( outlier_mask ) + eps )

        return mean_neg_logL
    
    
class Spectrum_masked_negLogit( nn.Module ):
    def __init__( self, fdr=train_fdr, ):
        super.__init__()
        self.fdr = fdr

    def forward( self, pred, true, weights, eps=epsilon, ):
        ion_mask =  ( true + 1.0 ) / ( true + 1.0 + eps ) # Incompatible ions = -1

        pred_masked = pred * ion_mask
        true_masked = true * ion_mask
        pred_norm = l2_norm( pred_masked, eps, )
        true_norm = l2_norm( true_masked, eps, )
        product = torch.sum( pred_norm * true_norm, dim=(1,-1), )
        neg_logit = neg_logit( product, )
        
        mean_neg_logit = ( torch.sum( neg_logit * outlier_mask ) + eps ) /\
                         ( torch.sum( outlier_mask ) + eps )
                         
        return mean_neg_logit
        
        