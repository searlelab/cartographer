import math
import torch

class gaussian:
    def __init__( self, data, ):
        self.mu = torch.mean( data, )
        self.sigma = torch.sqrt( ( data - self.mu )**2 / 
                                     ( data.size(0) - 1 ) )
        
    def cdf( self, x, ):
        return 0.5 * ( 1 + torch.erf( ( x - self.mu ) / self.sigma / math.sqrt(2) ) )
        
    def ppf( self, q, ):
        return self.mu + self.sigma * math.sqrt(2) * torch.erfinv( 2*q - 1 ) ## ERFINV WILL FAIL HERE CAUSE Q IS NOT TENSOR
    
    
class laplace:
    def __init__( self, data ):
        # MLE of mu is median, but use mean instead so differentiable 
        self.mu = torch.mean( data, )
        self.b = torch.mean( torch.abs( data - self.mu ), )
        
    def cdf( self, x, ):
        if x <= self.mu:
            return 0.5 * torch.exp( ( x - self.mu ) / self.b )
        else:
            return 1 - 0.5 * torch.exp( -( x - self.mu ) / self.b )
        
    def ppf( self, q, ):
        if q <= 0.5:
            return self.mu + self.b * math.log( 2*q )
        else:
            return self.mu - self.b * math.log( 2 - 2*q )
        
class gumbel:
    def __init__( self, data, ):
        mean = torch.mean( data, )
        std = torch.sum( ( data - mean )**2, ) / ( data.size(0) - 1 )
        self.beta = std * math.sqrt(6) / math.pi
        self.mu = mean - 0.57721 * self.beta
        
    def cdf( self, x ):
        return torch.exp( -torch.exp( -( x - self.mu ) / self.beta ) )
    
    def ppf( self, q, ):
        return self.mu - self.beta * math.log( -math.log( q ) )
        
def return_distribution_dict():
    return { 'gaussian' : gaussian,
             'laplace'  : laplace,
             'gumbel'   : gumbel, }