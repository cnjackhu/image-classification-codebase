import torch
import torch.nn as nn
from ratefunctiontorch import RateCumulant
from utils import zero_one_loss, get_losses


@torch.no_grad()
def update_metrics(model, loader, train_loss, loss_fn=nn.CrossEntropyLoss(reduction="none")):
    """
    Update metrics dictionary with current model performance metrics.
    """
    (log_loss, zero_one_losses) = get_losses(model, loader, loss_fn=[loss_fn, zero_one_loss])
    
    ratecumulant = RateCumulant.from_losses(log_loss)

    L = ratecumulant.compute_mean()
    alphaD, lamb, cummulant = ratecumulant.compute_rate_function(
        L - train_loss, return_lambdas=True, return_cummulants=True
    )
    var = ratecumulant.compute_variance()
    error = torch.mean(zero_one_losses).item()
    return L.item(),alphaD.item(),var.item(),lamb.item(),cummulant.item(),error



@torch.no_grad()
def update_metrics_online(ratecumulant, train_loss):
    """
    Update metrics dictionary with current model performance metrics.
    """
    L = ratecumulant.compute_mean()
    alphaD, lamb, cummulant = ratecumulant.compute_rate_function(
        L - train_loss, return_lambdas=True, return_cummulants=True
    )

    #rough estimate of error
    error = 1-torch.mean((torch.exp(-ratecumulant.get_losses())>0.5).float()).item()
    var = ratecumulant.compute_variance()
    return L.item(),alphaD.item(),var.item(),lamb.item(),cummulant.item(),error
    


