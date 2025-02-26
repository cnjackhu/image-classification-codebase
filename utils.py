import torch
import torchvision
from torch import nn
import torch.nn.functional as F
# Import the calibration error metric from TorchMetrics
from torchmetrics.classification import MulticlassCalibrationError







def zero_one_loss(logits, targets):
    """
    Compute 0-1 loss (misclassification error) for each sample.
    
    Parameters
    ----------
    logits : torch.Tensor
        Model output logits
    targets : torch.Tensor
        Ground truth labels
        
    Returns
    -------
    torch.Tensor
        0-1 loss for each sample (1 for misclassification, 0 for correct)
    """
    predictions = torch.argmax(logits, dim=1)
    return (predictions != targets).float()


def get_losses(
    model,
    loader,
    loss_fn=nn.CrossEntropyLoss(reduction="none"),
    n_bins=10,
    num_classes=10
):
    """
    Compute one or multiple losses for a given model using data from a DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        A DataLoader providing the dataset to evaluate.
    loss_fn : callable or list of callable, optional
        Single loss function or list of loss functions to evaluate.
        Default is nn.CrossEntropyLoss with no reduction.
        
    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        If single loss_fn: Tensor containing the losses for each batch
        If multiple loss_fns: Tuple of tensors, one for each loss function
    """
    # Get a parameter from the model to determine its device (GPU/CPU)
    p = next(model.parameters())
    device = p.device

    # Convert single loss function to list for unified processing
    if not isinstance(loss_fn, (list, tuple)):
        loss_fn = [loss_fn]
    
    # Initialize list of losses for each function
    all_losses = [[] for _ in loss_fn]
    all_probs = []
    all_targets = []
    # Set the model to evaluation mode
    model.eval()
    
    # Loop over the data loader
    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits = model(data)      
        # Compute all losses
        for i, fn in enumerate(loss_fn):
            loss = fn(logits, targets)
            all_losses[i].append(loss)
    
    # Concatenate losses for each function
    all_losses = [torch.cat(losses) for losses in all_losses]
    # Return a single tensor if only one loss fn, else a tuple of losses
    losses_out = all_losses[0] if len(all_losses) == 1 else tuple(all_losses)
    
    return losses_out
