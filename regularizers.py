import torch


def compute_regularizer(lambda_star, val_losses, overlap=0.0):
    """
    Compute the regularizer with configurable overlap between sets.
    
    Args:
        lambda_star: The lambda parameter
        val_losses: Tensor of validation losses
        overlap: Float between 0 and 1, controlling the overlap between sets
                0.0 = no overlap (like compute_regularizer_1)
                1.0 = full overlap (like compute_regularizer_0)
    """
    n = val_losses.size(0)
    split_idx = int(n * (0.5 + overlap/2))  # Adjusts split point based on overlap
    
    first_set = val_losses[:split_idx]
    second_set = val_losses[n-split_idx:]  # Take from end to maintain set size
    
    partial_cumulant = torch.logsumexp(-lambda_star * first_set - torch.log(torch.tensor(split_idx, device=val_losses.device)), 0)
    
    return partial_cumulant/lambda_star +  second_set.mean(), second_set.mean(), partial_cumulant/lambda_star



def compute_regularizer_extended(lambda_star, rateculcumulant, train_losses, overlap=0.0):
    """
    Compute the regularizer with configurable overlap between sets.
    
    Args:
        lambda_star: The lambda parameter
        val_losses: Tensor of validation losses
        overlap: Float between 0 and 1, controlling the overlap between sets
                0.0 = no overlap (like compute_regularizer_1)
                1.0 = full overlap (like compute_regularizer_0)
    """
    with torch.no_grad():
        partial_cumulant = (rateculcumulant.compute_cumulants(lambda_star)-lambda_star*rateculcumulant.compute_mean())
        weights = (1-torch.exp(-lambda_star*train_losses-partial_cumulant))
    
    regularizer = (weights*train_losses).mean()

    return regularizer


def compute_regularizer_extended2(lambda_star, rateculcumulant, train_losses, overlap=0.0):
    """
    Compute the regularizer with configurable overlap between sets.
    
    Args:
        lambda_star: The lambda parameter
        val_losses: Tensor of validation losses
        overlap: Float between 0 and 1, controlling the overlap between sets
                0.0 = no overlap (like compute_regularizer_1)
                1.0 = full overlap (like compute_regularizer_0)
    """
    n = train_losses.size(0)
    split_idx = int(n * (0.5 + overlap/2))  # Adjusts split point based on overlap
    
    first_set = train_losses[:split_idx]
    second_set = train_losses[n-split_idx:]  # Take from end to maintain set size
    
    
    with torch.no_grad():
        partial_cumulant = torch.logsumexp(-lambda_star * second_set - torch.log(torch.tensor(second_set.size(0), device=train_losses.device)), 0)
        weights = -torch.exp(-lambda_star*second_set-partial_cumulant)
    
    regularizer = (weights*second_set).mean()

    return 2*first_set.mean() + regularizer



def compute_regularizer_extended3(lambda_star, rateculcumulant, train_losses, overlap=0.0):
    """
    Compute the regularizer with configurable overlap between sets.
    
    Args:
        lambda_star: The lambda parameter
        val_losses: Tensor of validation losses
        overlap: Float between 0 and 1, controlling the overlap between sets
                0.0 = no overlap (like compute_regularizer_1)
                1.0 = full overlap (like compute_regularizer_0)
    """
    n = train_losses.size(0)
    split_idx = int(n * (0.5 + overlap/2))  # Adjusts split point based on overlap
    
    first_set = train_losses[:split_idx]
    second_set = train_losses[n-split_idx:]  # Take from end to maintain set size
    
    


    with torch.no_grad():
        partial_cumulant = (rateculcumulant.compute_cumulants(lambda_star)-lambda_star*rateculcumulant.compute_mean())
        weights = -torch.exp(-lambda_star*second_set-partial_cumulant)
    
    regularizer = (weights*second_set).mean()

    return 2*first_set.mean() + regularizer


def compute_regularizer_partial(lambda_star1, lambda_star2, val_losses, overlap=0.0):
    """
    Compute the regularizer with configurable overlap between sets.
    
    Args:
        lambda_star: The lambda parameter
        val_losses: Tensor of validation losses
        overlap: Float between 0 and 1, controlling the overlap between sets
                0.0 = no overlap (like compute_regularizer_1)
                1.0 = full overlap (like compute_regularizer_0)
    """
    n = val_losses.size(0)
    split_idx = int(n * (0.5 + overlap/2))  # Adjusts split point based on overlap
    
    first_set = val_losses[:split_idx]
    partial_cumulant1 = -torch.logsumexp(-lambda_star1 * first_set - torch.log(torch.tensor(first_set.size(0), device=val_losses.device)), 0)
    

    second_set = val_losses[n-split_idx:]  # Take from end to maintain set size
    partial_cumulant2 = -torch.logsumexp(-lambda_star2 * second_set + torch.log(torch.tensor(second_set.size(0), device=val_losses.device)), 0)

    return partial_cumulant2/lambda_star2-partial_cumulant1/lambda_star1
