import functools
import logging
from ratefunctiontorch import OnlineCumulant, RateCumulant
from regularizers import compute_regularizer
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.classification import MulticlassCalibrationError
from codebase.torchutils.distributed import world_size
from codebase.torchutils.metrics import AccuracyMetric, AverageMetric, EstimatedTimeArrival
from codebase.torchutils.common import GradientAccumulator
from codebase.torchutils.common import ThroughputTester, time_enumerate
from metrics import update_metrics, update_metrics_online
_logger = logging.getLogger(__name__)

scaler = None


def _run_one_epoch(is_training: bool,
                    reg: int,
                    lamb:float,
                   epoch: int,
                   model: nn.Module,
                   loader: data.DataLoader,
                   criterion: nn.modules.loss._Loss,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   use_amp: bool,
                   accmulated_steps: int,
                   device: str,
                   memory_format: str,
                   log_interval: int):
    phase = "train" if is_training else "eval"
    model.train(mode=is_training)

    global scaler
    if scaler is None:
        scaler = GradScaler(enabled=use_amp and is_training)

    gradident_accumulator = GradientAccumulator(steps=accmulated_steps, enabled=is_training)
    ece = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='l1')
    mce = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='max')
    all_probs = []
    all_targets = []

    time_cost_metric = AverageMetric("time_cost")
    loss_metric = AverageMetric("loss")
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    eta = EstimatedTimeArrival(len(loader))
    speed_tester = ThroughputTester()

    if is_training and scheduler is not None:
        scheduler.step(epoch)

    lr = optimizer.param_groups[0]['lr']
    _logger.info(f"{phase.upper()} start, epoch={epoch:04d}, lr={lr:.6f}")
    
    # Set criterion reduction based on training 
    criterion.reduction = 'none' if is_training  else 'mean'

    # Initialize online cumulant if needed
    if is_training:
        train_onlinecumulant = OnlineCumulant(device, len(loader.dataset), loss_fn=criterion)

    # Main training/validation loop
    for time_cost, iter_, (inputs, targets) in time_enumerate(loader, start=1):
        # Data loading
        inputs = inputs.to(device=device, non_blocking=True, memory_format=memory_format)
        targets = targets.to(device=device, non_blocking=True)

        # Forward pass
        #breakpoint()
        with torch.set_grad_enabled(mode=is_training):
            with autocast(enabled=use_amp and is_training):
                outputs = model(inputs)
                if is_training:
                    train_losses = criterion(outputs, targets) # not average on the batch
                    train_onlinecumulant.update_losses(train_losses.clone().to(device))

                    # Calculate regularizer and batch loss
                    if reg == 0:
                        batch_loss = torch.mean(train_losses)
                        regularizer = 0
                    elif reg == 3:
                        regularizer, batch_loss, _ = compute_regularizer(lamb, train_losses, overlap=0.0)
                    elif reg == 4:
                        regularizer, batch_loss, _ = compute_regularizer(lamb, train_losses, overlap=1.0)
                    elif reg == 5:
                        regularizer, batch_loss, _ = compute_regularizer(lamb, train_losses, overlap=0.5)
                    elif reg == 6:
                        s = torch.tensor(lamb, dtype=torch.float32, device=device)
                        _, lambda_star = train_onlinecumulant.compute_inverse_rate_function(s, return_lambdas=True)
                        lambda_star = torch.clamp(lambda_star.clone().to(device), min=0.00001, max=10.0)
                        regularizer, batch_loss, _ = compute_regularizer(lambda_star, train_losses, overlap=0.0)
                    elif reg == 7:
                        s = torch.tensor(np.log(1.0/lamb)/len(loader.dataset), dtype=torch.float32, device=device)
                        _, lambda_star = train_onlinecumulant.compute_inverse_rate_function(s, return_lambdas=True)
                        lambda_star = torch.clamp(lambda_star.clone().to(device), min=0.00001, max=10.0)
                        regularizer, batch_loss, _ = compute_regularizer(lambda_star, train_losses, overlap=0.0)
                    elif reg == 8:
                        s = torch.tensor(lamb, dtype=torch.float32, device=device)
                        with torch.no_grad():
                            _, lambda_star = train_onlinecumulant.compute_inverse_rate_function(s, return_lambdas=True)
                            lambda_star = torch.clamp(lambda_star.clone().to(device), min=0.00001, max=10.0)
                        regularizer, batch_loss, _ = compute_regularizer(lambda_star, train_losses, overlap=0.0)
                    loss = batch_loss + regularizer
                    loss_metric_value = torch.mean(train_losses).item()
                else: # for the eval epoch
                    loss = criterion(outputs, targets) # average on the batch
                    loss_metric_value = loss.item()
                    # maybe should caculate the lambda_star here

        # Backward pass and optimization
        gradident_accumulator.backward_step(model, loss, optimizer, scaler)
        # caculate calibration error
        probs = F.softmax(outputs.detach(), dim=1)
        all_probs.append(probs)
        all_targets.append(targets)
        
        # Update metrics
        time_cost_metric.update(time_cost)
        accuracy_metric.update(outputs, targets)
        loss_metric.update(loss_metric_value)
        eta.step()
        speed_tester.update(inputs)


        # Logging
        if iter_ % log_interval == 0 or iter_ == len(loader):
            _logger.info(", ".join([
                phase.upper(),
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{len(loader):05d}",
                f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} images/s",
                f"{loss_metric}",
                f"{accuracy_metric}",
                f"{eta}",
            ]))
            time_cost_metric.reset()
            speed_tester.reset()
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    ece_score = ece(all_probs, all_targets).item()
    mce_score = mce(all_probs, all_targets).item()

    if is_training:
        lambda_star = torch.tensor(lamb, dtype=torch.float32, device=device)
        if reg == 6 or reg == 7 or reg == 8:
            lambda_star = train_onlinecumulant.compute_inverse_rate_function(lambda_star, return_lambdas=True)

        L,alphaD,var,lambd,cummulant,error = update_metrics_online(train_onlinecumulant,lambda_star)
    else: 
        lambda_star = torch.tensor(lamb, dtype=torch.float32, device=device)
        L,alphaD,var,lambd,cummulant,error = update_metrics(model,loader,lambda_star)
    # variance of the model with the best log-loss
    

    # Final epoch logging
    _logger.info(", ".join([
        phase.upper(),
        f"epoch={epoch:04d} {phase} complete",
        f"{loss_metric}",
        f"{accuracy_metric}",
        f"{ece_score}",
        f"{mce_score}",
    ]))

    return {
        f"{phase}/lr": lr,
        f"{phase}/loss": loss_metric.compute(),
        f"{phase}/ece": ece_score,
        f"{phase}/mce": mce_score,
        f"{phase}/top1_acc": accuracy_metric.at(1).rate,
        f"{phase}/top5_acc": accuracy_metric.at(5).rate,
        f"{phase}/L": L,
        f"{phase}/alphaD": alphaD,
        f"{phase}/var": var,
        f"{phase}/lambd": lambd,
        f"{phase}/cummulant": cummulant,
    }



train_one_epoch = functools.partial(_run_one_epoch, is_training=True)
evaluate_one_epoch = functools.partial(_run_one_epoch, is_training=False)
