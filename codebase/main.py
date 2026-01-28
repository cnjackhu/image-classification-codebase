# import logging
import dataclasses
import json
import os
import pprint
import re
from pathlib import Path
import torch
import torch.cuda
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data
from pyhocon import ConfigTree, HOCONConverter
from torch.utils.collect_env import get_pretty_env_info
from sam import SAM
import wandb
from codebase.config import Args
from codebase.criterion import CRITERION
from codebase.data import DATA
from codebase.engine import evaluate_one_epoch, train_one_epoch, train_reg_epoch, evaluate_reg_epoch
from codebase.models import MODEL
from codebase.optimizer import OPTIMIZER
from codebase.scheduler import SCHEDULER
import sys
# Add the path to where you cloned it (outside your current repo)
sys.path.append("/ibex/user/xiey0d/hjh/BayesiPy")
from benchmarks.regression.regression_unified import load_dataset_and_model,make_loaders
# from codebase.torchutils.common import StateCheckPoint
from codebase.torchutils.common import (
    MetricsStore,
    disable_debug_api,
    get_device,
    set_cudnn_auto_tune,
    set_proper_device,
    set_reproducible,
    unwarp_module,
)
from codebase.torchutils.distributed import (
    distributed_init,
    is_dist_avail_and_init,
    world_size,
)
from codebase.torchutils.metrics import EstimatedTimeArrival

# from codebase.torchutils.logging import init_logger, create_code_snapshot


# _logger = logging.getLogger(__name__)


def excute_pipeline(
    only_evaluate: bool,
    reg: int,
    lamb: float,
    max_norm: float,
    start_epoch: int,
    max_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    states: dict,
    metric_store: MetricsStore,
    output_dir: Path,
    **kwargs,
):
    if only_evaluate:
        metric_store += evaluate_one_epoch(epoch=0, loader=val_loader, **kwargs)
        return

    eta = EstimatedTimeArrival(max_epochs)
    best_loss = 10000
    for epoch in range(start_epoch + 1, max_epochs + 1):
        if is_dist_avail_and_init():
            if hasattr(train_loader, "sampler"):
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)

        metric_store += train_reg_epoch(
            reg=reg,
            lamb=lamb,
            epoch=epoch,
            max_norm=max_norm,
            loader=train_loader,
            output_dir=output_dir,
            **kwargs,
        )

        metric_store += evaluate_reg_epoch(
            reg=reg,
            lamb=lamb,
            epoch=epoch,
            max_norm=max_norm,
            loader=val_loader,
            output_dir=output_dir,
            **kwargs,
        )

        test_metrics = evaluate_reg_epoch(
            reg=reg,
            lamb=lamb,
            epoch=epoch,
            max_norm=max_norm,
            loader=test_loader,
            output_dir=output_dir,
            **kwargs,
        )
        # Assuming your existing dict is called 'metrics'
        test_metrics = {k.replace("eval/", "test/"): v for k, v in test_metrics.items()}
        metric_store += test_metrics

        # using wandb to log,
        dic = metric_store.get_last_metrics()
        wandb.log(dic)
        # log Variance of the model with the best log-loss
        if dic["eval/loss"] < best_loss:
            wandb.run.summary["eval/var"] = dic["eval/var"]
            best_loss = dic["eval/loss"]
        # state_ckpt.save(metric_store=metric_store, states=states)

        eta.step()

        best_metrics = metric_store.get_best_metrics()
        print(
            f"Epoch={epoch:04d} complete, best val top1-acc={best_metrics['eval/top1_acc'] * 100:.2f}%, "
            f"top5-acc={best_metrics['eval/top5_acc'] * 100:.2f}% (epoch={metric_store.best_epoch + 1}), {eta}"
        )


def prepare_for_training(conf: ConfigTree, local_rank: int):
    model_config = conf.get("model")
    load_from = model_config.pop("load_from")
    model: nn.Module = MODEL.build_from(model_config)
    if load_from is not None:
        model.load_state_dict(torch.load(load_from, map_location="cpu"))

    if is_dist_avail_and_init() and conf.get_bool("sync_batchnorm"):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    train_loader, val_loader = DATA.build_from(
        conf.get("data"), dict(local_rank=local_rank)
    )

    criterion = CRITERION.build_from(conf.get("criterion"))

    optimizer_config: dict = conf.get("optimizer")
    basic_bs = optimizer_config.pop("basic_bs")
    optimizer_config["lr"] = optimizer_config["lr"] * (
        conf.get("data.batch_size") * world_size() / basic_bs
    )
    optimizer = OPTIMIZER.build_from(
        optimizer_config, dict(params=model.named_parameters())
    )
    # ============================================================
    # [ADDED] Wrap with SAM if reg == 10
    # ============================================================
    if conf.get("reg") == 10:
        print(f"Wrapping {type(optimizer).__name__} with SAM Optimizer (reg=10)")
        
        # We pass the *instance* of the optimizer we just built.
        # The modified SAM class will inherit its param_groups and settings.
        optimizer = SAM(
            
            base_optimizer=optimizer, 
            rho=0.05, 
            adaptive=False
        )
    # ============================================================
    

    print(
        f"Set lr={optimizer_config['lr']:.4f} with batch size={conf.get('data.batch_size') * world_size()}"
    )

    scheduler = SCHEDULER.build_from(conf.get("scheduler"), dict(optimizer=optimizer))

    if torch.cuda.is_available():
        model = model.to(
            device=get_device(), memory_format=getattr(torch, conf.get("memory_format"))
        )
        criterion = criterion.to(device=get_device())

    if conf.get_bool("use_compile"):
        if hasattr(torch, "compile"):
            print("Use torch.compile to optimize model, please wait for while.")
            model = torch.compile(model=model, **conf.get("compile"))
        else:
            print("PyTorch version is too old to support torch.compile, skip it.")

    if conf.get_bool("use_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # image_size = conf.get_int('data.image_size')
    # _logger.info(f"Model details: n_params={compute_nparam(model)/1e6:.2f}M, "
    #              f"flops={compute_flops(model,(1,3, image_size, image_size))/1e6:.2f}M.")

    metric_store = MetricsStore(dominant_metric_name="eval/top1_acc")
    states = dict(model=unwarp_module(model), optimizer=optimizer, scheduler=scheduler)
    # state_ckpt = StateCheckPoint(output_dir)

    # state_ckpt.restore(metric_store, states, device=get_device())

    if is_dist_avail_and_init():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    return (
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        metric_store,
        states,
    )


def prepare_for_training_reg(conf: ConfigTree, local_rank: int):
    # 1. Load Regression Datasets and Models
    # Uses the specialized interface for Airline, Year, or Taxi datasets
    dataset_name = conf.get("data.type_") 
    device = get_device()
    
    # Use float32 for regression to maintain precision for continuous targets
    dtype = torch.float64
    
    # Retrieve normalized datasets, model architecture, and target statistics
    (train_ds, test_ds, val_ds), model, y_mean, y_std = load_dataset_and_model(
        dataset_name, device, dtype
    )

    # Wrap Datasets into DataLoaders using the provided make_loaders interface
    # This standardizes batching and shuffling across train, test, and val splits
    train_loader, test_loader, val_loader = make_loaders(
        train_ds, 
        test_ds, 
        val_ds, 
        batch_size=conf.get("data.batch_size")
    )

    # 2. Define Loss Function
    # Mean Squared Error (MSE) is the standard objective for these regression tasks
    criterion = torch.nn.MSELoss()

    # 3. Optimizer Configuration
    optimizer_config: dict = conf.get("optimizer")
    basic_bs = optimizer_config.pop("basic_bs")
    
    # Linear scaling of learning rate based on total batch size across GPUs
    optimizer_config["lr"] = optimizer_config["lr"] * (
        conf.get("data.batch_size") * world_size() / basic_bs
    )
    
    # Build the chosen optimizer (e.g., Adam, SGD) from the project registry
    optimizer = OPTIMIZER.build_from(
        optimizer_config, dict(params=model.named_parameters())
    )

    # 4. Learning Rate Scheduler
    # Handles learning rate decay relative to the optimizer instance
    scheduler = SCHEDULER.build_from(conf.get("scheduler"), dict(optimizer=optimizer))

    # 5. Metrics Storage
    # Primary metric is set to MSE to track regression performance
    metric_store = MetricsStore(dominant_metric_name="eval/mse")

    # 6. State Management
    # Storing y_stats is critical for reversing Z-score normalization during evaluation
    states = dict(
        model=unwarp_module(model), 
        optimizer=optimizer, 
        scheduler=scheduler,
        y_stats={"mean": y_mean, "std": y_std}
    )

    # 7. Distributed Training Setup (DDP)
    # Required for multi-GPU execution on Ibex nodes
    if is_dist_avail_and_init():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    return (
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        metric_store,
        states,
    )


def _init(local_rank: int, ngpus_per_node: int, args: Args):
    set_proper_device(local_rank)
    rank = args.node_rank * ngpus_per_node + local_rank
    # init_logger(rank=rank, filenmae=args.output_dir/"default.log")

    # patch_download_in_cn()

    # if StateCheckPoint(args.output_dir).is_ckpt_exists():
    #     _logger.info("-"*30+"Resume from the last training checkpoints."+"-"*30)

    if set_reproducible:
        # random_seed = generate_random_seed()
        random_seed = 10
        set_reproducible(random_seed)
    else:
        set_cudnn_auto_tune()
        disable_debug_api()

    # create_code_snapshot(name="code", include_suffix=[".py", ".conf"],
    #                      source_directory=".", store_directory=args.output_dir)

    print("Collect envs from system:\n" + get_pretty_env_info())
    print("Args:\n" + pprint.pformat(dataclasses.asdict(args)))

    distributed_init(
        dist_backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=rank,
    )


def main_worker(local_rank: int, ngpus_per_node: int, args: Args, conf: ConfigTree):
    conf.put("data.aug", conf.data_aug)
    # update wd 
    if conf.lamb_wd == True:
        conf.put("wd",conf.wd / conf.lamb)
    conf.put("optimizer.weight_decay", conf.wd)
    config = json.loads(HOCONConverter.convert(conf, "json"))
    # Initialize wandb
    # Add SLURM job ID to config (if available)
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if slurm_job_id is None:
        config["job_id"] = "-1"
    elif array_job_id is not None and array_task_id is not None:
        config["job_id"] = f"{array_job_id}_{array_task_id}"
    else:
        config["job_id"] = slurm_job_id
    # define the celoss
    if config["criterion"]["type_"] == "MultiMarginLoss":
        config["celoss"] = 0
    else:
        config["celoss"] = 1
    # Define keys to include in sweep config (add your new key here)
    keys = ["reg", "lamb", "job_id", "wd", "data_aug", "celoss","lamb_wd"]
    run = wandb.init(config=config, config_include_keys=keys)
    # Logging
    print(
        f"reg={config['reg']}, lamb={config['lamb']}, job_id={config['job_id']},\
    weight_decay={config['wd']},data_aug={config['data_aug']},celoss={config['celoss']}"
    )
    # change the model_name in conf according to the sweep configuration
    conf.put("model.name", wandb.config.sweep_model)
    model_name = re.sub(r"^cifar(10|100)_", "", wandb.config.sweep_model)
    # rename wandb run name and run tags
    config_dict = dict(wandb.run.config)
    if config_dict["reg"] == 0:
        hyper_name = "baseline"
    else:
        hyper_name = ",".join(f"{k}:{config_dict[k]}" for k in ["reg", "lamb"])
    wandb.run.name = model_name + "-" + hyper_name
    wandb.run.tags = [model_name]

    # set output_dir
    # args.output_dir = args.output_dir / hyper_name / model_name
    # args.output_dir.mkdir(parents=True, exist_ok=True)
    # Create the full path
    save_path = Path(args.output_dir) / model_name / hyper_name

    # Make sure the directory exists
    save_path.mkdir(parents=True, exist_ok=True)
    _init(local_rank=local_rank, ngpus_per_node=ngpus_per_node, args=args)
    if conf.task == 'classification':
        (
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            metric_store,
            states,
        ) = prepare_for_training(conf, local_rank)

    if conf.task == "regression":
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            metric_store,
            states,
        ) = prepare_for_training_reg(conf, local_rank)

    excute_pipeline(
        only_evaluate=conf.get_bool("only_evaluate"),
        reg=conf.get_int("reg"),
        lamb=conf.get_float("lamb"),
        max_norm=conf.get_float("max_norm"),
        start_epoch=metric_store.total_epoch,
        max_epochs=conf.get_int("max_epochs"),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        # state_ckpt=saver,
        states=states,
        metric_store=metric_store,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        use_amp=conf.get_bool("use_amp"),
        accmulated_steps=conf.get_int("accmulated_steps"),
        device=get_device(),
        memory_format=getattr(torch, conf.get("memory_format")),
        log_interval=conf.get_int("log_interval"),
        output_dir=save_path,
    )


def main(args: Args):
    distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, args.conf)
        )
    else:
        local_rank = 0
        main_worker(local_rank, ngpus_per_node, args, args.conf)
