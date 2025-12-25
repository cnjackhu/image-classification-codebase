import torch
from pathlib import Path
from pyhocon import ConfigFactory
import pandas as pd
import torch.nn as nn
import sys
import os

# Assuming this script is located in a subdirectory (e.g., table_and_figure)
# We add the project root (one level up) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# --- END PATH FIX ---
from codebase.main import prepare_for_training
from codebase.engine import evaluate_one_epoch

def run_model_evaluation(dataset: str, root_dir: str, suffix: str = None, use_celoss: int = 1,):
    """
    Evaluates models, processes results, and saves them to a CSV file.

    Args:
        dataset_name (str): 'cifar10' or 'cifar100'.
        use_celoss (int): 0 or 1. If 0, uses MultiMarginLoss; otherwise uses CrossEntropyLoss.
        root_dir (str): The path to the root directory containing model checkpoints.
        suffix (str, optional): Suffix for the output filename (e.g., 'lamb_wd').
    Example:
        run_model_evaluation(dataset_name='cifar10', root_dir='output_10', suffix='sam')
        run_model_evaluation(dataset_name='cifar100', root_dir='output_100', suffix='sam')
    """
    print(f"--- Starting Evaluation: Dataset={dataset}, CELoss={use_celoss} ---")
    # 1. Setup Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_path = f'conf/{dataset}.conf'
    conf = ConfigFactory.parse_file(conf_path)

    # 2. Prepare Environment
    (_, train_loader, val_loader, criterion, optimizer, scheduler, metric_store, states) = prepare_for_training(conf, 0)
    
    # 3. Override Loss Function if requested
    if use_celoss == 0:
        print("Overriding criterion: Using MultiMarginLoss")
        criterion = nn.MultiMarginLoss() 
    else:
        print("Using default criterion (CrossEntropyLoss)")

    # 4. Run Evaluation Loop
    df = pd.DataFrame()
    root = Path(root_dir)

    for model_dir in root.iterdir():
        model_name = model_dir.name
        if model_dir.is_dir():
            for config_dir in model_dir.iterdir():
                if config_dir.is_dir():
                    s = config_dir.name
                    if s == "baseline":
                        reg = 0
                        lamb = 0.1
                    else:
                        parts = dict(part.split(":") for part in s.split(","))
                        reg = int(parts["reg"])
                        lamb = float(parts["lamb"])
                    
                    config = {'model': model_name, 'reg': reg, 'lamb': lamb}
                    model_path = config_dir / 'epoch_200.pt'
                    
                    # Load Model
                    model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{dataset}_{model_name}", pretrained=False)
                    model = model.to(device)
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    model.eval()
                    
                    # Evaluate
                    metric_test = evaluate_one_epoch(
                        loader=val_loader,
                        model=model,
                        reg=None,
                        lamb=0.1,
                        epoch=-100,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        use_amp=False,
                        accmulated_steps=conf.get_int("accmulated_steps"),
                        device="cuda",
                        memory_format=getattr(torch, conf.get("memory_format")),
                        log_interval=conf.get_int("log_interval"),
                        max_norm=conf.get_float("max_norm"),
                        output_dir=None
                    )
                    
                    # Rename keys and combine
                    renamed_metric_test = {k.replace("eval/", "test/"): v for k, v in metric_test.items()}
                    combined = {**config, **renamed_metric_test}
                    df = pd.concat([df, pd.DataFrame([combined])], ignore_index=True)

    # 5. Post-Processing (Check Removed)
    df["celoss"] = use_celoss
    df = df.rename(columns={"model": "name"})
    
    print(f"--- Finished Evaluation: Dataset={dataset}, CELoss={use_celoss} ---")

    # 6. Save to CSV
    if suffix:
        filename = f"test_{dataset}_{suffix}.csv"
    else:
        filename = f"test_{dataset}.csv"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(script_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    output_path = os.path.join(csv_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


