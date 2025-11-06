import logging
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from .utils import get_samplers
from .register import DATA
import numpy as np
from torch.utils.data import Subset

_logger = logging.getLogger(__name__)


def get_train_transforms(mean, std):
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(mean, std):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_vit_train_transforms(mean, std, img_size):
    return T.Compose([
        T.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_vit_val_transforms(mean, std, img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _cifar(root, image_size, mean, std, batch_size, num_workers, is_vit, dataset_builder, aug,**kwargs):
    # --- 1. Get Transform pipelines ---
    if is_vit:
        train_transforms = get_vit_train_transforms(mean, std, image_size)
        val_transforms = get_vit_val_transforms(mean, std, image_size)
    else:
        if aug:
            train_transforms = get_train_transforms(mean, std)
            val_transforms = get_val_transforms(mean, std)
        else:
            train_transforms =  get_val_transforms(mean, std)
            val_transforms = get_val_transforms(mean, std)
    # --- 2. Load the full training set TWICE ---
    # We load it twice: once with training transforms, once with validation transforms.
    # This is the standard way to handle splitting with different transforms.
    breakpoint()
    train_dataset_with_aug = dataset_builder(root, train=True, transform=train_transforms, download=True)
    train_dataset_no_aug = dataset_builder(root, train=True, transform=val_transforms, download=True)

    # --- 3. Create 90/10 split indices ---
    num_train = len(train_dataset_with_aug)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))  # 10% for validation

    # We use a fixed seed for reproducible splits.
    # You might want to pass this seed as an argument.
    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    # --- 4. Create the new Subset datasets ---
    # The trainset (90%) uses the dataset WITH augmentations
    trainset = Subset(train_dataset_with_aug, train_idx)
    # The valset (10%) uses the dataset WITHOUT augmentations
    valset = Subset(train_dataset_no_aug, val_idx)
    _logger.info(f"Loading {dataset_builder.__name__} dataset.")
    _logger.info(f"  Total original training images: {num_train}")
    _logger.info(f"  Splitting into 90% trainset (len={len(trainset)}) and 10% valset (len={len(valset)})")
    _logger.info(f"  The original test set (train=False) is NOT used.")
    # --- 5. Create Samplers and DataLoaders (unchanged) ---
    train_sampler = get_samplers(trainset, is_training=True)
    val_sampler = get_samplers(valset, is_training=False)

    train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=(train_sampler is None),
                                   sampler=train_sampler,
                                   num_workers=num_workers,
                                   persistent_workers=True)
    val_loader = data.DataLoader(valset, batch_size=batch_size,
                                 shuffle=(val_sampler is None),
                                 sampler=val_sampler,
                                 num_workers=num_workers,
                                 persistent_workers=True)

    return train_loader, val_loader


@DATA.register
def cifar10(root, image_size, mean, std, batch_size, num_workers, is_vit,aug, **kwargs):
    return _cifar(
        root, image_size, mean, std, batch_size, num_workers, is_vit, CIFAR10,aug, **kwargs
    )


@DATA.register
def cifar100(root, image_size, mean, std, batch_size, num_workers, is_vit,aug, **kwargs):
    return _cifar(
        root, image_size, mean, std, batch_size, num_workers, is_vit, CIFAR100,aug, **kwargs
    )
