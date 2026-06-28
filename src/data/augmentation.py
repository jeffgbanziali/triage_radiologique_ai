"""Pipelines d'augmentation pour l'imagerie thoracique (plausibilite anatomique respectee)."""
from torchvision import transforms


def get_train_transforms(resolution: int = 64, normalize: bool = True) -> transforms.Compose:
    """Pipeline d'augmentation pour l'entrainement (flip, rotation, crop, jitter)."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    ops = [
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(
            size=resolution,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def get_val_transforms(resolution: int = 64, normalize: bool = True) -> transforms.Compose:
    """Pipeline deterministe pour validation et test (resize + ToTensor + normalize)."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    ops = [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def get_anomaly_transforms(resolution: int = 64) -> transforms.Compose:
    """Pipeline AE/VAE : sans normalisation ImageNet, sortie en [0,1] pour MSE."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
