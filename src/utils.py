import os
import torch
from torchvision.utils import make_grid, save_image


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def save_sample_grid(images, path, nrow=8):
    images = images.clamp(-1, 1)
    images = (images + 1) / 2
    grid = make_grid(images, nrow=nrow)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(grid, path)