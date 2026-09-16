import sys
sys.path.append("..")

import torch
import matplotlib.pyplot as plt
from src.schedule import NoiseSchedule
from src.diffusion import q_sample
from src.data import get_fashion_mnist_dataloader


def main():
    T = 1000
    schedule = NoiseSchedule(timesteps=T, schedule_type="cosine")

    loader = get_fashion_mnist_dataloader(batch_size=1, train=True)
    x0, _ = next(iter(loader))

    timesteps_to_show = [0, 100, 300, 600, 999]

    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(15, 3))

    for i, t_val in enumerate(timesteps_to_show):
        t = torch.tensor([t_val])
        x_t, _ = q_sample(x0, t, schedule)

        img = x_t[0, 0].clamp(-1, 1)
        img = (img + 1) / 2

        axes[i].imshow(img.numpy(), cmap="gray")
        axes[i].set_title(f"t={t_val}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("../outputs/samples/forward_process_sanity_check.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()