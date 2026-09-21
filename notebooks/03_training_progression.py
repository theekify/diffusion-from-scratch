import sys
sys.path.append("..")

import torch
import matplotlib.pyplot as plt
from src.schedule import NoiseSchedule
from src.unet import UNet
from src.sampling import sample


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = NoiseSchedule(timesteps=1000, schedule_type="cosine").to(device)

    epochs_to_show = [0, 4, 9, 14, 19]
    fig, axes = plt.subplots(1, len(epochs_to_show), figsize=(15, 3))

    torch.manual_seed(42)  # same noise seed across checkpoints for a fair visual comparison

    for i, epoch in enumerate(epochs_to_show):
        model = UNet(in_channels=1, base_channels=32).to(device)
        checkpoint = torch.load(f"../outputs/checkpoints/model_epoch{epoch}.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        torch.manual_seed(42)
        images = sample(model, schedule, num_samples=1, device=device)
        img = ((images[0, 0].clamp(-1, 1) + 1) / 2).cpu().numpy()

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"epoch {epoch}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("../outputs/samples/training_progression.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()