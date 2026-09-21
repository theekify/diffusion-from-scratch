import sys
sys.path.append("..")

import torch
import matplotlib.pyplot as plt
from src.schedule import NoiseSchedule
from src.unet import UNet
from src.sampling import sample


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedules = {"linear": "linear", "cosine": "cosine"}
    samples_per_schedule = 4

    fig, axes = plt.subplots(samples_per_schedule, len(schedules), figsize=(6, 12))

    for col, (label, schedule_type) in enumerate(schedules.items()):
        schedule = NoiseSchedule(timesteps=1000, schedule_type=schedule_type).to(device)
        model = UNet(in_channels=1, base_channels=32).to(device)
        checkpoint = torch.load(f"../outputs/checkpoints/{schedule_type}_epoch19.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        torch.manual_seed(42)
        images = sample(model, schedule, num_samples=samples_per_schedule, device=device)

        for row in range(samples_per_schedule):
            img = ((images[row, 0].clamp(-1, 1) + 1) / 2).cpu().numpy()
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(label)

    plt.tight_layout()
    plt.savefig("../outputs/samples/schedule_ablation.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()