import sys
sys.path.append("..")

import torch
from src.schedule import NoiseSchedule
from src.unet import UNet
from src.sampling import sample
from src.utils import save_sample_grid


def generate(checkpoint_path, num_samples=16, timesteps=1000, schedule_type="cosine", output_path="../outputs/samples/generated.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    schedule = NoiseSchedule(timesteps=timesteps, schedule_type=schedule_type).to(device)
    model = UNet(in_channels=1, base_channels=32).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    images = sample(model, schedule, num_samples=num_samples, device=device)
    save_sample_grid(images, output_path)
    print(f"saved to {output_path}")


if __name__ == "__main__":
    generate(checkpoint_path="../outputs/checkpoints/model_epoch19.pt")