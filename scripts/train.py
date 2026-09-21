import sys
sys.path.append("..")

import torch
from torch.optim import Adam

from src.schedule import NoiseSchedule
from src.diffusion import p_losses
from src.unet import UNet
from src.data import get_fashion_mnist_dataloader
from src.utils import save_checkpoint


def train(
    epochs=20,
    batch_size=128,
    lr=2e-4,
    timesteps=1000,
    schedule_type="cosine",
    checkpoint_dir="../outputs/checkpoints",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}, schedule={schedule_type}")

    schedule = NoiseSchedule(timesteps=timesteps, schedule_type=schedule_type).to(device)
    model = UNet(in_channels=1, base_channels=32).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    loader = get_fashion_mnist_dataloader(batch_size=batch_size, train=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for step, (x0, _) in enumerate(loader):
            x0 = x0.to(device)
            t = torch.randint(0, timesteps, (x0.shape[0],), device=device)

            loss = p_losses(model, x0, t, schedule)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"epoch {epoch} avg loss {avg_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, f"{checkpoint_dir}/{schedule_type}_epoch{epoch}.pt")

    return model, schedule


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train(epochs=args.epochs, schedule_type=args.schedule)