import torch
from src.unet import UNet

model = UNet(in_channels=1, base_channels=32)
x = torch.randn(4, 1, 28, 28)
t = torch.randint(0, 1000, (4,))

out = model(x, t)
print(out.shape)  # should be torch.Size([4, 1, 28, 28])
print(sum(p.numel() for p in model.parameters()))