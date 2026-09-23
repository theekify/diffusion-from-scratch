import torch
from src.schedule import NoiseSchedule


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule, noise: torch.Tensor = None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)

    x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise


def predict_start_from_noise(x_t, t, noise, schedule: NoiseSchedule):
    sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t


def p_losses(model, x0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule,
             loss_type: str = "l2", parameterization: str = "epsilon"):
    noise = torch.randn_like(x0)
    x_t, _ = q_sample(x0, t, schedule, noise=noise)

    model_output = model(x_t, t)

    if parameterization == "epsilon":
        target = noise
    elif parameterization == "x0":
        target = x0
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    if loss_type == "l2":
        return torch.nn.functional.mse_loss(model_output, target)
    elif loss_type == "l1":
        return torch.nn.functional.l1_loss(model_output, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")