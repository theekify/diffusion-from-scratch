import torch
from src.diffusion import extract
from src.schedule import NoiseSchedule


@torch.no_grad()
def p_sample(model, x_t, t, t_index, schedule: NoiseSchedule):
    betas_t = extract(schedule.betas, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    sqrt_recip_alphas_t = extract(1.0 / torch.sqrt(schedule.alphas), t, x_t.shape)

    predicted_noise = model(x_t, t)

    model_mean = sqrt_recip_alphas_t * (
        x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean

    posterior_variance_t = extract(schedule.posterior_variance, t, x_t.shape)
    noise = torch.randn_like(x_t)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample(model, schedule: NoiseSchedule, image_size=28, channels=1, num_samples=16, device="cpu"):
    model.eval()
    x_t = torch.randn(num_samples, channels, image_size, image_size, device=device)

    for t_index in reversed(range(schedule.timesteps)):
        t = torch.full((num_samples,), t_index, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, t_index, schedule)

    return x_t