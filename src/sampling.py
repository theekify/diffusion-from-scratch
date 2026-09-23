import torch
from src.diffusion import extract, predict_start_from_noise
from src.schedule import NoiseSchedule


@torch.no_grad()
def p_sample(model, x_t, t, t_index, schedule: NoiseSchedule, parameterization: str = "epsilon"):
    model_output = model(x_t, t)

    if parameterization == "epsilon":
        x0_pred = predict_start_from_noise(x_t, t, model_output, schedule)
    elif parameterization == "x0":
        x0_pred = model_output
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    x0_pred = x0_pred.clamp(-1.0, 1.0)

    alphas_cumprod_t = extract(schedule.alphas_cumprod, t, x_t.shape)
    alphas_cumprod_prev_t = extract(schedule.alphas_cumprod_prev, t, x_t.shape)
    betas_t = extract(schedule.betas, t, x_t.shape)
    alphas_t = extract(schedule.alphas, t, x_t.shape)

    posterior_mean_coef_x0 = (torch.sqrt(alphas_cumprod_prev_t) * betas_t) / (1.0 - alphas_cumprod_t)
    posterior_mean_coef_xt = (torch.sqrt(alphas_t) * (1.0 - alphas_cumprod_prev_t)) / (1.0 - alphas_cumprod_t)

    model_mean = posterior_mean_coef_x0 * x0_pred + posterior_mean_coef_xt * x_t

    if t_index == 0:
        return model_mean

    posterior_variance_t = extract(schedule.posterior_variance, t, x_t.shape)
    noise = torch.randn_like(x_t)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample(model, schedule: NoiseSchedule, image_size=28, channels=1, num_samples=16,
           device="cpu", parameterization: str = "epsilon"):
    model.eval()
    x_t = torch.randn(num_samples, channels, image_size, image_size, device=device)

    for t_index in reversed(range(schedule.timesteps)):
        t = torch.full((num_samples,), t_index, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, t_index, schedule, parameterization=parameterization)

    return x_t