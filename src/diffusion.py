import torch
from src.schedule import NoiseSchedule

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extracts values from a 1D tensor `a` at indices `t` (one per batch element),
    then reshapes for broadcasting against an image batch of shape x_shape.

    Needed because each item in a batch can be noised at a *different*
    timestep t during training (that's how we cover all t values without
    needing a separate pass per timestep).
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # (batch_size,)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # (batch_size, 1, 1, 1) for images


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule, noise: torch.Tensor = None):
    """
    Forward process: sample x_t given x_0 and timestep t, using the
    closed-form equation:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

    Args:
        x0: clean images, shape (B, C, H, W)
        t: timesteps, shape (B,) — one timestep per image in the batch
        schedule: precomputed NoiseSchedule
        noise: optional pre-sampled noise (for reproducibility); sampled fresh if None

    Returns:
        x_t: noised images, same shape as x0
        noise: the noise that was added (needed as the training target)
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)

    x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise


def p_losses(model, x0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule, loss_type: str = "l2"):
    """
    Training loss for the denoiser: given a noised image x_t, have the model
    predict the noise that was added, and compare to the true noise.

    This is the epsilon-prediction parameterization from Ho et al. (2020),
    which the paper found works better in practice than predicting x_0
    directly (we'll ablate this ourselves in Week 3).

    Args:
        model: the U-Net denoiser, called as model(x_t, t) -> predicted noise
        x0: clean images, shape (B, C, H, W)
        t: timesteps, shape (B,)
        schedule: precomputed NoiseSchedule
        loss_type: "l2" (MSE) or "l1"
    """
    noise = torch.randn_like(x0)
    x_t, _ = q_sample(x0, t, schedule, noise=noise)

    predicted_noise = model(x_t, t)

    if loss_type == "l2":
        return torch.nn.functional.mse_loss(predicted_noise, noise)
    elif loss_type == "l1":
        return torch.nn.functional.l1_loss(predicted_noise, noise)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")