import torch


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the Pearson correlation coefficient between two 1D tensors."""
    x = x.view(-1)
    y = y.view(-1)

    x_mean = x.mean()
    y_mean = y.mean()
    xm = x - x_mean
    ym = y - y_mean

    cov = (xm * ym).sum()
    x_std = xm.square().sum().sqrt()
    y_std = ym.square().sum().sqrt()

    eps = 1e-8
    return cov / (x_std * y_std + eps)


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the explained variance between ``pred`` and ``target``."""
    pred = pred.view(-1)
    target = target.view(-1)

    var_target = target.var(unbiased=False)
    if var_target == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    var_error = (target - pred).var(unbiased=False)
    return 1 - var_error / var_target
