import torch


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Pearson correlation coefficient between two 1D tensors x and y.
    """
    # Ensure x and y are 1D (or view them as 1D if they have a single dimension of interest)
    x = x.view(-1)
    y = y.view(-1)

    # Compute means
    x_mean = x.mean()
    y_mean = y.mean()

    # Subtract mean
    xm = x - x_mean
    ym = y - y_mean

    # Numerator: covariance
    cov = (xm * ym).sum()

    # Denominator: product of standard deviations
    x_std = xm.square().sum().sqrt()
    y_std = ym.square().sum().sqrt()

    # Avoid division by zero
    eps = 1e-8
    corr = cov / (x_std * y_std + eps)

    return corr
