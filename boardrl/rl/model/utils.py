import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def zero(m):
    assert isinstance(m.weight, torch.Tensor)
    nn.init.constant_(m.weight, 0)
    if hasattr(m, "bias") and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def init(m, var_scale: float = 1.0):
    assert isinstance(m.weight, torch.Tensor)
    out_dim = m.weight.size(0)
    bound = math.sqrt(3 * var_scale / m.weight.view(out_dim, -1).size(1))
    nn.init.uniform_(m.weight, -bound, bound)
    if hasattr(m, "bias") and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def js_div(log_p, log_q):
    """
    Computes the Jensen-Shannon Divergence between two distributions in a numerically stable manner
    using log probabilities.

    Args:
        log_p (torch.Tensor): Log probabilities for distribution P. Shape: (batch_size, num_classes).
        log_q (torch.Tensor): Log probabilities for distribution Q. Shape: (batch_size, num_classes).

    Returns:
        torch.Tensor: The JS divergence for each sample in the batch.
    """
    assert log_p.shape == log_q.shape
    assert log_p.ndim == 2
    log_p = F.log_softmax(log_p, dim=1)
    log_q = F.log_softmax(log_q, dim=1)

    # Calculate log-m directly using the log-sum-exp trick
    log_m = torch.logaddexp(log_p, log_q) - torch.log(torch.tensor(2.0))

    # Compute KL divergences using log probabilities
    kl_p_m = F.kl_div(log_m, log_p, reduction="batchmean", log_target=True)
    kl_q_m = F.kl_div(log_m, log_q, reduction="batchmean", log_target=True)

    # Jensen-Shannon Divergence is the mean of the KL divergences
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div


def jeffreys_div(log_p, log_q):
    """
    Computes the Jeffreys Divergence between two distributions in a numerically stable manner
    using log probabilities.
    Args:
        log_p (torch.Tensor): Log probabilities for distribution P. Shape: (batch_size, num_classes).
        log_q (torch.Tensor): Log probabilities for distribution Q. Shape: (batch_size, num_classes).
    Returns:
        torch.Tensor: The Jeffreys divergence for each sample in the batch.
    """
    assert log_p.shape == log_q.shape
    assert log_p.ndim == 2
    log_p = F.log_softmax(log_p, dim=1)
    log_q = F.log_softmax(log_q, dim=1)
    # Calculate log-m directly using the log-sum-exp trick
    # Compute KL divergences using log probabilities
    kl_p_q = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
    kl_q_p = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
    # Jeffreys Divergence is the mean of the KL divergences
    jeffreys_div = 0.5 * (kl_p_q + kl_q_p)
    return jeffreys_div
