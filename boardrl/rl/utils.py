import copy
import torch
from boardrl.utils import PythonExec


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


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the explained variance between ``pred`` and ``target``.

    The explained variance is ``1 - Var[target - pred] / Var[target]`` and is a
    measure of how much of the variance in ``target`` is captured by ``pred``.
    The output is ``0`` when ``target`` has zero variance.
    """

    # Flatten to 1D tensors to simplify computation
    pred = pred.view(-1)
    target = target.view(-1)

    # Variance of the target
    var_target = target.var(unbiased=False)

    # Avoid division by zero when the target has no variance
    if var_target == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Variance of the prediction error
    var_error = (target - pred).var(unbiased=False)

    return 1 - var_error / var_target


def R_squared(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ss_res = ((x - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    eps = 1e-8
    r2 = 1 - ss_res / (ss_tot + eps)
    return r2


class ReferenceModelHandler:
    def __init__(self, base, update_str):
        self.model = copy.deepcopy(base)
        self.model.eval()
        self.version = 0
        self.reference_update_exec = PythonExec(update_str)

    @torch.no_grad()
    def copy_from(self, src):
        for reference_param, param in zip(
            self.model.state_dict().values(),
            src.state_dict().values(),
        ):
            reference_param.data.copy_(param.data)

    def update(self, src, *, epoch, pit_results, episode_results):
        env = {
            "epoch": epoch,
            "pit": pit_results,
            "episode": episode_results,
            "True": True,
            "False": False,
            "version": self.version,
            "__builtins__": {
                "print": print,
            },
        }
        env["__builtins__"]["exists"] = lambda s: s in self.reference_update_exec.ctx

        update = self.reference_update_exec(env)
        if update:
            self.copy_from(src)
            self.model.eval()
            self.model.version = epoch
