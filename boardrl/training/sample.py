import torch


def collate(xs):
    if isinstance(xs[0], (int, float)):
        return torch.tensor(xs, pin_memory=True)
    if isinstance(xs[0], torch.Tensor):
        try:
            return torch.stack(xs, dim=0)
        except RuntimeError:
            return xs
    return xs


class TrainingSample:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def collate(samples):
        return TrainingSample(
            **{
                k: collate([getattr(s, k) for s in samples])
                for k in samples[0].__dict__
            }
        )

    def to(self, *args, **kwargs):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(*args, **kwargs)
            elif isinstance(v[0], torch.Tensor):
                self.__dict__[k] = [x.to(*args, **kwargs) for x in v]
        return self

    def __repr__(self):
        out = ["TrainingSample:"]
        for k, v in self.__dict__.items():
            if k == "next":
                continue
            out.append(f"{k}: {v}")
        return "\n".join(out)
