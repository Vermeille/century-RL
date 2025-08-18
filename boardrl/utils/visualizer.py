import torch
from visdom import Visdom


class VisdomVisualizer:
    def __init__(self, tag, url: str, port: int):
        self.viz = Visdom(
            env=tag,
            server=url,
            port=port,
        )
        self.viz.close()

    def push(self, name, value, epoch):
        optional = {}
        if isinstance(value, list):
            optional["legend"] = [str(i) for i in range(len(value))]
        self.viz.line(
            torch.tensor([value]),
            torch.tensor([epoch]),
            win=name,
            update="append",
            opts=dict(
                title=name,
                **optional,
            ),
        )

    def html(self, name, value):
        self.viz.text(value, win=name)

    def visdom(self, fn, *args, **kwargs):
        getattr(self.viz, "fn")(*args, **kwargs)


class OfflineVisualizer:
    def __init__(self): ...
    def push(self, name, value, epoch): ...
    def html(self, name, value): ...
    def visdom(self, fn, *args, **kwargs): ...


def Visualizer(tag, url, port):
    if url == "offline":
        return OfflineVisualizer()
    return VisdomVisualizer(tag, url, port)
