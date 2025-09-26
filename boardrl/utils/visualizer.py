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

    def push_range(self, name: str, stats, epoch: int):
        """Plot a mean line with a filled min/max band.

        - Creates/updates a window named "{name} (min/max)" to avoid
          interfering with existing single-line plots.
        - Shades the area between min and max, and overlays the mean line.
        """
        if len(stats) == 0:
            return
        # Order matters: min first, then max (filled to previous), then mean.
        mean = sum(stats) / len(stats)
        vmin = min(stats)
        vmax = max(stats)
        Y = torch.tensor([[vmin, vmax, mean]], dtype=torch.float32)
        X = torch.tensor([[epoch, epoch, epoch]], dtype=torch.float32)

        # Use per-trace Plotly options so only max fills to previous (min).
        # Keep the mean as an unfilled, prominent line.
        band_color = "rgba(66,135,245,0.25)"  # light blue translucent
        min_line_color = "rgba(66,135,245,0.6)"

        self.viz.line(
            Y,
            X,
            win=name,
            update="append",
            opts=dict(
                title=name,
                legend=["min", "max", "mean"],
                xlabel="epoch",
                ylabel=name,
                traceopts={
                    "plotly": {
                        "min": {"line": {"color": min_line_color}},
                        "max": {
                            # Fill area between max and previous (min) trace.
                            "fill": "tonexty",
                            "fillcolor": band_color,
                            # Hide the max outline; rely on the band.
                            "line": {"color": "rgba(0,0,0,0)", "width": 0},
                        },
                    }
                },
            ),
        )

    def html(self, name, value):
        self.viz.text(value, win=name)

    def visdom(self, fn, *args, **kwargs):
        getattr(self.viz, fn)(*args, **kwargs)


class OfflineVisualizer:
    def __init__(self): ...
    def push(self, name, value, epoch): ...
    def push_band(
        self, name: str, mean: float, vmin: float, vmax: float, epoch: int
    ): ...
    def html(self, name, value): ...
    def visdom(self, fn, *args, **kwargs): ...


def Visualizer(tag, url, port):
    if url == "offline":
        return OfflineVisualizer()
    return VisdomVisualizer(tag, url, port)
