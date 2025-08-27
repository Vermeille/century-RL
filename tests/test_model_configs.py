import contextlib
import io
from pathlib import Path

import yaml

from boardrl.config import NetConfig
from boardrl.rl.model.model import Model


CONFIG_DIR = Path(__file__).resolve().parent.parent / "model-configs"


def test_model_configs_have_param_comments():
    for cfg_path in CONFIG_DIR.glob("*.yaml"):
        with open(cfg_path) as f:
            first = f.readline().strip()
            assert first.startswith(
                "# Params:"
            ), f"{cfg_path.name} missing param comment"
            expected = float(first.split(":", 1)[1].strip().rstrip("M"))
            cfg = yaml.safe_load(f)
        with contextlib.redirect_stdout(io.StringIO()):
            net_cfg = NetConfig(**cfg)
            model = Model(**net_cfg.model_dump())
        params = sum(p.numel() for p in model.parameters()) / 1e6
        assert abs(params - expected) < 0.01, (
            f"{cfg_path.name} param count mismatch: "
            f"file says {expected:.2f}M but model is {params:.2f}M"
        )
