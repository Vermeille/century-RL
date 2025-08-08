import subprocess
import sys
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))

pytestmark = [pytest.mark.config, pytest.mark.slow]


@pytest.mark.parametrize(
    "config_path", CONFIG_FILES, ids=[str(p) for p in CONFIG_FILES]
)
def test_config_runs(config_path):
    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(config_path),
            "-x",
            "train.iterations=10",
            "-x",
            "visdom_url=offline",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
