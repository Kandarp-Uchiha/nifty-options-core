import importlib

import pandas as pd

import nifty_core.paths
from nifty_core.research.option_utils import price_at, round_atm


def test_round_atm_uses_strike_step():
    assert round_atm(23_574.9) == 23_550
    assert round_atm(23_575.1) == 23_600
    assert round_atm(81_249.0, step=100) == 81_200


def test_price_at_picks_the_exact_minute(tmp_path):
    p = tmp_path / "bars.parquet"
    pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-06 09:24", "2026-01-06 09:25", "2026-01-06 14:30"]),
        "close": [101.0, 102.5, 0.0],
    }).to_parquet(p)
    assert price_at(p, 9, 25) == 102.5
    assert price_at(p, 10, 0) is None          # minute missing
    assert price_at(p, 14, 30) is None         # non-positive price
    assert price_at(tmp_path / "nope.parquet", 9, 25) is None


def test_project_root_prefers_nifty_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NIFTY_HOME", str(tmp_path))
    paths = importlib.reload(nifty_core.paths)
    assert paths.PROJECT_ROOT == tmp_path.resolve()
    assert paths.DATA_ROOT == tmp_path.resolve() / "data"


def test_project_root_falls_back_to_nearest_pyproject(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    sub = tmp_path / "scripts" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.delenv("NIFTY_HOME", raising=False)
    monkeypatch.chdir(sub)
    paths = importlib.reload(nifty_core.paths)
    assert paths.PROJECT_ROOT == tmp_path.resolve()
    monkeypatch.undo()
    importlib.reload(nifty_core.paths)
