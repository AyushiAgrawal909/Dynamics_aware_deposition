"""Command-line interface and the packaged example configurations."""

import json
from importlib import resources

import pytest

from dads.cli import EXAMPLES, main
from dads.config import load, run

FAST = ["two_stream", "landau", "ion_acoustic"]
SLOW = ["beam_plasma", "two_stream_3d"]


def _config(name, tmp_path):
    text = resources.files("dads").joinpath("configs", f"{name}.yaml").read_text()
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    return path


def test_select_writes_outputs(tmp_path, capsys):
    cfg = _config("two_stream", tmp_path)
    out = tmp_path / "out"
    assert main(["select", str(cfg), "-o", str(out), "-q"]) == 0
    text = capsys.readouterr().out
    assert "RECOMMENDED KERNEL: cic+binomial(p=4,n=8)" in text
    data = json.loads((out / "selection.json").read_text())
    assert data["best"]["filter"]["p"] == 4
    for f in ("summary.txt", "selection.pdf", "selection.png", "best_transfer_fftn.npy"):
        assert (out / f).exists()


def test_tolerance_override_changes_choice(tmp_path, capsys):
    cfg = _config("two_stream", tmp_path)
    main(["select", str(cfg), "--tolerance", "0.02", "--no-plot", "-q"])
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("RECOMMENDED"))
    assert "binomial(p=4,n=8)" not in line         # R_gamma = 0.9555 is outside 2 %


def test_stencil_and_example_commands(tmp_path, capsys):
    assert main(["stencil", "--p", "0", "--n", "1"]) == 0
    out = capsys.readouterr().out
    assert "+0.250000000000" in out and "+0.500000000000" in out
    assert main(["example"]) == 0
    listing = capsys.readouterr().out
    assert all(name in listing for name in EXAMPLES)


@pytest.mark.parametrize("name", FAST)
def test_fast_examples_select_something(name, tmp_path):
    res = run(load(_config(name, tmp_path)))
    assert res.best is not None and res.best.admissible


@pytest.mark.slow
@pytest.mark.parametrize("name", SLOW)
def test_slow_examples_select_something(name, tmp_path):
    res = run(load(_config(name, tmp_path)))
    assert res.best is not None and res.best.admissible
    assert len(res.modes) >= 2      # auto-protection found several modes


@pytest.mark.slow
def test_validate_command():
    assert main(["validate"]) == 0
