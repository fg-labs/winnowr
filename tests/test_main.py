from collections.abc import Callable
from inspect import isfunction

import pytest
from defopt import signature
from pytest import CaptureFixture
from pytest import MonkeyPatch

from winnowr import __version__
from winnowr import main


@pytest.mark.parametrize("tool", main._tools)
def test_tools_are_defined(tool: Callable[..., None]) -> None:
    """Test that all command line tools passed to defopt are defined functions."""
    assert isfunction(tool)


@pytest.mark.parametrize("tool", main._tools)
def test_tools_have_valid_docstrings(tool: Callable[..., None]) -> None:
    """Test that all command line tools have a valid defopt docstring."""
    try:
        signature(tool)
    except TypeError:
        raise AssertionError(f"defopt could not parse docstring for {tool.__name__}") from None


def test_cli_version_flag_prints_version_and_exits(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """`winnowr --version` should print the package version and exit cleanly."""
    monkeypatch.setattr("sys.argv", ["winnowr", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        main.run()
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__
