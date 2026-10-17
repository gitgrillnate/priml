"""Tests for the golden regeneration flags."""

from __future__ import annotations

import pytest

from priml.testing import regenerate


def test_flags_default_off_and_follow_the_command_line(
    pytestconfig: pytest.Config,
) -> None:
    assert regenerate.golden() is bool(pytestconfig.getoption("--regenerate-golden"))
    assert regenerate.b4b() is bool(pytestconfig.getoption("--regenerate-b4b"))


def test_overriding_one_flag_leaves_the_other(monkeypatch: pytest.MonkeyPatch) -> None:
    regenerate.override(monkeypatch, golden=False, b4b=False)
    regenerate.override(monkeypatch, golden=True)
    assert regenerate.golden()
    assert not regenerate.b4b()
    regenerate.override(monkeypatch, b4b=True)
    assert regenerate.golden()
    assert regenerate.b4b()


def test_suppressed_turns_both_flags_off_then_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regenerate.override(monkeypatch, golden=True, b4b=True)
    with regenerate.suppressed():
        assert not regenerate.golden()
        assert not regenerate.b4b()
    assert regenerate.golden()
    assert regenerate.b4b()


def test_forced_turns_the_tensor_flag_on_then_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regenerate.override(monkeypatch, golden=False, b4b=False)
    with regenerate.forced():
        assert regenerate.b4b()
        assert not regenerate.golden()
    assert not regenerate.b4b()


def test_options_register_once_per_parser() -> None:
    parser = pytest.Parser(_ispytest=True)
    regenerate.add_options(parser)
    regenerate.add_options(parser)
    options = parser.parse(["--regenerate-golden", "--regenerate-b4b"])
    assert vars(options) == {
        "regenerate_golden": True,
        "regenerate_b4b": True,
        "file_or_dir": [],
    }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
