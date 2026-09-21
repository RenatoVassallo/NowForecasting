"""X13 binary discovery: MacroPy bundles the binary; overrides still win.

MacroPy ships the Census x13as per platform (``MacroPy.x13.x13_path``), so
a fresh checkout needs no system install. Explicit operator choices (a
binary on PATH, the X13PATH env var) take precedence over the bundled
copy, and everything degrades to the legacy filesystem candidates when
MacroPy is absent or has no binary for the platform.
"""

from __future__ import annotations

import sys
import types

import pytest

from core import preprocess


def _fake_macropy_x13(monkeypatch, exe):
    mod = types.ModuleType("MacroPy.x13")
    mod.x13_path = lambda: exe
    monkeypatch.setitem(sys.modules, "MacroPy.x13", mod)


def _exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    path.chmod(0o755)
    return path


def test_macropy_bundled_binary_is_discovered(tmp_path, monkeypatch):
    exe = _exe(tmp_path / "bundle" / "x13as")
    _fake_macropy_x13(monkeypatch, exe)
    monkeypatch.delenv("X13PATH", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "no-binaries-here"))
    assert preprocess.locate_x13_binary() == exe.resolve()


def test_x13path_env_beats_the_bundled_binary(tmp_path, monkeypatch):
    bundled = _exe(tmp_path / "bundle" / "x13as")
    override = _exe(tmp_path / "override" / "x13as")
    _fake_macropy_x13(monkeypatch, bundled)
    monkeypatch.setenv("X13PATH", str(override))
    monkeypatch.setenv("PATH", str(tmp_path / "no-binaries-here"))
    assert preprocess.locate_x13_binary() == override.resolve()


def test_macropy_absence_degrades_to_the_old_error(tmp_path, monkeypatch):
    broken = types.ModuleType("MacroPy.x13")
    broken.x13_path = lambda: (_ for _ in ()).throw(FileNotFoundError("no bundle"))
    monkeypatch.setitem(sys.modules, "MacroPy.x13", broken)
    monkeypatch.delenv("X13PATH", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "no-binaries-here"))
    monkeypatch.setattr(preprocess, "LOCAL_X13", tmp_path / "absent" / "x13as")
    with pytest.raises(FileNotFoundError, match="X13 binary not found"):
        preprocess.locate_x13_binary()
