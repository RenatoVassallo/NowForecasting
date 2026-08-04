"""G2 (P1): the code fingerprint covers untracked production files.

`_code_version` hashed ``git diff HEAD``, which is blind to untracked files:
new production modules could change without changing the recorded code
identity (and the exact-chain fingerprint inherited the blind spot). The
fingerprint now covers the HEAD commit, staged and unstaged tracked changes,
and the paths AND contents of untracked non-ignored files, deterministically;
the run manifest records the deviating inventory (relative path, sha256) so
two differing fingerprints can be diagnosed. A hash detects differences but
cannot reconstruct an uncommitted tree: a clean commit remains the preferred
operator condition for an external release.
"""

from __future__ import annotations

import subprocess

import pytest

from pipeline.lib.context import _code_version, dirty_inventory


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "prod.py").write_text("A = 1\n")
    (tmp_path / ".gitignore").write_text("*.parquet\nscratch/\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_clean_tree_has_a_stable_bare_version(repo):
    v1, v2 = _code_version(repo), _code_version(repo)
    assert v1 == v2
    assert "+dirty" not in v1
    assert dirty_inventory(repo) == {}


def test_tracked_edit_changes_the_version(repo):
    base = _code_version(repo)
    (repo / "prod.py").write_text("A = 2\n")
    v = _code_version(repo)
    assert v != base and "+dirty" in v
    assert "prod.py" in dirty_inventory(repo)


def test_staged_change_changes_the_version(repo):
    base = _code_version(repo)
    (repo / "prod.py").write_text("A = 3\n")
    _git(repo, "add", "prod.py")
    assert _code_version(repo) != base
    assert "prod.py" in dirty_inventory(repo)


def test_untracked_production_file_changes_the_version(repo):
    base = _code_version(repo)
    (repo / "newmod.py").write_text("def f(): return 1\n")
    v1 = _code_version(repo)
    assert v1 != base, "an untracked production module must change the identity"
    inv = dirty_inventory(repo)
    assert "newmod.py" in inv and len(inv["newmod.py"]) == 64
    (repo / "newmod.py").write_text("def f(): return 2\n")
    assert _code_version(repo) != v1, "editing the untracked file must change it too"


def test_ignored_files_do_not_enter_the_fingerprint(repo):
    base = _code_version(repo)
    (repo / "cache.parquet").write_bytes(b"binary junk")
    (repo / "scratch").mkdir()
    (repo / "scratch" / "tmp.py").write_text("x = 1\n")
    assert _code_version(repo) == base
    assert dirty_inventory(repo) == {}


def test_creation_order_does_not_affect_the_digest(tmp_path):
    versions = []
    for order in (("a.py", "b.py"), ("b.py", "a.py")):
        d = tmp_path / f"r{order[0][0]}"
        d.mkdir()
        _git(d, "init", "-q")
        _git(d, "config", "user.email", "t@t")
        _git(d, "config", "user.name", "t")
        (d / "base.py").write_text("B = 0\n")
        _git(d, "add", "-A")
        _git(d, "commit", "-qm", "init")
        for name in order:
            (d / name).write_text(f"# {name}\n")
        versions.append(_code_version(d).split("+dirty.")[1])
    assert versions[0] == versions[1]
