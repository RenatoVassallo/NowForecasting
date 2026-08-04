"""F8: the data a run consumes is pinned at run start and verified at the end.

Downstream blocks reload target panels from local caches, so a cache mutation
between the data stage and a later stage could silently change the run. The
guarantee is now verified immutability: every production input file is hashed
into the manifest when the run starts, re-verified before ``_SUCCESS``, and
any drift fails the run (fail closed, no mixed-input publication). Because
``sources/`` is git-ignored, the commit hash cannot capture loader changes:
a content hash over the private loader code is recorded too.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.lib import inputs as pin


def _tree(tmp_path):
    d = tmp_path / "input"
    (d / "peru").mkdir(parents=True)
    (d / "us").mkdir()
    (d / "peru" / "monthly.parquet").write_bytes(b"peru-monthly-v1")
    (d / "us" / "us_monthly.parquet").write_bytes(b"us-v1")
    return d


def test_pin_records_every_file_with_hash(tmp_path):
    root = _tree(tmp_path)
    pins = pin.pin_inputs(root=root)
    assert set(pins) == {"peru/monthly.parquet", "us/us_monthly.parquet"}
    assert all(len(v["sha256"]) == 64 and v["bytes"] > 0 for v in pins.values())


def test_mutation_after_pin_fails_verification(tmp_path):
    root = _tree(tmp_path)
    pins = pin.pin_inputs(root=root)
    (root / "us" / "us_monthly.parquet").write_bytes(b"us-v2 MUTATED")
    with pytest.raises(RuntimeError, match="us_monthly.parquet"):
        pin.verify_inputs(pins, root=root)


def test_deleted_input_fails_verification(tmp_path):
    root = _tree(tmp_path)
    pins = pin.pin_inputs(root=root)
    (root / "peru" / "monthly.parquet").unlink()
    with pytest.raises(RuntimeError, match="monthly.parquet"):
        pin.verify_inputs(pins, root=root)


def test_unchanged_inputs_verify_quietly(tmp_path):
    root = _tree(tmp_path)
    pins = pin.pin_inputs(root=root)
    pin.verify_inputs(pins, root=root)          # no raise


def test_sources_code_hash_tracks_loader_changes(tmp_path):
    s = tmp_path / "sources"
    s.mkdir()
    (s / "bcrp.py").write_text("def fetch(): return 1\n")
    h1 = pin.sources_code_sha(root=s)
    (s / "bcrp.py").write_text("def fetch(): return 2\n")
    h2 = pin.sources_code_sha(root=s)
    assert h1 != h2 and len(h1) == 64
    assert pin.sources_code_sha(root=tmp_path / "absent") == "absent"
