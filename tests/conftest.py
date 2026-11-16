"""Gemeinsame Fixtures: das Handbeispiel (n = 8, von Hand nachgerechnet) und der Importpfad."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mwl_exact as E  # noqa: E402


@pytest.fixture
def hand():
    """x = -7, -5, ..., 7 und y = 3, 0, 8, 3, 5, 6, 9, 6 (Summe x = 0, Summe x² = 168, Summe y = 40, Summe y² = 260)."""
    return list(E.HAND_X), list(E.HAND_Y)
