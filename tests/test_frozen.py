"""Die vorgerechneten Messreihen (`mwl_frozen.json`) passen zur Suite: die vier Zellen werden voll neu gerechnet, die Wahl ohne Daten und der Irrfahrt-Grenzwert mit weniger
Wiederholungen und im Band verglichen (Zufall ohne C-Bibliothek, aber Gleichstände und letzte Bits können je Plattform kippen). Vollständiger Abgleich: `python tools/mwl_make_frozen.py --check`."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import mwl_make_frozen as M  # noqa: E402

import mwl_theory as T  # noqa: E402

FROZEN = json.loads((ROOT / "mwl_frozen.json").read_text(encoding="utf-8"))


def test_structure_of_the_frozen_file():
    assert set(FROZEN) == {"zellen", "pool", "grenzwert"}
    assert set(FROZEN["zellen"]) == {M.cell_key(s, n) for s, n in M.ZELLEN}
    assert FROZEN["pool"]["reps"] == M.POOL_REPS and sorted(int(n) for n in FROZEN["pool"]["n"]) == sorted(M.POOL_NS)
    assert FROZEN["grenzwert"]["reps"] == M.LIMIT_REPS
    for cell in FROZEN["zellen"].values():
        assert cell["reps"] == M.ZELLEN_REPS and set(cell["shares"]) == {"AIC", "AICc", "BIC", "CV5", "LOO"}
        assert all(abs(sum(v) - 100.0) < 1e-3 for v in cell["shares"].values())
    for row in FROZEN["pool"]["n"].values():
        assert all(abs(sum(v) - 100.0) < 1e-3 for v in row.values())


@pytest.mark.parametrize("sigma, n", M.ZELLEN)
def test_cells_recomputed_equal_the_frozen_values_within_the_band(sigma, n):
    key = M.cell_key(sigma, n)
    assert M.differences({key: FROZEN["zellen"][key]}, {key: M.compute_cell(sigma, n)}) == []


@pytest.mark.parametrize("n", [30, 300])
def test_pool_walk_recomputed_with_fewer_repetitions_is_inside_the_statistical_band(n):
    reps = 200_000
    new = T.pool_walk(n, reps, 77)
    for rule in ("AIC", "AICc", "BIC"):
        for j in range(6):
            old = FROZEN["pool"]["n"][str(n)][rule][j]
            p = max(min(old / 100.0, 1 - 1e-6), 1e-6)
            se = 100.0 * np.sqrt(p * (1 - p) * (1.0 / reps + 1.0 / M.POOL_REPS))
            assert abs(new[rule][j] - old) < 5 * se + 0.02, (rule, j, new[rule][j], old)


def test_random_walk_limit_recomputed_is_inside_the_band():
    assert abs(T.irrfahrt_limit(300_000, 77) - FROZEN["grenzwert"]["wert"]) < 0.5


def test_the_comparison_detects_a_changed_value_and_a_missing_key():
    changed = copy.deepcopy(FROZEN)
    changed["zellen"]["s1_n300"]["shares"]["BIC"][1] += 1.0
    assert M.differences(FROZEN, changed)
    missing = copy.deepcopy(FROZEN)
    del missing["grenzwert"]
    assert M.differences(FROZEN, missing)
    assert M.differences(FROZEN, copy.deepcopy(FROZEN)) == []
