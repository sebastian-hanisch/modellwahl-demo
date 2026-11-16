"""Erzeuger: Kundeneffekte gegen numpy, Verteilung der Merkmale, wahre Funktionen, Reproduzierbarkeit."""

import numpy as np
import pytest

import mwl_generator as G


def test_customer_effects_equal_the_numpy_draw_of_the_verweilzeit_demo():
    assert np.allclose(G.UK, np.random.default_rng(12345).normal(0.0, 0.3, 12), atol=1e-15)    # Quelle der Zahlen: default_rng(12345).normal(0, 0,3, 12)


def test_error_variance_without_customers_in_the_model():
    assert G.VAR_KUNDEN == pytest.approx(0.1319, abs=5e-5)
    assert G.SIGMA2_IMPORT == pytest.approx(0.55 ** 2 + 0.1319, abs=5e-5) and np.sqrt(G.SIGMA2_IMPORT) == pytest.approx(0.6591, abs=5e-5)
    assert G.SIGMA2_EXPORT == pytest.approx(0.25 ** 2 + 0.09 * 0.1319, abs=5e-5)


def test_poly_truth_by_hand():
    assert G.poly_truth(0.0) == 0.5 and G.poly_truth(1.0) == pytest.approx(0.5 + 1 - 2 + 3) and G.poly_truth(-1.0) == pytest.approx(0.5 - 1 - 2 - 3)


def test_poly_data_has_the_stated_distribution():
    x, y = G.poly_data(200, 500, 2.0, 4)
    assert x.min() >= -1 and x.max() <= 1 and abs(x.mean()) < 0.01 and abs(x.var() - 1 / 3) < 0.01
    resid = y - G.poly_truth(x)
    assert abs(resid.mean()) < 0.02 and abs(resid.std() - 2.0) < 0.02


def test_import_data_moments_and_error_variance():
    x, y = G.import_data(400, 500, 5, 11)
    assert x.shape == (400, 500, 8) and np.all(x[:, :, 0] == 1)
    assert abs(x[:, :, 1].mean() - 0.25) < 0.005 and abs(x[:, :, 2].mean() - 2 / 7) < 0.005
    assert set(np.unique(x[:, :, 1])) == {0.0, 1.0} and abs(x[:, :, 3:].mean()) < 0.005 and abs(x[:, :, 3:].std() - 1.0) < 0.005
    resid = y - x[:, :, :3] @ G.import_beta(0)[:3]
    assert abs(resid.mean()) < 0.01 and resid.var() == pytest.approx(G.SIGMA2_IMPORT, rel=0.02)


def test_import_moment_matrix_by_hand():
    m = G.import_moments(2)
    assert m.shape == (5, 5) and m[0, 0] == 1 and m[0, 1] == 0.25 and m[1, 1] == 0.25 and m[1, 2] == pytest.approx(0.25 * 2 / 7) and m[2, 2] == pytest.approx(2 / 7)
    assert m[3, 3] == 1 and m[0, 3] == 0 and m[1, 3] == 0
    x, _ = G.import_data(1, 400000, 2, 5)
    assert np.allclose(x[0].T @ x[0] / 400000, m, atol=0.01)       # Stichprobenmoment gegen exakte Momentenmatrix


def test_export_data_follows_the_stated_curve():
    v, y = G.export_data(300, 400, 8)
    assert v.min() >= 1 and v.max() <= 10 and abs(v.mean() - 5.5) < 0.05
    resid = y - np.log(0.6 + 0.85 * v)
    assert abs(resid.mean() - 0.3 * G.UK.mean()) < 0.01 and resid.var() == pytest.approx(G.SIGMA2_EXPORT, rel=0.03)
    assert G.export_x(1.0) == -1.0 and G.export_x(10.0) == 1.0


def test_first_repetitions_do_not_depend_on_the_number_of_repetitions():
    x1, y1 = G.poly_data(3, 50, 1.0, 9)
    x2, y2 = G.poly_data(10, 50, 1.0, 9)
    assert np.array_equal(x1, x2[:3]) and np.array_equal(y1, y2[:3])
    a, b = G.import_data(2, 40, 5, 3), G.import_data(6, 40, 5, 3)
    assert np.array_equal(a[0], b[0][:2]) and np.array_equal(a[1], b[1][:2])


def test_same_seed_same_data_other_seed_other_data():
    assert np.array_equal(G.poly_data(5, 30, 1.0, 1)[1], G.poly_data(5, 30, 1.0, 1)[1])
    assert not np.array_equal(G.poly_data(5, 30, 1.0, 1)[1], G.poly_data(5, 30, 1.0, 2)[1])
