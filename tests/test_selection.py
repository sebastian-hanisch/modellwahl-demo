"""Einheiten von mwl_selection.py an Mini-Instanzen von Hand und gegen die Brüche des Handbeispiels."""

import numpy as np
import pytest

import mwl_exact as E
import mwl_selection as S


def _hand_arrays(hand):
    x, y = hand
    return np.array([x], float), np.array([y], float)


def test_legendre_and_monomial_designs_by_hand():
    x = np.array([0.5])
    v = S.legendre_design(x, 3)[0]
    assert np.allclose(v, [1.0, 0.5, (3 * 0.25 - 1) / 2, (5 * 0.125 - 3 * 0.5) / 2])      # P2 = -0,125, P3 = -0,4375
    assert np.allclose(S.monomial_design(x, 3)[0], [1.0, 0.5, 0.25, 0.125])


def test_family_equals_the_exact_fractions_on_the_hand_example(hand):
    x, y = _hand_arrays(hand)
    rows = E.table(hand[0], hand[1], 2)
    for basis in ("legendre", "monom"):
        fam = S.fit_family(x, y, 2, basis)
        assert np.allclose(fam.rss[0], [float(r["rss"]) for r in rows], rtol=1e-10)
        assert np.allclose(fam.press[0], [float(r["press"]) for r in rows], rtol=1e-9)


def test_criteria_equal_the_exact_values_on_the_hand_example(hand):
    x, y = _hand_arrays(hand)
    fam = S.fit_family(x, y, 2)
    for count_variance in (True, False):
        rows = E.table(hand[0], hand[1], 2, count_variance)
        cr = S.criteria_from_rss(fam.rss, 8, [1, 2, 3], count_variance)
        for name in ("AIC", "AICc", "BIC"):
            assert np.allclose(cr[name][0], [float(r[name]) for r in rows], atol=1e-9), (name, count_variance)


def test_choices_on_the_hand_example(hand):
    x, y = _hand_arrays(hand)
    fam = S.fit_family(x, y, 2)
    cr = S.criteria_from_rss(fam.rss, 8, [1, 2, 3])
    assert [int(S.choose(cr[k])[0][0]) for k in ("AIC", "BIC", "AICc")] == [1, 1, 0]
    assert int(S.choose(fam.press)[0][0]) == 1


def test_choose_ties_go_to_the_smallest_index_and_are_counted():
    crit = np.array([[3.0, 1.0, 1.0, 2.0], [5.0, 4.0, 4.0 + 1e-12, 4.5], [2.0, 2.5, 3.0, 4.0]])
    idx, tie = S.choose(crit)
    assert idx.tolist() == [1, 1, 0] and tie.tolist() == [True, True, False]


def test_choose_handles_minus_infinity_nan_and_infinity():
    idx, tie = S.choose(np.array([[1.0, -np.inf, -np.inf], [np.nan, 2.0, 1.0], [np.inf, np.inf, np.inf]]))
    assert idx.tolist() == [1, 2, 0] and tie.tolist() == [True, False, True]


def test_aicc_is_infinite_when_n_minus_k_minus_one_is_not_positive():
    cr = S.criteria_from_rss(np.array([[1.0, 1.0, 1.0]]), 8, [5, 6, 7])          # K = 6, 7, 8: n - K - 1 = 1, 0, -1
    assert np.isfinite(cr["AICc"][0, 0]) and np.isinf(cr["AICc"][0, 1]) and np.isinf(cr["AICc"][0, 2])


def test_rss_zero_gives_minus_infinity_and_the_choice_stays_defined():
    cr = S.criteria_from_rss(np.array([[4.0, 0.0, 0.0]]), 10, [1, 2, 3])
    assert np.isneginf(cr["AIC"][0, 1]) and np.isneginf(cr["BIC"][0, 2])
    assert int(S.choose(cr["AIC"])[0][0]) == 1                                    # Gleichstand bei -inf: kleinster Grad


def test_cross_validation_by_hand():
    """x = 0, 1, 2, 3, y = 1, 3, 5, 7, Faltungen {0, 1} und {2, 3}. Grad 0: Prüffaltung 1 gegen Mittel 6 gibt 25 + 9, Faltung 2 gegen Mittel 2 gibt 9 + 25: CV = 68/4 = 17. Grad 1 trifft die Gerade: CV = 0."""
    v = S.monomial_design(np.array([[0.0, 1.0, 2.0, 3.0]]), 1)
    y = np.array([[1.0, 3.0, 5.0, 7.0]])
    cv = S.cv_family(v, y, np.array([[0, 0, 1, 1]]), 2)
    assert cv[0, 0] == pytest.approx(17.0) and cv[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_cross_validation_needs_enough_observations():
    v = S.monomial_design(np.array([[0.0, 1.0, 2.0]]), 1)
    assert np.all(np.isinf(S.cv_family(v, np.array([[1.0, 2.0, 3.0]]), np.array([[0, 1, 2]]), 5)))        # n < Faltungen
    cv = S.cv_family(v, np.array([[1.0, 2.0, 3.0]]), np.array([[0, 1, 2]]), 3)                              # Training mit 2 Punkten: Grad 1 geht, Grad 0 geht
    assert np.all(np.isfinite(cv))


def test_subsets_are_complete_and_contain_the_intercept():
    subs = S.all_subsets(5)
    assert len(subs) == 128 and all(s[0] == 0 for s in subs)
    assert len({tuple(s) for s in subs}) == 128 and (0, 1, 2) in {tuple(s) for s in subs} and tuple(range(8)) in {tuple(s) for s in subs}
    assert {tuple(s) for s in S.all_subsets(0)} == {(0,), (0, 1), (0, 2), (0, 1, 2)}


def test_subset_tables_equal_a_least_squares_fit_per_subset():
    rg = np.random.default_rng(3)
    x = np.concatenate([np.ones((4, 40, 1)), rg.normal(size=(4, 40, 3))], axis=2)
    y = rg.normal(size=(4, 40))
    subs = S.all_subsets(0)                                     # Achsenabschnitt plus zwei Merkmale: vier Teilmengen
    rss, press, coef = S.subset_tables(x[:, :, :3], y, subs, with_loo=True)
    for r in range(4):
        for i, s in enumerate(subs):
            b = np.linalg.lstsq(x[r][:, s], y[r], rcond=None)[0]
            e = y[r] - x[r][:, s] @ b
            h = np.einsum("nk,kj,nj->n", x[r][:, s], np.linalg.inv(x[r][:, s].T @ x[r][:, s]), x[r][:, s])
            assert rss[r, i] == pytest.approx(e @ e, rel=1e-9)
            assert press[r, i] == pytest.approx(np.sum((e / (1 - h)) ** 2), rel=1e-9)
            assert np.allclose(coef[r, i, s], b) and np.all(coef[r, i, np.setdiff1d(np.arange(3), s)] == 0)


def test_fit_subset_standard_errors_by_the_textbook_formula():
    rg = np.random.default_rng(5)
    x = np.concatenate([np.ones((2, 30, 1)), rg.normal(size=(2, 30, 2))], axis=2)
    y = rg.normal(size=(2, 30))
    s = np.array([0, 2])
    b, se, df = S.fit_subset(x, y, s)
    assert df == 28
    for r in range(2):
        xs = x[r][:, s]
        bb = np.linalg.solve(xs.T @ xs, xs.T @ y[r])
        s2 = np.sum((y[r] - xs @ bb) ** 2) / 28
        assert np.allclose(b[r], bb) and np.allclose(se[r], np.sqrt(s2 * np.diag(np.linalg.inv(xs.T @ xs))))


def test_degree_larger_than_n_minus_one_is_rejected():
    with pytest.raises(ValueError):
        S.fit_family(np.zeros((1, 5)), np.zeros((1, 5)), 5)


def test_prediction_error_of_the_true_degree_is_small_and_of_degree_zero_is_the_variance_of_the_curve():
    x = np.linspace(-1, 1, 200)[None, :]
    y = 0.5 + x - 2 * x ** 2 + 3 * x ** 3                          # ohne Rauschen
    fam = S.fit_family(x, y, 4)
    grid = np.linspace(-1, 1, 201)
    truth = 0.5 + grid - 2 * grid ** 2 + 3 * grid ** 3
    pe = S.prediction_error(fam, S.legendre_design(grid, 4), truth)[0]
    assert pe[3] < 1e-6 and pe[4] < 1e-6 and pe[0] > pe[1] > pe[2] > pe[3]
