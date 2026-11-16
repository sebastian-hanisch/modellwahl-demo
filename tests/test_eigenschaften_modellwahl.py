"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Beziehungen, die für die Wahl nach Kriterium gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 20 bis 50 Zufallsdatensätze mit festem Zufalls-Seed. Gilt eine Eigenschaft nur unter Voraussetzungen, stehen sie im Test (BIC ≤ AIC nur ab n ≥ 8, Skalierung nur mit Achsenabschnitt im Modell)."""

import numpy as np
import pytest

import mwl_evaluation as ev
import mwl_generator as G
import mwl_rng as R
import mwl_selection as S
import mwl_theory as T


def _choices(x, y, deg_max=6, basis="legendre"):
    n = x.shape[1]
    fam = S.fit_family(x, y, deg_max, basis)
    cr = S.criteria_from_rss(fam.rss, n, np.arange(deg_max + 1) + 1)
    cr["LOO"] = fam.press / n
    return fam, cr, {k: S.choose(v)[0] for k, v in cr.items()}


def test_scaling_and_shifting_y_does_not_change_the_choice():
    x, y = G.poly_data(40, 50, 2.0, 71)
    _, _, base = _choices(x, y)
    for c in (0.001, 7.5):
        _, _, scaled = _choices(x, c * y)
        _, _, shifted = _choices(x, y + 100.0 * c)
        for k in base:
            assert np.array_equal(base[k], scaled[k]) and np.array_equal(base[k], shifted[k]), (k, c)


def test_reordering_the_observations_changes_nothing():
    x, y = G.poly_data(30, 40, 2.0, 73)
    fam, cr, base = _choices(x, y)
    perm = np.random.default_rng(1).permutation(40)
    fam2, cr2, other = _choices(x[:, perm], y[:, perm])
    assert np.allclose(fam.rss, fam2.rss, rtol=1e-9) and np.allclose(fam.press, fam2.press, rtol=1e-7)
    for k in base:
        assert np.array_equal(base[k], other[k]), k


def test_change_of_basis_gives_the_same_residual_sums_and_choices():
    x, y = G.poly_data(30, 60, 1.5, 79)
    fam_l, _, ch_l = _choices(x, y, 5, "legendre")
    fam_m, _, ch_m = _choices(x, y, 5, "monom")
    assert np.allclose(fam_l.rss, fam_m.rss, rtol=1e-8)
    for k in ch_l:
        assert np.array_equal(ch_l[k], ch_m[k]), k


def test_affine_change_of_x_gives_the_same_choice_with_monomials():
    x, y = G.poly_data(30, 60, 1.5, 83)
    _, _, a = _choices(x, y, 5, "monom")
    _, _, b = _choices(3.0 * x + 2.0, y, 5, "monom")
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_doubling_every_observation_doubles_the_fit_term_but_not_the_penalty():
    x, y = G.poly_data(20, 35, 2.0, 89)
    fam = S.fit_family(x, y, 5)
    fam2 = S.fit_family(np.concatenate([x, x], axis=1), np.concatenate([y, y], axis=1), 5)
    assert np.allclose(fam2.rss, 2 * fam.rss, rtol=1e-8)                        # RSS und n verdoppeln sich
    base1 = 35 * np.log(fam.rss / 35)
    base2 = 70 * np.log(fam2.rss / 70)
    assert np.allclose(base2, 2 * base1, rtol=1e-8)                              # n ln(RSS/n) verdoppelt sich, die Differenz zweier Modelle ebenfalls
    aic1 = S.criteria_from_rss(fam.rss, 35, np.arange(6) + 1)["AIC"]
    aic2 = S.criteria_from_rss(fam2.rss, 70, np.arange(6) + 1)["AIC"]
    assert np.allclose(aic1 - base1, 2.0 * (np.arange(6) + 2)) and np.allclose(aic2 - base2, 2.0 * (np.arange(6) + 2))     # die Strafe 2K hängt nicht an n


def test_aicc_approaches_aic_with_growing_n():
    k = np.array([2, 4, 6])
    gaps = []
    for n in (20, 100, 1000, 10000):
        cr = S.criteria_from_rss(np.full((1, 3), 3.0), n, k)
        gaps.append(np.max(cr["AICc"] - cr["AIC"]))
    assert gaps == sorted(gaps, reverse=True) and gaps[-1] < 0.02 and gaps[0] > 0.5       # Zuschlag 2K(K+1)/(n-K-1) = 0,0112 bei K = 7, n = 10 000


def test_bic_penalty_per_parameter_exceeds_the_aic_penalty_exactly_from_n_8():
    for n in range(1, 60):
        assert (np.log(n) > 2.0) == (n >= 8), n


def test_bic_never_chooses_a_larger_nested_model_than_aic_from_n_8():
    for n in (8, 12, 30, 200):
        x, y = G.poly_data(40, n, 2.0, 97 + n)
        deg_max = min(6, n - 5)
        fam = S.fit_family(x, y, deg_max)
        cr = S.criteria_from_rss(fam.rss, n, np.arange(deg_max + 1) + 1)
        assert np.all(S.choose(cr["BIC"])[0] <= S.choose(cr["AIC"])[0]), n


def test_k_fold_with_k_equal_n_is_leave_one_out():
    x, y = G.poly_data(15, 14, 2.0, 101)
    fam = S.fit_family(x, y, 4)
    ident = np.tile(np.arange(14), (15, 1))
    cv = S.cv_family(fam.design, y, ident, 14)
    assert np.allclose(cv, fam.press / 14, rtol=1e-8)


def test_the_cross_validation_choice_depends_on_the_fold_and_only_the_count_is_claimed():
    x, y = G.poly_data(200, 40, 3.0, 103)
    fam = S.fit_family(x, y, 8)
    f1 = R.ranks_array(R.derive_seed(103, 7), 200, 40) % 5
    f2 = R.ranks_array(R.derive_seed(103, 8), 200, 40) % 5
    c1, c2 = (S.choose(S.cv_family(fam.design, y, f, 5))[0] for f in (f1, f2))
    differ = int(np.sum(c1 != c2))
    assert 0 < differ < 200                                                        # nur eine Zählung, nie Gleichheit


def test_data_splitting_uses_disjoint_rows_that_cover_everything():
    ia, ib = ev.split_halves(50, 31, 7)
    assert ia.shape == (50, 15) and ib.shape == (50, 16)
    for a, b in zip(ia, ib):
        assert not set(a) & set(b) and set(a) | set(b) == set(range(31))
    assert not np.array_equal(ia[0], ia[1])


def test_a_chosen_model_never_has_a_larger_rss_than_a_contained_smaller_one():
    x, y = G.poly_data(60, 30, 3.0, 107)
    fam = S.fit_family(x, y, 8)
    cr = S.criteria_from_rss(fam.rss, 30, np.arange(9) + 1)
    assert np.all(np.diff(fam.rss, axis=1) <= 1e-9 * fam.rss[:, :-1])              # geschachtelte Modelle: RSS fällt mit dem Grad
    for name in ("AIC", "AICc", "BIC"):
        idx = S.choose(cr[name])[0]
        for r in range(60):
            assert np.all(fam.rss[r, idx[r]] <= fam.rss[r, : idx[r]] * (1 + 1e-9))


def test_same_seed_gives_the_same_frequencies_and_other_seed_other_ones():
    a, b, c = ev.selection_frequency(40, 2.0, 100, 5), ev.selection_frequency(40, 2.0, 100, 5), ev.selection_frequency(40, 2.0, 100, 6)
    assert a.share == b.share and a.pe_mean == b.pe_mean
    assert a.share != c.share


def test_frequencies_of_the_true_degree_do_not_change_with_the_order_of_independent_runs():
    """Die ersten Wiederholungen eines großen Laufs sind dieselben wie die eines kleinen: Anteile über die ersten 100 stimmen mit dem kleinen Lauf überein."""
    x_small, y_small = G.poly_data(100, 40, 2.0, 5)
    x_big, y_big = G.poly_data(300, 40, 2.0, 5)
    assert np.array_equal(S.choose(S.criteria_from_rss(S.fit_family(x_small, y_small, 8).rss, 40, np.arange(9) + 1)["AIC"])[0],
                          S.choose(S.criteria_from_rss(S.fit_family(x_big, y_big, 8).rss, 40, np.arange(9) + 1)["AIC"])[0][:100])


def test_inclusion_probability_of_noise_falls_with_n_toward_the_limit_from_above():
    p = [T.p_include_noise(n, 4, "AIC") for n in (10, 30, 100, 1000, 10000)]
    assert p == sorted(p, reverse=True) and p[-1] > 0.1573                             # P(χ²₁ > 2) ist der Grenzwert
    q = [T.p_include_noise(n, 4, "BIC") for n in (30, 100, 1000, 10000)]
    assert q == sorted(q, reverse=True) and q[-1] < 0.01


def test_more_signal_makes_the_larger_model_more_likely():
    lam = np.linspace(0, 40, 50)
    for rule in ("AIC", "BIC"):
        p = np.asarray(T.p_choose_bigger(300, 4, lam, rule))
        assert np.all(np.diff(p) > 0) and p[0] == pytest.approx(T.p_include_noise(300, 4, rule), abs=1e-12)
    assert np.all(np.asarray(T.p_choose_bigger(300, 4, lam, "AIC")) >= np.asarray(T.p_choose_bigger(300, 4, lam, "BIC")))


def test_noncentral_f_at_lambda_zero_is_continuous_with_the_central_f():
    """scipy.stats.ncf liefert bei nc = 0 unbrauchbare Werte (negative Wahrscheinlichkeit): `p_choose_bigger` weicht dort auf das zentrale F aus; stetig bei λ → 0."""
    for n in (30, 300):
        for rule in ("AIC", "BIC"):
            p0 = float(T.p_choose_bigger(n, 4, 0.0, rule))
            assert p0 == pytest.approx(T.p_include_noise(n, 4, rule), abs=1e-12)
            assert float(T.p_choose_bigger(n, 4, 1e-4, rule)) == pytest.approx(p0, abs=1e-3) and float(T.p_choose_bigger(n, 4, 1e-4, rule)) >= p0
            assert 0.0 <= float(T.p_choose_bigger(n, 4, 1e-9, rule)) <= 1.0
