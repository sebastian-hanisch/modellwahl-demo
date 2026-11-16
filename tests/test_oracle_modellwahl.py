"""Orakel-Tests: jede Kernroutine gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, §6; DEMO-PLAYBOOK §12).

Rechenwege der Gegenseite: Brüche (`fractions`) und `decimal` für Anpassung, Hebel, Leave-one-out und die Kriterien; `numpy.polynomial.legendre` und `numpy.linalg.lstsq` für die Basis und die
Anpassung; scikit-learn (`LeaveOneOut`, `PredefinedSplit`, `LinearRegression`) für die Kreuzvalidierung; `scipy.stats.f`/`ncf` für die Aufnahmewahrscheinlichkeiten; direkte Simulation mit
Anpassung gegen die Quadratsummen-Rechnung ohne Daten. Simulationen werden als Band geprüft (4 Standardfehler plus Rundung), nie bitgleich (plattformabhängig: letztes Bit, Gleichstände, BLAS)."""

import decimal
import random
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import stats

import mwl_evaluation as ev
import mwl_exact as E
import mwl_generator as G
import mwl_rng as R
import mwl_selection as S
import mwl_theory as T


def _band(p, reps, k=4.0, extra=0.0):
    """k Standardfehler eines Anteils p (in Prozent) bei `reps` Wiederholungen, plus eine feste Zugabe."""
    return k * 100.0 * np.sqrt(max(p / 100.0 * (1 - p / 100.0), 1e-4) / reps) + extra


# --- Das Kriterium ist eine F-Schwelle (Gleichung, exakt mit decimal geprüft) -------------------------------------------------------------

def test_criterion_decision_equals_the_f_threshold_in_random_instances():
    ctx = decimal.Context(prec=50)
    rnd = random.Random(20261116)

    def ln(q):
        return ctx.ln(decimal.Decimal(q.numerator) / decimal.Decimal(q.denominator))

    for _ in range(400):
        n = rnd.randint(6, 40)
        k_small = rnd.randint(1, min(4, n - 3))
        ra = Fr(rnd.randint(50, 400), 7)
        rb = ra * Fr(rnd.randint(40, 99), 100)
        f = (ra - rb) / (rb / (n - k_small - 1))
        fd = decimal.Decimal(f.numerator) / decimal.Decimal(f.denominator)
        gain = n * (ln(ra) - ln(rb))
        thr_aic = (n - k_small - 1) * (ctx.exp(decimal.Decimal(2) / n) - 1)
        thr_bic = (n - k_small - 1) * (ctx.exp(ctx.ln(decimal.Decimal(n)) / n) - 1)
        assert (gain > 2) == (fd > thr_aic) and (gain > ctx.ln(decimal.Decimal(n))) == (fd > thr_bic)
        assert abs(float(thr_aic) - T.f_threshold(n, k_small + 1, "AIC")) < 1e-9 and abs(float(thr_bic) - T.f_threshold(n, k_small + 1, "BIC")) < 1e-9


# --- Gestapelte Gleitkommarechnung gegen Brüche und lstsq ----------------------------------------------------------------------------------

def test_family_rss_and_press_equal_fractions_on_random_integer_data():
    rnd = random.Random(7)
    for _ in range(25):
        n = rnd.randint(7, 12)
        x = rnd.sample(range(-9, 10), n)
        y = [rnd.randint(-8, 8) for _ in range(n)]
        fam = S.fit_family(np.array([x], float), np.array([y], float), 3)
        for d in range(4):
            assert fam.rss[0, d] == pytest.approx(float(E.fit(x, y, d)[2]), rel=1e-8, abs=1e-9)
            p = E.press(x, y, d)
            assert fam.press[0, d] == pytest.approx(float(p), rel=1e-6)


def test_legendre_basis_equals_numpy_and_gives_the_same_span_as_monomials():
    x = np.linspace(-1, 1, 37)
    assert np.allclose(S.legendre_design(x, 8), np.polynomial.legendre.legvander(x, 8), atol=1e-13)
    y = np.sin(3 * x) + x ** 2
    for d in range(9):
        for design in (S.legendre_design, S.monomial_design):
            v = design(x, d)
            e = y - v @ np.linalg.lstsq(v, y, rcond=None)[0]
            f = S.fit_family(x[None, :], y[None, :], d)
            assert f.rss[0, d] == pytest.approx(e @ e, rel=1e-8, abs=1e-10)


def test_pipeline_choices_equal_a_naive_loop_per_dataset_and_degree():
    x, y = G.poly_data(30, 40, 2.0, 17)
    fam = S.fit_family(x, y, 8)
    cr = S.criteria_from_rss(fam.rss, 40, np.arange(9) + 1)
    idx_aic, idx_bic = S.choose(cr["AIC"])[0], S.choose(cr["BIC"])[0]
    for r in range(30):
        rss = []
        for d in range(9):
            v = np.polynomial.legendre.legvander(x[r], d)
            e = y[r] - v @ np.linalg.lstsq(v, y[r], rcond=None)[0]
            rss.append(e @ e)
        aic = [40 * np.log(s / 40) + 2 * (d + 2) for d, s in enumerate(rss)]
        bic = [40 * np.log(s / 40) + (d + 2) * np.log(40) for d, s in enumerate(rss)]
        assert idx_aic[r] == int(np.argmin(aic)) and idx_bic[r] == int(np.argmin(bic))


# --- Kreuzvalidierung gegen scikit-learn ---------------------------------------------------------------------------------------------------

def test_leave_one_out_and_five_fold_equal_scikit_learn():
    sk = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut, PredefinedSplit, cross_val_predict

    x, y = G.poly_data(6, 45, 2.0, 23)
    fam = S.fit_family(x, y, 5)
    fid = R.ranks_array(R.derive_seed(23, 7), 6, 45) % 5
    cv = S.cv_family(fam.design, y, fid, 5)
    assert sk is not None
    for r in range(6):
        for d in (0, 2, 5):
            v = fam.design[r][:, : d + 1]
            lr = LinearRegression(fit_intercept=False)
            loo = cross_val_predict(lr, v, y[r], cv=LeaveOneOut())
            assert np.sum((y[r] - loo) ** 2) == pytest.approx(fam.press[r, d], rel=1e-8)
            kf = cross_val_predict(lr, v, y[r], cv=PredefinedSplit(fid[r]))
            assert np.mean((y[r] - kf) ** 2) == pytest.approx(cv[r, d], rel=1e-8)


# --- Aufnahme eines Rauschmerkmals gegen scipy.stats.f ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [30, 100])
def test_noise_inclusion_simulation_matches_the_f_distribution(n):
    reps = 12000
    x = 2 * R.uniform_matrix(R.derive_seed(5, 1), (reps, n)) - 1
    y = G.poly_truth(x) + 1.5 * R.normal_matrix(R.derive_seed(5, 2), (reps, n))
    noise = R.normal_matrix(R.derive_seed(5, 3), (reps, n))
    v = np.concatenate([np.polynomial.legendre.legvander(x, 3), noise[..., None]], axis=2)
    q = np.linalg.qr(v)[0]
    c = np.einsum("rnk,rn->rk", q, y)
    yy = (y ** 2).sum(1)
    rss_big, rss_small = yy - (c ** 2).sum(1), yy - (c[:, :4] ** 2).sum(1)
    gain = n * np.log(rss_small / rss_big)
    for rule, pen in (("AIC", 2.0), ("BIC", np.log(n))):
        expect = 100.0 * stats.f.sf((n - 5) * (np.exp(pen / n) - 1), 1, n - 5)
        got = 100.0 * np.mean(gain > pen)
        assert abs(got - expect) < _band(expect, reps, extra=0.05), (rule, got, expect)
        assert abs(expect - 100 * T.p_include_noise(n, 5, rule)) < 1e-9


# --- Wahl unter den Graden 3 bis 8: drei Rechenwege ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("n, sigma", [(300, 1.0), (300, 3.0), (30, 1.0)])
def test_pool_choice_by_direct_fits_equals_the_chi_square_route_and_does_not_depend_on_sigma(n, sigma):
    reps = 1500
    x, y = G.poly_data(reps, n, sigma, 31)
    fam = S.fit_family(x, y, 8)
    cr = S.criteria_from_rss(fam.rss[:, 3:], n, np.arange(3, 9) + 1)             # nur die Kandidaten 3 bis 8
    walk = T.pool_walk(n, 200_000, 9)
    for rule in ("AIC", "AICc", "BIC"):
        direct = 100.0 * np.mean(S.choose(cr[rule])[0] == 0)
        assert abs(direct - walk[rule][0]) < _band(walk[rule][0], reps, extra=0.3), (rule, direct, walk[rule][0])


def test_pool_choice_by_orthonormal_coordinates_equals_the_chi_square_route():
    """Dritter Weg: Koordinaten c = Q'y der Vandermonde-Matrix; RSS_d = RSS_8 + Σ_{i>d} c_i². Aus den Daten gezogen, nicht aus χ²."""
    n, reps = 300, 3000
    x, y = G.poly_data(reps, n, 1.0, 41)
    fam = S.fit_family(x, y, 8)
    rss8 = fam.rss[:, 8]
    tail = np.cumsum((fam.c ** 2)[:, ::-1], axis=1)[:, ::-1]                         # tail[:, i] = Σ_{j>=i} c_j²
    rss = np.concatenate([rss8[:, None] + tail[:, 4:], rss8[:, None]], axis=1)[:, :6]  # Grade 3 bis 8
    assert np.allclose(rss, fam.rss[:, 3:], rtol=1e-8)
    walk = T.pool_walk(n, 200_000, 11)
    cr = S.criteria_from_rss(rss, n, np.arange(3, 9) + 1)
    for rule in ("AIC", "BIC"):
        assert abs(100.0 * np.mean(S.choose(cr[rule])[0] == 0) - walk[rule][0]) < _band(walk[rule][0], reps, extra=0.3)


def test_aic_overfit_limit_is_the_random_walk_value_and_bic_vanishes():
    assert abs(T.irrfahrt_limit(600_000, 3) - 26.53) < 0.4                             # Band 4 Standardfehler: 0,23
    one_step = 100.0 * stats.chi2.sf(2.0, 1)
    assert one_step == pytest.approx(15.73, abs=0.005)                                 # eine Stufe: P(χ²₁ > 2)
    big = T.pool_walk(100000, 200_000, 3)
    assert abs((100 - big["AIC"][0]) - 26.53) < 0.5 and 100 - big["BIC"][0] < 0.3 and abs((100 - big["AICc"][0]) - (100 - big["AIC"][0])) < 0.6


# --- Nichtzentralität gegen scipy.stats.ncf -----------------------------------------------------------------------------------------------

def test_lambda_of_the_cubic_term_equals_a_least_squares_residual():
    x, _ = G.poly_data(20, 60, 1.0, 5)
    lam = T.lambda_cubic(x, 3.0, 2.0)
    for r in range(20):
        v = np.stack([x[r] ** j for j in range(4)], axis=1)
        z = v[:, 3] - v[:, :3] @ np.linalg.lstsq(v[:, :3], v[:, 3], rcond=None)[0]
        assert lam[r] == pytest.approx(9.0 * (z @ z) / 4.0, rel=1e-8)


@pytest.mark.parametrize("n, sigma", [(30, 3.0), (300, 3.0)])
def test_pair_choice_simulation_matches_the_noncentral_f(n, sigma):
    reps = 4000
    x, y = G.poly_data(reps, n, sigma, 53)
    v = np.stack([x ** j for j in range(4)], axis=-1)
    q = np.linalg.qr(v)[0]
    c = np.einsum("rnk,rn->rk", q, y)
    yy = (y ** 2).sum(1)
    gain = n * np.log((yy - (c[:, :3] ** 2).sum(1)) / (yy - (c ** 2).sum(1)))
    lam = T.lambda_cubic(x, 3.0, sigma)
    for rule, pen in (("AIC", 2.0), ("BIC", np.log(n))):
        formula = 100.0 * float(np.mean(stats.ncf.sf((n - 4) * (np.exp(pen / n) - 1), 1, n - 4, lam)))
        assert abs(100.0 * np.mean(gain > pen) - formula) < _band(formula, reps, extra=0.2), rule
        assert formula == pytest.approx(100.0 * float(np.mean(T.p_choose_bigger(n, 4, lam, rule))), abs=1e-9)


def test_tipping_points_are_the_inverse_of_the_noncentral_f_probability():
    for rule in ("AIC", "BIC"):
        for p in (0.5, 0.9):
            lam = T.kipp_lambda(300, p, rule)
            assert float(T.p_choose_bigger(300, 4, lam, rule)) == pytest.approx(p, abs=1e-9)
    assert T.kipp_lambda(300, 0.5, "AIC") == pytest.approx(1.96, abs=0.01) and T.kipp_lambda(300, 0.5, "BIC") == pytest.approx(5.67, abs=0.01)
    assert T.kipp_lambda(300, 0.9, "AIC") == pytest.approx(7.23, abs=0.01) and T.kipp_lambda(300, 0.9, "BIC") == pytest.approx(13.46, abs=0.01)


# --- Überdeckung gegeben die Aufnahme gegen die Simulation --------------------------------------------------------------------------------

@pytest.mark.parametrize("rule, theta", [("AIC", 0.0), ("AIC", 1.0), ("AIC", 3.0), ("BIC", 0.0), ("BIC", 1.0), ("BIC", 2.0)])
def test_coverage_given_inclusion_equals_the_simulation(rule, theta):
    n, reps = 100, 40000
    we = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    beta = theta / np.sqrt(n)
    y = beta * we[None, :] + R.normal_matrix(R.derive_seed(61, 1), (reps, n))
    ybar, bhat = y.mean(1), (y * we[None, :]).mean(1)
    rss0 = ((y - ybar[:, None]) ** 2).sum(1)
    rss1 = rss0 - n * bhat ** 2
    pen = 2.0 if rule == "AIC" else np.log(n)
    incl = n * np.log(rss0 / rss1) > pen
    se = np.sqrt(rss1 / (n - 2) / n)
    cover = np.abs(bhat - beta) < stats.t.ppf(0.975, n - 2) * se
    exact_cover, exact_in = T.coverage_given_inclusion(theta, n, rule)
    assert abs(incl.mean() - exact_in) < 4 * np.sqrt(exact_in * (1 - exact_in) / reps) + 1e-3
    m = int(incl.sum())
    assert abs(cover[incl].mean() - exact_cover) < 4 * np.sqrt(max(exact_cover * (1 - exact_cover), 1e-4) / m) + 2e-3


# --- Export-Kurve, Import-Prognosefehler ---------------------------------------------------------------------------------------------------

def test_export_bias_equals_a_monomial_least_squares_fit():
    v = np.linspace(1, 10, 361)
    x = G.export_x(v)
    f = G.export_truth(v)
    for d, expect in enumerate(ev.export_bias2()):
        m = np.stack([x ** j for j in range(d + 1)], axis=1)
        e = f - m @ np.linalg.lstsq(m, f, rcond=None)[0]
        assert expect == pytest.approx(np.mean(e ** 2), rel=1e-5, abs=1e-12)
    assert np.all(np.diff(ev.export_bias2()) < 0)


def test_true_subset_prediction_error_equals_the_theory_value():
    """Erwartung des Prognosefehlers der wahren Menge: σ² · Spur(M · E[(X'X)^-1]) ≈ σ² · 3 / n bei n = 300 (Näherung, 10 %)."""
    f = ev.feature_selection(300, 5, 600, 4)
    assert f.pe_mean["wahre Menge"] == pytest.approx(G.SIGMA2_IMPORT * 3 / 300, rel=0.1)
    assert f.pe_mean["alle Merkmale"] == pytest.approx(G.SIGMA2_IMPORT * 8 / 300, rel=0.1)
