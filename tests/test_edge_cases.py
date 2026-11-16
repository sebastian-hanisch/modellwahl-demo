"""Randfälle, an denen Kriterien erfahrungsgemäß brechen: kleines n, RSS = 0, Hebel 1, Gleichstände, zu wenige Faltungen, σ = 0, Regler am Rand (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import mwl_evaluation as ev
import mwl_exact as E
import mwl_selection as S
import mwl_theory as T


def test_n_not_larger_than_k_plus_one_makes_aicc_undefined_not_an_error():
    x = np.array([[0.0, 1.0, 2.0, 3.0]])
    y = np.array([[1.0, 0.5, 2.0, 2.5]])
    fam = S.fit_family(x, y, 1)                                       # n = 4: Grad 0 hat K = 2 (n - K - 1 = 1), Grad 1 hat K = 3 (n - K - 1 = 0)
    cr = S.criteria_from_rss(fam.rss, 4, [1, 2])
    assert np.isinf(cr["AICc"][0, 1]) and np.isfinite(cr["AICc"][0, 0]) and int(S.choose(cr["AICc"])[0][0]) == 0


def test_interpolation_gives_rss_zero_aic_minus_infinity_and_a_reported_choice():
    x, y = np.array([E.HAND_X], float), np.array([E.HAND_Y], float)
    fam = S.fit_family(x, y, 7)
    assert fam.rss[0, 7] == 0.0 and np.isinf(fam.press[0, 7])
    cr = S.criteria_from_rss(fam.rss, 8, np.arange(8) + 1)
    assert int(S.choose(cr["AIC"])[0][0]) == 7 and int(S.choose(cr["BIC"])[0][0]) == 7 and int(S.choose(cr["AICc"])[0][0]) != 7
    assert E.pick(E.table(E.HAND_X, E.HAND_Y, 7), "AIC") == 7          # auch die exakte Rechnung: -∞ gewinnt, die Wahl bleibt definiert


def test_constant_y_has_rss_zero_for_every_degree_and_the_smallest_degree_wins():
    x = np.linspace(-1, 1, 20)[None, :]
    fam = S.fit_family(x, np.full((1, 20), 4.0), 4)
    assert np.all(fam.rss == 0.0)
    cr = S.criteria_from_rss(fam.rss, 20, np.arange(5) + 1)
    idx, tie = S.choose(cr["AIC"])
    assert int(idx[0]) == 0 and bool(tie[0])


def test_sigma_zero_picks_the_true_degree_by_the_tie_rule():
    """Ohne Rauschen ist RSS ab Grad 3 genau null (Wahrheit enthalten): der Gleichstand bei -∞ geht an den kleinsten Grad, also den wahren."""
    f = ev.selection_frequency(40, 0.0, 20, 3)
    for k in ("AIC", "BIC"):
        assert f.share[k][1] == 100.0 and f.ties[k] == 20


def test_leverage_one_makes_leave_one_out_undefined():
    """Anscombe-IV-Muster: zehn Beobachtungen bei x = 8, eine bei x = 19. Die Gerade geht durch den Einzelpunkt, sein Hebel ist 1."""
    x = np.array([[8.0] * 10 + [19.0]])
    y = np.array([[6.6, 5.8, 7.7, 8.8, 8.5, 7.0, 5.3, 12.5, 5.6, 7.9, 6.9]])
    fam = S.fit_family(x, y, 1)
    assert np.isfinite(fam.press[0, 0]) and np.isinf(fam.press[0, 1])
    assert E.press([8] * 10 + [19], [66, 58, 77, 88, 85, 70, 53, 125, 56, 79, 69], 1) is None


def test_fewer_observations_than_folds_gives_undefined_cross_validation():
    v = S.monomial_design(np.array([[0.0, 1.0, 2.0, 3.0]]), 1)
    assert np.all(np.isinf(S.cv_family(v, np.array([[1.0, 2.0, 2.0, 4.0]]), np.array([[0, 1, 2, 3]]), 5)))


def test_degree_above_n_minus_one_in_the_drawn_example_is_clipped():
    x, y, rows, picks = ev.drawn_rows(12, 1.0, 7, True, 3)
    assert len(rows) == 8 and all(0 <= p <= 7 for p in picks.values())
    x, y, rows, picks = ev.drawn_rows(12, 1.0, 20, True, 3)
    assert len(rows) == 12                                          # höchstens n - 1 = 11


def test_handbeispiel_with_interpolating_candidate_in_the_app_table():
    rows, picks = ev.exact_rows(7, True)
    assert len(rows) == 8 and rows[7]["LOO"] == float("inf") and rows[7]["AIC"] == float("-inf") and picks["AIC"] == 7 and picks["LOO"] == 1


def test_ties_are_counted_not_hidden():
    crit = np.array([[2.0, 1.0, 1.0]])
    idx, tie = S.choose(crit)
    assert int(idx[0]) == 1 and bool(tie[0])


def test_theory_at_the_slider_edges_returns_probabilities():
    for n in (30, 1000):
        for rule in ("AIC", "BIC"):
            for lam in (0.0, 60.0):
                p = float(T.p_choose_bigger(n, 4, lam, rule))
                assert 0.0 <= p <= 1.0
    sig = ev.signal_curve(30, 3.0, 0.0)                              # β₃ = 0: λ = 0, der Kubikterm ist Rauschen
    assert sig.lam == 0.0 and sig.p_now["AIC"] == pytest.approx(100 * T.p_include_noise(30, 4, "AIC"), abs=1e-9)
    assert np.all(sig.curve["AIC"] >= 0) and np.all(sig.curve["AIC"] <= 100)


def test_features_without_noise_columns_and_at_the_smallest_n():
    f = ev.feature_selection(50, 0, 60, 3)
    assert f.expected_noise == {"AIC": 0.0, "BIC": 0.0} and all(v == 0.0 for v in f.noise_mean.values())
    assert 0.0 <= min(f.exact.values()) and max(f.exact.values()) <= 100.0


def test_import_features_for_five_noise_columns_have_128_subsets_and_the_loo_cap():
    assert len(S.all_subsets(5)) == 128 and len(S.all_subsets(0)) == 4
    assert ev.feature_reps(1000, 1000) == 300 and ev.feature_reps(50, 1000) == 1000 and ev.feature_reps(300, 100) == 100
    f = ev.feature_selection(300, 5, 60, 3)
    assert f.reps_loo == 60 and ev.feature_selection(1000, 5, 60, 3).reps_loo == 60


def test_export_budget_keeps_at_least_fifty_repetitions():
    assert ev.export_reps(3000, 500) == 200 and ev.export_reps(30, 500) == 500 and ev.export_reps(3000, 100) == 100
    assert ev.export_reps(10 ** 6, 500) == 50


def test_after_selection_at_the_smallest_sample_does_not_fail():
    a = ev.after_selection(50, 100, 3)
    assert all(0.0 <= a.chosen_we[p] <= 100.0 for p in ev.PROCEDURES)


def test_coverage_for_theta_zero_and_bic_is_exactly_zero():
    """Die BIC-Schwelle |t| > 2,15 liegt über dem Kritischwert 1,98: das Intervall der aufgenommenen schließt die Null nie ein."""
    assert T.coverage_given_inclusion(0.0, 100, "BIC")[0] == pytest.approx(0.0, abs=1e-12)


def test_an_empty_feature_column_in_a_small_half_does_not_crash_the_fit():
    """Bei n = 50 hat in der Datenteilung manchmal eine Hälfte (25 Zeilen) keinen einzigen Zollhalt: die Gram-Matrix ist singulär (Absturz beim ersten Bau, Seed 35, 500 Wiederholungen)."""
    rg = np.random.default_rng(1)
    x = np.concatenate([np.ones((3, 25, 1)), np.zeros((3, 25, 1)), rg.normal(size=(3, 25, 1))], axis=2)
    y = rg.normal(size=(3, 25))
    b, se, df = S.fit_subset(x, y, np.array([0, 1, 2]))
    assert df == 22 and np.all(np.isinf(se[:, 1])) and np.all(np.isfinite(se[:, [0, 2]])) and np.all(b[:, 1] == 0.0)
    rss, press, coef = S.subset_tables(x, y, S.all_subsets(0), with_loo=True)
    assert np.all(np.isfinite(rss)) and rss.shape == (3, 4)


def test_after_selection_at_n_50_with_the_seed_that_once_crashed():
    a = ev.after_selection(50, 500, 35)
    assert all(0.0 <= a.chosen_we[p] <= 100.0 for p in ev.PROCEDURES)
