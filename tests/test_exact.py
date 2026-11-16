"""Handbeispiel (n = 8) von Hand nachgerechnet: jeder Erwartungswert steht hier als Rechnung, nicht als Abschrift des Codes (DEMO-PLAYBOOK §5).

x = -7, -5, ..., 7 (Summe 0, Summe x² = 168), y = 3, 0, 8, 3, 5, 6, 9, 6 (Summe 40, Summe y² = 260), Kandidaten Grad 0, 1, 2. Alles in Brüchen, Logarithmen mit `decimal`."""

import decimal
from fractions import Fraction as Fr

import mwl_exact as E


def test_sums_of_the_example(hand):
    x, y = hand
    assert sum(x) == 0 and sum(v * v for v in x) == 168 and sum(y) == 40 and sum(v * v for v in y) == 260 and sum(a * b for a, b in zip(x, y)) == 62


def test_residual_sums_of_squares_by_hand(hand):
    x, y = hand
    rss0 = Fr(260) - Fr(8) * 5 ** 2                                   # Σy² - n ȳ² = 60
    rss1 = rss0 - Fr(62 ** 2, 168)                                    # minus (Σxy)² / Σx² = 1559/42
    q2 = [xi * xi - 21 for xi in x]                                   # Parabel senkrecht auf 1 und x (Werte 28, 4, -12, -20, ...), Norm² 2688
    assert sum(q2) == 0 and sum(a * b for a, b in zip(q2, x)) == 0 and sum(v * v for v in q2) == 2688
    rss2 = rss1 - Fr(sum(a * b for a, b in zip(q2, y)) ** 2, 2688)    # Σ q y = -40: 40² / 2688 = 25/42
    assert (rss0, rss1, rss2) == (Fr(60), Fr(1559, 42), Fr(767, 21))
    assert [E.fit(x, y, d)[2] for d in (0, 1, 2)] == [rss0, rss1, rss2]
    assert E.fit(x, y, 2)[0][2] == Fr(-5, 336)                        # Koeffizient des Quadrats = -40/2688


def test_hat_values_and_press_by_hand(hand):
    x, y = hand
    assert E.hat_diagonal(x, 0) == [Fr(1, 8)] * 8
    h1 = E.hat_diagonal(x, 1)                                         # 1/8 + x²/168
    assert h1 == [Fr(1, 8) + Fr(xi * xi, 168) for xi in x] and h1[0] == Fr(5, 12) and h1[1] == Fr(23, 84) and h1[2] == Fr(5, 28) and h1[3] == Fr(11, 84)
    assert sum(h1) == 2 and sum(E.hat_diagonal(x, 2)) == 3            # Spur der Hat-Matrix = Zahl der Koeffizienten
    assert E.press(x, y, 0) == Fr(60) * Fr(8, 7) ** 2 == Fr(3840, 49)  # alle Hebel 1/8: e / (7/8)
    assert E.press(x, y, 1) == Fr(33288089817208, 513992926489)
    assert E.press(x, y, 2) == Fr(119406765248, 1326489241)
    assert E.press(x, y, 1) < E.press(x, y, 0) < E.press(x, y, 2)


def test_leave_one_out_equals_refit_in_fractions(hand):
    x, y = hand
    for d in (0, 1, 2):
        assert E.press(x, y, d) == E.press_by_refit(x, y, d)


def test_criteria_values_and_choices(hand):
    x, y = hand
    rows = E.table(x, y, 2)
    vals = {k: [round(float(r[k]), 2) for r in rows] for k in ("AIC", "AICc", "BIC")}
    assert vals["AIC"] == [20.12, 18.28, 20.15] and vals["BIC"] == [20.28, 18.52, 20.47] and vals["AICc"] == [22.52, 24.28, 33.48]
    assert E.pick(rows, "AIC") == 1 and E.pick(rows, "BIC") == 1 and E.pick(rows, "press") == 1 and E.pick(rows, "AICc") == 0
    for k in ("AIC", "AICc", "BIC"):                                    # Abstand zum Zweitplatzierten über 1,5: keine knappe Entscheidung
        s = sorted(float(r[k]) for r in rows)
        assert s[1] - s[0] > 1.5


def test_aicc_penalty_by_hand(hand):
    x, y = hand
    rows = E.table(x, y, 2)
    # K = Grad + 2: Zuschlag 2K(K+1)/(n-K-1) = 2·2·3/5 = 12/5 bei Grad 0, 2·3·4/4 = 6 bei Grad 1
    assert abs(float(rows[0]["AICc"] - rows[0]["AIC"]) - 12 / 5) < 1e-12 and abs(float(rows[1]["AICc"] - rows[1]["AIC"]) - 6) < 1e-12


def test_variance_count_shifts_aic_but_not_the_aicc_penalty(hand):
    x, y = hand
    with_var, without = E.table(x, y, 2, True), E.table(x, y, 2, False)
    for a, b in zip(with_var, without):
        assert abs(float(a["AIC"] - b["AIC"]) - 2) < 1e-12 and abs(float(a["BIC"] - b["BIC"]) - float(decimal.Decimal(8).ln())) < 1e-12
    assert E.pick(with_var, "AIC") == E.pick(without, "AIC") == 1
    assert E.pick(with_var, "AICc") == 0 and E.pick(without, "AICc") == 1                     # die Konvention entscheidet


def test_bic_penalty_exceeds_aic_penalty_exactly_from_n_8():
    assert decimal.Decimal(8).ln() > 2 > decimal.Decimal(7).ln()


def test_f_statistic_and_thresholds_of_the_example(hand):
    x, y = hand
    rss1, rss2 = E.fit(x, y, 1)[2], E.fit(x, y, 2)[2]
    f = (rss1 - rss2) / (rss2 / (8 - 2 - 1))                         # Freiheitsgrade des größeren Modells: n - 3 = 5
    assert f == Fr(125, 1534)
    import mwl_theory as T
    assert abs(T.f_threshold(8, 3, "AIC") - 1.4201) < 5e-4 and abs(T.f_threshold(8, 3, "BIC") - 1.4841) < 5e-4
    assert float(f) < T.f_threshold(8, 3, "AIC")                      # die Parabel wird nicht aufgenommen


def test_variance_of_cubic_rest_is_4_over_175():
    moment = {k: Fr(1, k + 1) if k % 2 == 0 else Fr(0) for k in range(8)}     # E[x^k] bei Gleich(-1, 1)
    assert moment[6] - Fr(6, 5) * moment[4] + Fr(9, 25) * moment[2] == Fr(4, 175)   # Var(x³ - 3x/5)


def test_interpolation_has_rss_zero_and_undefined_press(hand):
    x, y = hand
    rows = E.table(x, y, 7)
    assert rows[7]["rss"] == 0 and rows[7]["press"] is None and float(rows[7]["AIC"]) == float("-inf")
    assert float(rows[7]["AICc"]) == float("inf")
