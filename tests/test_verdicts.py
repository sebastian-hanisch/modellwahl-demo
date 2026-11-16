"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): Schwellen der Art und jede genannte Zahl stehen hier."""

import numpy as np

import mwl_evaluation as ev
from mwl_evaluation import de, de_int, de_sig, pct

NAMES = ("AIC", "AICc", "BIC", "CV5", "LOO")


def _freq(rows: dict, n=300, sigma=1.0, reps=1000):
    """rows: Kriterium -> (zu klein, wahr, zu groß) in %."""
    return ev.Frequency(n, sigma, reps, rows, {k: np.zeros(9) for k in NAMES}, {}, {}, {k: 0 for k in NAMES}, 1.0)


def _rows(true_shares):
    return {k: (0.0, v, 100.0 - v) for k, v in zip(NAMES, true_shares)}


def test_number_formats():
    assert de(3.14159, 2) == "3,14" and pct(26.53, 1) == "26,5 %" and pct(0.0704, 2) == "0,07 %"
    assert de_sig(0.001234) == "0,00123" and de_sig(4.94e-09) == "4,94·10⁻⁹" and de_sig(0.2) == "0,2"
    assert de_int(1000) == "1000" and de_int(10000) == "10.000" and de_int(100000) == "100.000"


def test_standard_error_of_a_share_by_hand():
    assert ev.binomial_se(50.0, 100) == 5.0 and ev.binomial_se(0.0, 100) == 0.0 and abs(ev.binomial_se(72.3, 1000) - 1.4151) < 1e-3


def test_frequency_verdict_is_success_only_if_the_best_criterion_reaches_90_percent():
    kind, text = ev.verdict_frequency(_freq(_rows((72.3, 74.7, 98.2, 69.0, 72.8))))
    assert kind == "success" and text.endswith("Nur BIC findet den wahren Grad hier fast immer.")
    assert "σ = 1,0, n = 300, 1000 Wiederholungen: wahrer Grad 3 gewählt von AIC 72,3 %, AICc 74,7 %, BIC 98,2 %, CV5 69,0 %, LOO 72,8 %." in text
    assert "zu große AIC in 27,7 % und BIC in 1,8 %" in text
    assert "Standardfehler der Anteile bis 1,5 Punkte" in text          # größter Wert bei 69,0 %: 100·√(0,69·0,31/1000) = 1,46
    kind, text = ev.verdict_frequency(_freq(_rows((89.9, 80.0, 70.0, 60.0, 60.0))))
    assert kind == "info" and "Am häufigsten findet ihn AIC, aber nicht sicher." in text
    kind, text = ev.verdict_frequency(_freq(_rows((49.9, 40.0, 30.0, 20.0, 20.0))))
    assert kind == "warning" and "in mehr als der Hälfte" in text
    assert ev.verdict_frequency(_freq(_rows((90.0, 10.0, 10.0, 10.0, 10.0))))[0] == "success"        # Grenze 90 inklusive
    assert ev.verdict_frequency(_freq(_rows((50.0, 10.0, 10.0, 10.0, 10.0))))[0] == "info"            # Grenze 50: darunter Warnung


def _signal(p_aic, p_bic):
    return ev.Signal(300, 6.857, np.zeros(2), {}, {"AIC": p_aic, "BIC": p_bic}, {"AIC": 1.96, "BIC": 5.67}, {"AIC": 7.23, "BIC": 13.46}, {"AIC": 1.98, "BIC": 5.68})


def test_signal_verdict_names_tipping_points_and_the_position_of_the_sliders():
    kind, text = ev.verdict_signal(_signal(88.1, 58.4))
    assert kind == "warning" and "ab λ = 1,96 (AIC) und 5,67 (BIC)" in text and "ab 7,23 und 13,46" in text and "λ ≈ 6,9" in text and "88,1 %" in text and "58,4 %" in text
    assert ev.verdict_signal(_signal(80.0, 70.0))[0] == "warning"                           # genau 10 Prozentpunkte Abstand und AIC genau 80 %: inklusive
    assert ev.verdict_signal(_signal(80.0, 70.1))[0] == "info"                              # 9,9 Punkte Abstand: keine Warnung
    assert ev.verdict_signal(_signal(99.0, 95.0))[0] == "success"
    assert ev.verdict_signal(_signal(60.0, 30.0))[0] == "info"                              # AIC unter 80 %: keine Warnung
    assert ev.verdict_signal(_signal(79.9, 40.0))[0] == "info"


def test_overfit_verdict_quotes_the_frozen_table():
    frozen = {"pool": {"n": {"30": {"AIC": [60.56], "BIC": [84.83]}, "300": {"AIC": [72.46], "BIC": [98.0]}, "100000": {"AIC": [73.46], "BIC": [99.93]}}},
              "grenzwert": {"wert": 26.5078}}
    kind, text = ev.verdict_overfit(frozen)
    assert kind == "info" and "39,4 % der Wiederholungen bei n = 30" in text and "27,5 % bei n = 300" in text and "26,54 % bei n = 100.000" in text
    assert "etwa 26,5 %" in text and "BIC fällt auf 0,07 %" in text and "nicht für jede" in text


def test_export_verdict_quotes_ratios_and_hits():
    bias = np.array([0.24, 0.012, 0.001, 1.09e-4, 1.3e-5, 1.7e-6, 2.3e-7, 3.3e-8, 4.94e-9])
    e = ev.ExportResult(n=300, reps=1000, reps_asked=1000, bias2=bias, deg_median={"AIC": (3.0, 3.0, 4.0), "BIC": (2.0, 2.0, 3.0)}, dist={},
                        hit_best={"AIC": 20.4, "BIC": 17.5}, pe_median={"AIC": 0.0014, "BIC": 0.0015}, ratio={"AIC": 1.214, "BIC": 1.559},
                        best_median=(3.0, 3.0, 4.0), pe_best_median=0.00094, fold_differ=40.8)
    kind, text = ev.verdict_export(e)
    assert kind == "info" and "bei Grad 3: 0,000109, bei Grad 8: 4,94·10⁻⁹" in text and "im Median Grad 3 (Quartile 3 bis 4), BIC Grad 2" in text
    assert "1,21- bis 1,56-Fachen" in text and "nur in 18 % bis 20 %" in text and "in 40,8 % der Datensätze" in text


def _features(pe_aic, pe_bic, k_noise=5):
    return ev.FeatureResult(n=300, k_noise=k_noise, reps=300, reps_loo=200, exact={"AIC": 41.3, "AICc": 44.1, "BIC": 80.5, "LOO": 46.5}, zoll_in={}, we_in={},
                            noise_mean={"AIC": 0.81, "BIC": 0.08}, noise_any={}, pe_mean={"AIC": pe_aic, "BIC": pe_bic, "wahre Menge": 0.0044}, pe_median={},
                            expected_noise={"AIC": 0.80, "BIC": 0.09}, expected_none={})


def test_features_verdict_kinds_and_numbers():
    kind, text = ev.verdict_features(_features(0.0089, 0.0072))
    assert kind == "success" and "AIC in 41,3 %, AICc in 44,1 % und BIC in 80,5 %" in text and "AIC 0,81 und BIC 0,08" in text and "Einzelstufe: 0,80 und 0,09" in text
    assert "AIC 0,0089, BIC 0,0072" in text and "wahre Menge 0,0044" in text and text.endswith("Am Import ist BIC hier der bessere Vorhersager.")
    assert ev.verdict_features(_features(0.0070, 0.0072))[0] == "warning"
    assert ev.verdict_features(_features(0.0072, 0.0072))[0] == "warning"                  # Gleichheit ist nicht „besser“
    assert ev.verdict_features(_features(0.0070, 0.0072, k_noise=0))[0] == "info"


def test_after_verdict_quotes_exact_values_and_the_price_of_splitting():
    p = ev.PROCEDURES
    a = ev.AfterResult(n=100, reps=500, chosen_we={p[0]: 100.0, "AIC": 73.5, "BIC": 47.0, "AIC, Datenteilung": 52.6, "BIC, Datenteilung": 30.5}, cover_we={}, noise_mean={},
                       noise_sig={p[0]: 5.1, "AIC": 32.0, "BIC": 100.0, "AIC, Datenteilung": 4.9, "BIC, Datenteilung": 6.4}, any_sig={})
    thetas = np.linspace(0.0, 5.0, 11)
    cov = {"AIC": (np.linspace(60, 100, 11), None), "BIC": (np.linspace(0, 90, 11), None)}
    kind, text = ev.verdict_after(a, 1.0, cov, thetas)
    assert kind == "warning" and "θ = 1,0" in text and "bei AIC in 68,0 %, bei BIC in 18,0 %" in text
    assert "32,0 % (AIC) und 100,0 % (BIC)" in text and "gegen 5,1 % ohne Wahl" in text and "mit Datenteilung 4,9 % und 6,4 %" in text
    assert "in 53 % statt 74 % (AIC) und 30 % statt 47 % (BIC) gewählt" in text


def test_hand_verdict_groups_equal_choices():
    kind, text = ev.verdict_hand({"AIC": 1, "BIC": 1, "LOO": 1, "AICc": 0}, 2, True)
    assert kind == "warning" and "AIC, BIC, LOO wählen Grad 1; AICc wählt Grad 0" in text and "bis Grad 2 und mit Zählung der Varianz" in text
    kind, text = ev.verdict_hand({"AIC": 1, "BIC": 1, "LOO": 1, "AICc": 1}, 2, False)
    assert kind == "info" and "ohne Zählung der Varianz" in text and "Alle Kriterien wählen dasselbe Modell." in text
