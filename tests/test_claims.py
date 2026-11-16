"""Claims-Tests: jede Zahl der Befunde in der README, mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: Zufalls-Seed 35 (Standard der App), Wiederholungszahlen wie in der README genannt. Jede README-Zahl steht hier als Literal (und wird in der README gesucht) und
wird neu gerechnet. Plattform: Brüche, `decimal` und Integrale exakt; Anteile aus Neu-Rechnung als Band (drei Zählungen plus Rundung), Mittel und Prognosefehler relativ (1 %),
Floats aus Formeln relativ 1e-9. Die vorgerechneten Zahlen (`mwl_frozen.json`) werden gegen die README gelesen; ihre Neu-Rechnung prüft `tests/test_frozen.py`."""

import json
import math
from pathlib import Path

import pytest
from scipy import stats

import mwl_constants as C
import mwl_evaluation as ev
import mwl_exact as E
import mwl_theory as T
from mwl_evaluation import de

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8").replace("\\|", "|").replace("\\*", "*")
FROZEN = json.loads((ROOT / "mwl_frozen.json").read_text(encoding="utf-8"))
SEED = 35
CRIT = ("AIC", "AICc", "BIC", "CV5", "LOO")


def has(*texts):
    for t in texts:
        assert t in README, t


def sig3(x):
    """Drei gültige Stellen mit Dezimalkomma, Nullen am Ende bleiben (0,0210)."""
    return f"{x:#.3g}".replace(".", ",")


def sci3(x):
    mant, exp = f"{x:.2e}".split("e")
    sup = str(int(exp)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant}·10{sup}".replace(".", ",")


def pc(x, nd=1):
    return de(x, nd) + " %"


def near(value, literal, tol):
    """Neu gerechneter Wert gegen das README-Literal im Band (Rundung 0,05 steckt in tol)."""
    assert abs(value - literal) <= tol, (value, literal)


# --- Handbeispiel: exakt --------------------------------------------------------------------------------------------------------------------

def test_b1_handbeispiel_numbers():
    rows = E.table(E.HAND_X, E.HAND_Y, 2)
    has("60, 1559/42, 767/21", "3840/49 = 78,37", "64,76", "90,02")
    assert [str(r["rss"]) for r in rows] == ["60", "1559/42", "767/21"] and str(rows[0]["press"]) == "3840/49"
    assert [round(float(r["press"]), 2) for r in rows] == [78.37, 64.76, 90.02]
    for key, expected in (("AIC", [20.12, 18.28, 20.15]), ("BIC", [20.28, 18.52, 20.47]), ("AICc", [22.52, 24.28, 33.48])):
        assert [round(float(r[key]), 2) for r in rows] == expected
        has(" / ".join(de(v, 2) for v in expected))
    picks = {k: E.pick(rows, k) for k in ("AIC", "BIC", "AICc", "press")}
    assert picks == {"AIC": 1, "BIC": 1, "AICc": 0, "press": 1}
    has("AIC, BIC und LOO wählen Grad 1, AICc wählt Grad 0")
    for k in ("AIC", "BIC", "AICc"):
        s = sorted(float(r[k]) for r in rows)
        assert s[1] - s[0] > 1.5
    has("Abstand zum Zweitplatzierten jeweils über 1,5")
    assert E.pick(E.table(E.HAND_X, E.HAND_Y, 2, False), "AICc") == 1
    has("wählt AICc mit `K = k + 1` den Grad 0, mit `K = k` den Grad 1")


def test_b1_f_statistic_thresholds_and_inclusion():
    rss1, rss2 = E.fit(E.HAND_X, E.HAND_Y, 1)[2], E.fit(E.HAND_X, E.HAND_Y, 2)[2]
    f = (rss1 - rss2) / (rss2 / 5)
    assert str(f) == "125/1534" and de(float(f), 4) == "0,0815"
    assert de(T.f_threshold(8, 3, "AIC"), 3) == "1,420" and de(T.f_threshold(8, 3, "BIC"), 3) == "1,484"
    assert pc(100 * T.p_include_noise(8, 3, "AIC")) == "28,7 %" and pc(100 * T.p_include_noise(8, 3, "BIC")) == "27,7 %"
    has("`F = 125/1534 = 0,0815`", "1,420 (AIC) und 1,484 (BIC)", "28,7 % (AIC) beziehungsweise 27,7 % (BIC)", "(3 Koeffizienten im größeren Modell)")
    assert abs(math.log(8) - 2.0794) < 1e-4 and math.log(7) < 2
    has("`ln 8 = 2,079 > 2 > ln 7`")


def test_b1_candidates_up_to_degree_six_overfit_at_n_8():
    rows = E.table(E.HAND_X, E.HAND_Y, 6)
    assert E.pick(rows, "AIC") == 6 and E.pick(rows, "BIC") == 6 and E.pick(rows, "press") == 1
    has("wählen AIC und BIC wegen der fast verschwindenden RSS den Grad 6", "LOO bleibt bei Grad 1")


def test_b1_single_stage_probabilities_for_the_import_basis():
    aic = [100 * T.p_include_noise(n, 4, "AIC") for n in (50, 100, 300, 1000)]
    bic = [100 * T.p_include_noise(n, 4, "BIC") for n in (50, 100, 300, 1000)]
    assert [5 * a / 100 for a in aic] == pytest.approx([0.89, 0.83, 0.80, 0.79], abs=0.005)
    assert [5 * b / 100 for b in bic] == pytest.approx([0.30, 0.18, 0.09, 0.04], abs=0.005)
    has("bei AIC 0,89 / 0,83 / 0,80 / 0,79, bei BIC 0,30 / 0,18 / 0,09 / 0,04")
    assert pc(100 * stats.chi2.sf(2.0, 1), 2) == "15,73 %"                                                   # P(chi2_1 > 2) = 1 - erf(1)
    has("`P(χ²₁ > 2)` = 15,73 %")


# --- Experiment 2: Zellen (vorgerechnet, gegen die README gelesen) ----------------------------------------------------------------------

CELLS = {"s1_n30": "1, 30", "s1_n300": "1, 300", "s3_n30": "3, 30", "s3_n300": "3, 300"}


def test_b2_cell_table_in_the_readme_equals_the_frozen_values():
    for key, label in CELLS.items():
        sh = FROZEN["zellen"][key]["shares"]
        row = f"| {label} | " + " | ".join((f"**{pc(sh[k][1])}**" if (key, k) == ("s1_n300", "BIC") else pc(sh[k][1])) for k in CRIT) + " |"
        has(row)
        assert FROZEN["zellen"][key]["reps"] == 1000 and FROZEN["zellen"][key]["seed"] == SEED
    has("Polynom, Kandidaten 0 bis 8, 1000 Wiederholungen je Zelle")


def test_b2_statements_about_the_cell_table():
    sh = {k: v["shares"] for k, v in FROZEN["zellen"].items()}
    eq = [sh[c][k][1] for c in sh for k in CRIT]
    assert (min(eq), max(eq)) == (9.4, 98.2)
    has("in 9,4 bis 98,2 % der Wiederholungen gewählt")
    assert max(sh["s3_n30"][k][1] for k in CRIT) == 16.4
    has("bei σ = 3 und n = 30 von keinem Kriterium in mehr als 16,4 %")
    assert sh["s3_n300"]["BIC"][0] == 41.8 and sh["s3_n300"]["AIC"][0] == 10.3
    has("wählt BIC in 41,8 % einen zu kleinen Grad (AIC: 10,3 %)")
    assert sh["s1_n300"]["AIC"][2] == 27.7 and sh["s1_n300"]["BIC"][2] == 1.8
    has("wählt AIC in 27,7 % einen zu großen Grad (BIC: 1,8 %)")
    assert math.sqrt(0.5 * 0.5 / 1000) * 100 < 1.6
    has("Standardfehler bis 1,6 Punkte (1000 Wiederholungen)")


def test_b2_prediction_error_numbers():
    z = FROZEN["zellen"]
    a, b = z["s1_n300"], z["s3_n300"]
    has(f"Mittel {sig3(a['pe_mean']['BIC'])} gegen {sig3(a['pe_mean']['AIC'])}, Median {sig3(a['pe_median']['BIC'])} gegen {sig3(a['pe_median']['AIC'])}",
        f"Grad 3 fest: {sig3(a['pe_mean']['Grad 3 fest'])} und {sig3(a['pe_median']['Grad 3 fest'])}")
    assert a["pe_mean"]["BIC"] < a["pe_mean"]["AIC"] and a["pe_median"]["BIC"] < a["pe_median"]["AIC"]
    has(f"Mittel {sig3(b['pe_mean']['BIC'])} gegen {sig3(b['pe_mean']['AIC'])}, Median {sig3(b['pe_median']['BIC'])} gegen {sig3(b['pe_median']['AIC'])}",
        f"Grad 3 fest: {sig3(b['pe_mean']['Grad 3 fest'])} und {sig3(b['pe_median']['Grad 3 fest'])}")
    assert b["pe_mean"]["BIC"] > b["pe_mean"]["AIC"] and b["pe_median"]["BIC"] > b["pe_median"]["AIC"]
    c = z["s1_n30"]
    has(f"AIC: Mittel {sig3(c['pe_mean']['AIC'])} gegen Median {sig3(c['pe_median']['AIC'])}")
    assert c["pe_mean"]["AIC"] > 30 * c["pe_median"]["AIC"]                                      # wenige extreme Werte bestimmen das Mittel


def test_b2_the_recomputed_cell_equals_the_frozen_cell_at_n_300():
    f = ev.selection_frequency(300, 1.0, 1000, SEED)
    for k in CRIT:
        near(f.share[k][1], FROZEN["zellen"]["s1_n300"]["shares"][k][1], 0.35)
    near(f.pe_mean["BIC"], FROZEN["zellen"]["s1_n300"]["pe_mean"]["BIC"], 0.01 * f.pe_mean["BIC"])


# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

def test_b3_tipping_points_and_the_pair_formula():
    assert [de(T.kipp_lambda(300, p, r), 2) for p, r in ((0.5, "AIC"), (0.5, "BIC"), (0.9, "AIC"), (0.9, "BIC"))] == ["1,96", "5,67", "7,23", "13,46"]
    has("ab `λ` = 1,96 (AIC) und 5,67 (BIC), in 90 % ab 7,23 und 13,46")
    assert de(T.lambda_approx(300, 3.0, 3.0), 2) == "6,86"
    z = FROZEN["zellen"]["s3_n300"]
    assert de(z["mean_lambda"], 2) == "6,74" and pc(z["paar_formel"]["AIC"]) == "88,1 %" and pc(z["paar_formel"]["BIC"]) == "58,3 %"
    has("den Wert 6,86", "das mittlere `λ` über die 1000 Datensätze der Zelle ist 6,74", "Dort liegt der Paarvergleich bei 88,1 % (AIC) und 58,3 % (BIC)")
    assert pc(100 - z["shares"]["AIC"][0]) == "89,7 %" and pc(100 - z["shares"]["BIC"][0]) == "58,2 %"
    has("in 89,7 % (AIC) und 58,2 % (BIC)")
    assert abs(float(T.p_choose_bigger(300, 4, 0.0, "AIC")) - T.p_include_noise(300, 4, "AIC")) < 1e-12


def test_b3_overfit_table_and_limit():
    tab = FROZEN["pool"]["n"]
    ns = [30, 100, 300, 1000, 10000, 100000]
    for rule in ("AIC", "AICc", "BIC"):
        has(" / ".join(de(100 - tab[str(n)][rule][0], 2) for n in ns))
    assert FROZEN["pool"]["reps"] == 1_000_000 and FROZEN["grenzwert"]["reps"] == 2_000_000
    has("1 Mio. Wiederholungen je Zeile", "2 Mio. Wiederholungen")
    lim = FROZEN["grenzwert"]["wert"]
    assert de(lim, 1) == "26,5" and de(100 - lim, 1) == "73,5"
    has("etwa 26,5 %", "höchstens 73,5 % unter den Graden 3 bis 8", "bei nur zwei Kandidaten wäre der Grenzwert 15,73 %")
    assert de(100 - tab["100000"]["BIC"][0], 2) == "0,07"
    has("einen zu großen Grad nur in 0,07 %")


# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

def test_b4_export_numbers():
    has("Bester Näherungsfehler je Grad 0 bis 8 (n → ∞)")
    bias = ev.export_bias2()
    has(", ".join(sci3(b) for b in bias))
    plan = ((30, 1000), (100, 1000), (300, 1000), (1000, 600), (3000, 200))
    res = {n: ev.export_selection(n, r, SEED) for n, r in plan}
    assert [res[n].deg_median["AIC"][1] for n, _ in plan] == [2, 3, 3, 3, 4] and [res[n].deg_median["BIC"][1] for n, _ in plan] == [2, 2, 2, 3, 3]
    has("AIC wählt im Median Grad 2 / 3 / 3 / 3 / 4 bei n = 30 / 100 / 300 / 1000 / 3000, BIC Grad 2 / 2 / 2 / 3 / 3", "(1000, 1000, 1000, 600 und 200 Wiederholungen)")
    ratios = [res[n].ratio[k] for n, _ in plan for k in CRIT]
    assert round(min(ratios), 1) == 1.2 and round(max(ratios), 1) == 1.6
    has("zwischen dem 1,2- und dem 1,6-Fachen des im Nachhinein besten Grades")
    assert de(res[3000].ratio["BIC"], 2) == "1,58" and max(ratios) == res[3000].ratio["BIC"]
    has("der größte Wert ist BIC bei n = 3000 mit 1,58")
    hits = [res[n].hit_best[k] for n, _ in plan for k in CRIT]
    assert 9.0 <= min(hits) <= 12.0 and 24.0 <= max(hits) <= 27.0                              # „etwa 10 bis 26 %“ (200 Wiederholungen bei n = 3000: Band 3 Zählungen)
    has("nur in etwa 10 bis 26 % der Wiederholungen")
    folds = [res[n].fold_differ for n, _ in plan]
    assert 37.0 <= min(folds) <= 39.0 and 42.0 <= max(folds) <= 44.0
    has("verschiedene Grade in etwa 38 bis 43 % der Datensätze")
    assert res[100].deg_median["AIC"][1] == 3
    has("Der gewählte AIC-Grad bei n = 100 ist hier im Median 3")


# --- Experiment 5 ----------------------------------------------------------------------------------------------------------------------------

IMPORT_AIC_BIC = {
    50: dict(exact=(20.9, 22.1), we=(54.2, 33.2), noise=(0.93, 0.31), pe=(0.0683, 0.0616), wahre=0.0271, alle=0.0817),
    100: dict(exact=(33.0, 42.4), we=(74.7, 48.6), noise=(0.86, 0.16), pe=(0.0308, 0.0261), wahre=0.0134, alle=0.0384),
    300: dict(exact=(41.3, 80.5), we=(98.3, 86.6), noise=(0.81, 0.08), pe=(0.00892, 0.00725), wahre=0.00444, alle=0.0119),
    1000: dict(exact=(43.5, 96.3), we=(100.0, 100.0), noise=(0.78, 0.04), pe=(0.00246, 0.00141), wahre=0.00126, alle=0.00341),
}


def test_b5_import_table():
    for n, e in IMPORT_AIC_BIC.items():
        bic_exact = f"**{de(e['exact'][1])} %**" if n == 1000 else f"{de(e['exact'][1])} %"
        we = "100 / 100 %" if n == 1000 else f"{de(e['we'][0])} / {de(e['we'][1])} %"
        row = (f"| {n} | AIC / BIC | {de(e['exact'][0])} / {bic_exact} | {we} | {de(e['noise'][0], 2)} / {de(e['noise'][1], 2)} | "
               f"{sig3(e['pe'][0])} / {sig3(e['pe'][1])} |")
        has(row)
        f = ev.feature_selection(n, 5, 1000, SEED)
        for k, i in (("AIC", 0), ("BIC", 1)):
            near(f.exact[k], e["exact"][i], 0.35 + 0.05)
            near(f.we_in[k], e["we"][i], 0.35 + 0.05)
            near(f.noise_mean[k], e["noise"][i], 0.015)
            assert f.pe_mean[k] == pytest.approx(e["pe"][i], rel=0.012)
        assert f.pe_mean["BIC"] < f.pe_mean["AIC"]                                                 # BIC ist bei jedem der vier n der bessere Vorhersager (im Mittel)
        assert f.pe_mean["wahre Menge"] == pytest.approx(e["wahre"], rel=0.012) and f.pe_mean["alle Merkmale"] == pytest.approx(e["alle"], rel=0.012)
    has("Am Import ist BIC bei jedem der vier n der bessere Vorhersager")
    has("wahre Menge {Zoll, Wochenende} 0,0271 / 0,0134 / 0,00444 / 0,00126, alle sieben Merkmale 0,0817 / 0,0384 / 0,0119 / 0,00341")


def test_b5_aicc_and_loo_numbers():
    expected_aicc = {50: 23.8, 100: 36.1, 300: 44.1, 1000: 43.9}
    for n, v in expected_aicc.items():
        near(ev.feature_selection(n, 5, 1000, SEED).exact["AICc"], v, 0.4)
    aic_exact = {50: 20.9, 100: 33.0, 300: 41.3, 1000: 43.5}
    gaps = [round(expected_aicc[n] - aic_exact[n], 1) for n in expected_aicc]
    assert (min(gaps), max(gaps)) == (0.4, 3.1)
    has("um 0,4 bis 3,1 Punkte häufiger als AIC (23,8 / 36,1 / 44,1 / 43,9 %)")
    loo = {50: (21.3, 1000), 100: (32.5, 600), 300: (46.5, 200), 1000: (43.3, 60)}
    for n, (v, m) in loo.items():
        f = ev.feature_selection(n, 5, 1000, SEED)
        assert f.reps_loo == m
        near(f.exact["LOO"], v, max(0.4, 100 * 3.0 / m))                                           # drei Zählungen bei m Wiederholungen
    has("21,3 % bei n = 50 (1000 Wiederholungen), 32,5 % bei n = 100 (600), 46,5 % bei n = 300 (200) und 43,3 % bei n = 1000 (60)")


# --- Experiment 6 ----------------------------------------------------------------------------------------------------------------------------

AFTER = {
    100: {"ohne Wahl (alle 7 Merkmale)": (100, 94.9, 5.00, 5.1, 22.6), "AIC": (73.5, 96.2, 0.86, 32.0, 24.2), "BIC": (47.0, 94.5, 0.17, 100.0, 15.9),
          "AIC, Datenteilung": (52.6, 94.9, 0.92, 4.9, 4.5), "BIC, Datenteilung": (30.5, 95.9, 0.30, 6.4, 1.9)},
    300: {"ohne Wahl (alle 7 Merkmale)": (100, 95.7, 5.00, 4.9, 21.6), "AIC": (98.2, 97.0, 0.79, 32.8, 22.6), "BIC": (87.4, 97.2, 0.09, 100.0, 8.2),
          "AIC, Datenteilung": (85.2, 95.6, 0.83, 6.0, 4.8), "BIC, Datenteilung": (58.4, 95.2, 0.13, 4.2, 0.5)},
}


def test_b6_single_case_exact():
    thetas = (0, 1, 2, 3, 4)
    aic = [T.coverage_given_inclusion(t, 100, "AIC") for t in thetas]
    bic = [T.coverage_given_inclusion(t, 100, "BIC") for t in thetas]
    assert [pc(100 * c) for c, _ in aic] == ["69,2 %", "90,5 %", "96,5 %", "97,3 %", "95,5 %"]
    assert [pc(100 * c) for c, _ in bic] == ["0,0 %", "80,1 %", "94,4 %", "96,9 %", "97,0 %"]
    assert [pc(100 * p) for _, p in aic] == ["16,3 %", "35,2 %", "72,4 %", "94,4 %", "99,5 %"]
    assert [pc(100 * p) for _, p in bic] == ["3,4 %", "13,0 %", "44,4 %", "80,1 %", "96,7 %"]
    has("AIC bei θ = 0, 1, 2, 3, 4: 69,2 / 90,5 / 96,5 / 97,3 / 95,5 %; BIC: **0,0** / 80,1 / 94,4 / 96,9 / 97,0 %")
    has("Aufnahmewahrscheinlichkeit AIC 16,3 / 35,2 / 72,4 / 94,4 / 99,5 %, BIC 3,4 / 13,0 / 44,4 / 80,1 / 96,7 %")
    thr = T.f_threshold(100, 2, "BIC")
    assert de(math.sqrt(thr), 2) == "2,15" and de(float(stats.t.ppf(0.975, 98)), 2) == "1,98"
    has("die Schwelle `|t| > 2,15` über dem Kritischwert 1,98")
    assert bic[0][0] < 1e-12                                                                                   # BIC, θ = 0: nie überdeckt
    high = [c for c, _ in aic[2:] + bic[2:]]
    assert min(high) > 0.94 and max(high) < 0.974 and sum(c > 0.95 for c in high) == 5 and bic[2][0] < 0.95
    has("bei θ ≥ 2 liegt die Überdeckung in allen sechs Werten zwischen 94,4 und 97,3 %, in fünf davon über 95 %")


@pytest.mark.parametrize("n", [100, 300])
def test_b6_vehicle_table(n):
    a = ev.after_selection(n, 2000, SEED)
    for name, (we, cover, noise, sig, anys) in AFTER[n].items():
        we_t = "100 %" if we == 100 else pc(we)
        sig_t = f"**{pc(sig)}**" if (name == "BIC") else pc(sig)
        label = "ohne Wahl (alle 7)" if name.startswith("ohne") else name
        row = f"| {n} | {label} | {we_t} | {pc(cover)} | {de(noise, 2)} | {sig_t} | {pc(anys)} |"
        has(row)
        near(a.chosen_we[name], we, 0.45)
        near(a.cover_we[name], cover, 0.6)
        near(a.noise_mean[name], noise, 0.015)
        near(a.noise_sig[name], sig, 0.7)
        near(a.any_sig[name], anys, 0.45)
    covers = [AFTER[m][p][1] for m in AFTER for p in AFTER[m]]
    assert min(covers) == 94.5 and max(covers) == 97.2
    has("in allen zehn Zeilen zwischen 94,5 und 97,2 %")
    split_sig = [v[3] for m in AFTER for p, v in AFTER[m].items() if "Datenteilung" in p]
    assert (min(split_sig), max(split_sig)) == (4.2, 6.4)
    has("Die Datenteilung stellt die 5 % her (4,2 bis 6,4 %)")
    has("32,0 und 32,8 % bei AIC, 100,0 % bei BIC")
    has("bei n = 100 in 52,6 % (AIC) und 30,5 % (BIC) gewählt statt in 73,5 und 47,0 %")
    assert abs(100 * (1 - 0.95 ** 5) - 22.6) < 0.05


# --- Presets und Texte -----------------------------------------------------------------------------------------------------------------------

def test_presets_help_names_measured_numbers():
    sh = {k: v["shares"] for k, v in FROZEN["zellen"].items()}
    h = C.PRESET_HELP
    assert f"BIC {pc(sh['s1_n300']['BIC'][1])}, AIC {pc(sh['s1_n300']['AIC'][1])}" in h["Viele Daten, wenig Rauschen"]
    assert f"BIC {pc(sh['s3_n300']['BIC'][1])} ({pc(sh['s3_n300']['BIC'][0])} zu klein), AIC {pc(sh['s3_n300']['AIC'][1])}" in h["Schwaches Signal"]
    eq = [sh["s3_n30"][k][1] for k in CRIT]
    assert f"{de(min(eq), 0)} bis {de(max(eq), 0)} %" in h["Wenig Daten"] and (round(min(eq)), round(max(eq))) == (9, 16)
    e = IMPORT_AIC_BIC[1000]["exact"]
    assert f"BIC wählt genau {{Zoll, Wochenende}} in {pc(e[1])}, AIC in {pc(e[0])}" in h["Rauschmerkmale (Import)"]
    assert "AIC im Median Grad 4" in h["Kein wahrer Grad (Export)"]


def test_app_texts_do_not_use_planning_words():
    text = (ROOT / "app.py").read_text(encoding="utf-8") + (ROOT / "README.md").read_text(encoding="utf-8")
    for word in ("Phase 0", "Vormessung", "Plan Z", " Stück ", "geplant", "Saatwert"):
        assert word not in text, word


def test_readme_names_every_experiment_and_the_kernfragen():
    for i in range(1, 7):
        has(f"**Experiment {i}:**")
    has("Befunde und Korrekturen gegenüber der ersten Rechnung", "Ehrliche Grenzen", "Bewusst nicht umgesetzt")
    assert "TODO" not in README.replace("<!-- TODO nach dem Deploy", "")
