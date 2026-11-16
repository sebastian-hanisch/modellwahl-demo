"""Experimente und Verdikte. Zähleinheit überall: die Wiederholung (ein Datensatz mit n Beobachtungen), wenn nichts anderes dabeisteht.

1  Ein Datensatz, vier Kriterien (Handbeispiel in Brüchen, gezogener Datensatz in Gleitkomma)
2  Wie oft findet ein Kriterium den wahren Grad (Polynom vom Grad 3, Kandidaten 0 bis 8)
3  Signal gegen Schwelle (Nichtzentralität, Überanpassung gegen n; vorgerechnet in mwl_frozen.json)
4  Wenn es keinen wahren Grad gibt (Export-Vorlauf)
5  Welche Merkmale (Import mit Rauschmerkmalen, alle Teilmengen)
6  Was nach der Wahl gilt (Überdeckung, Signifikanz gewählter Rauschmerkmale, Datenteilung)
Die Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

import mwl_constants as C
import mwl_exact as E
import mwl_generator as G
import mwl_rng as R
import mwl_selection as S
import mwl_theory as T

FROZEN_PATH = Path(__file__).resolve().parent / "mwl_frozen.json"


# --- Zahlenformat -----------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def de_sig(x: float) -> str:
    """Kleine Zahl mit drei gültigen Stellen (Prognosefehler), Dezimalkomma; unter 0,0001 als Zehnerpotenz (4,94·10⁻⁹)."""
    if x != 0 and abs(x) < 1e-4:
        mant, exp = f"{x:.2e}".split("e")
        sup = str(int(exp)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{mant}·10{sup}".replace(".", ",")
    return f"{x:.3g}".replace(".", ",")


def de_int(n: int) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner ab 10.000."""
    return f"{n:,}".replace(",", ".") if n >= 10000 else str(n)


def pct(x: float, nd: int = 1) -> str:
    return de(x, nd) + " %"


def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Settings:
    """Alle Regler der App (hashbar für den Zwischenspeicher)."""
    seed: int = C.DEFAULT_SEED
    deg1: int = C.DEG1_DEFAULT
    vz: bool = True
    n1: int = C.N1_DEFAULT
    sigma1: float = C.SIGMA1_DEFAULT
    sigma2: float = C.SIGMA2_DEFAULT
    n2: int = C.N2_DEFAULT
    reps2: int = C.REPS2_DEFAULT
    n3: int = C.N3_DEFAULT
    sigma3: float = C.SIGMA3_DEFAULT
    beta3: float = C.BETA3_DEFAULT
    n4: int = C.N4_DEFAULT
    reps4: int = C.REPS4_DEFAULT
    krit4: str = C.KRIT4_DEFAULT
    n5: int = C.N5_DEFAULT
    noise5: int = C.NOISE5_DEFAULT
    krit5: str = C.KRIT5_DEFAULT
    reps5: int = C.REPS5_DEFAULT
    n6: int = C.N6_DEFAULT
    theta6: float = C.THETA6_DEFAULT
    reps6: int = C.REPS6_DEFAULT


# =====================================================================================================================================
# Experiment 1: ein Datensatz, vier Kriterien
# =====================================================================================================================================

def exact_rows(deg_max: int, count_variance: bool) -> list:
    """Tabelle des Handbeispiels aus Brüchen und 50-stelligen Logarithmen: je Grad RSS, PRESS (Brüche), AIC, AICc, BIC (Zahlen), dazu die Wahl je Kriterium."""
    rows = E.table(E.HAND_X, E.HAND_Y, deg_max, count_variance)
    picks = {"AIC": E.pick(rows, "AIC"), "AICc": E.pick(rows, "AICc"), "BIC": E.pick(rows, "BIC"), "LOO": E.pick(rows, "press")}
    out = []
    for r in rows:
        out.append({"deg": r["deg"], "rss": r["rss"], "press": r["press"], "RSS": float(r["rss"]),
                    "AIC": float(r["AIC"]), "AICc": float(r["AICc"]), "BIC": float(r["BIC"]),
                    "LOO": float("inf") if r["press"] is None else float(r["press"])})
    return out, picks


def drawn_rows(n: int, sigma: float, deg_max: int, count_variance: bool, seed: int):
    """Ein gezogener Datensatz (Polynom vom Grad 3): Tabelle je Grad (Gleitkomma, gestapelte Rechnung mit R = 1) und Wahl je Kriterium."""
    deg_max = min(deg_max, n - 1)
    x, y = G.poly_data(1, n, sigma, seed)
    fam = S.fit_family(x, y, deg_max)
    k = np.arange(deg_max + 1) + 1
    cr = S.criteria_from_rss(fam.rss, n, k, count_variance)
    cr["LOO"] = fam.press / n
    fid = R.ranks_array(R.derive_seed(seed, G.T_FOLD1), 1, n) % C.FOLDS
    cr["CV5"] = S.cv_family(fam.design, y, fid, C.FOLDS)
    picks = {name: int(S.choose(cr[name])[0][0]) for name in cr}
    rows = [{"deg": d, "RSS": float(fam.rss[0, d]), "AIC": float(cr["AIC"][0, d]), "AICc": float(cr["AICc"][0, d]), "BIC": float(cr["BIC"][0, d]),
             "CV5": float(cr["CV5"][0, d]), "LOO": float(cr["LOO"][0, d])} for d in range(deg_max + 1)]
    return x[0], y[0], rows, picks


def monomial_curves(x, y, deg_max: int, grid):
    """Angepasste Kurven aller Grade 0 bis deg_max (Monome, kleinste Quadrate) auf dem Gitter; nur für die Zeichnung."""
    out = []
    for d in range(deg_max + 1):
        coef = np.linalg.lstsq(S.monomial_design(np.asarray(x, float), d), np.asarray(y, float), rcond=None)[0]
        out.append(S.monomial_design(grid, d) @ coef)
    return out


def verdict_hand(picks: dict, deg_max: int, count_variance: bool) -> tuple:
    """Verdikt zu Experiment 1 aus den Wahlen je Kriterium (reine Funktion)."""
    same = {d: [k for k in ("AIC", "BIC", "LOO", "AICc") if picks[k] == d] for d in set(picks.values())}
    parts = [f"{', '.join(v)} wählen Grad {d}" if len(v) > 1 else f"{v[0]} wählt Grad {d}" for d, v in sorted(same.items(), key=lambda kv: -len(kv[1]))]
    text = f"Bei Kandidaten bis Grad {deg_max} und {'mit' if count_variance else 'ohne'} Zählung der Varianz: " + "; ".join(parts) + "."
    if len(same) == 1:
        return "info", text + " Alle Kriterien wählen dasselbe Modell."
    return "warning", text + " Die Kriterien sind sich nicht einig: Wahl und Konvention gehören zusammen."


# =====================================================================================================================================
# Experiment 2: wie oft der wahre Grad
# =====================================================================================================================================

@dataclass
class Frequency:
    n: int
    sigma: float
    reps: int
    share: dict          # Kriterium -> (zu klein, wahr, zu groß) in %
    dist: dict           # Kriterium -> Anteile der Grade 0..8 in %
    pe_mean: dict        # Kriterium und "Grad 3 fest" -> mittlerer Prognosefehler
    pe_median: dict
    ties: dict           # Kriterium -> Zahl der Wiederholungen mit Gleichstand
    mean_lambda: float   # mittlere Nichtzentralität des Kubikterms über die Datensätze


def _grid_poly():
    grid = np.linspace(-1.0, 1.0, C.GRID_POINTS_POLY)
    return S.legendre_design(grid, C.DEG_MAX), G.poly_truth(grid)


def poly_criteria(fam: S.PolyFamily, y: np.ndarray, seed: int, count_variance: bool = True) -> dict:
    n = fam.n
    k = np.arange(fam.degrees) + 1
    cr = S.criteria_from_rss(fam.rss, n, k, count_variance)
    cr["LOO"] = fam.press / n
    fid = R.ranks_array(R.derive_seed(seed, G.T_FOLD1), y.shape[0], n) % C.FOLDS
    cr["CV5"] = S.cv_family(fam.design, y, fid, C.FOLDS)
    return {name: cr[name] for name in C.CRITERIA_POLY}


def selection_frequency(n: int, sigma: float, reps: int, seed: int) -> Frequency:
    x, y = G.poly_data(reps, n, sigma, seed)
    fam = S.fit_family(x, y, C.DEG_MAX)
    cr = poly_criteria(fam, y, seed)
    vg, truth = _grid_poly()
    pe = S.prediction_error(fam, vg, truth)
    share, dist, pe_mean, pe_median, ties = {}, {}, {}, {}, {}
    rows = np.arange(reps)
    for name in C.CRITERIA_POLY:
        idx, tie = S.choose(cr[name])
        share[name] = (100.0 * float(np.mean(idx < C.TRUE_DEGREE)), 100.0 * float(np.mean(idx == C.TRUE_DEGREE)), 100.0 * float(np.mean(idx > C.TRUE_DEGREE)))
        dist[name] = 100.0 * np.bincount(idx, minlength=C.DEG_MAX + 1) / reps
        err = pe[rows, idx]
        pe_mean[name], pe_median[name] = float(err.mean()), float(np.median(err))
        ties[name] = int(tie.sum())
    pe_mean["Grad 3 fest"], pe_median["Grad 3 fest"] = float(pe[:, C.TRUE_DEGREE].mean()), float(np.median(pe[:, C.TRUE_DEGREE]))
    return Frequency(n, sigma, reps, share, dist, pe_mean, pe_median, ties, float(T.lambda_cubic(x, 3.0, sigma).mean()))


def binomial_se(p_percent: float, reps: int) -> float:
    """Standardfehler eines Anteils in Prozentpunkten."""
    p = p_percent / 100.0
    return 100.0 * math.sqrt(max(p * (1.0 - p), 0.0) / reps)


def verdict_frequency(f: Frequency) -> tuple:
    names = C.CRITERIA_POLY
    true = {k: f.share[k][1] for k in names}
    best = max(names, key=lambda k: true[k])
    se = max(binomial_se(true[k], f.reps) for k in names)
    listing = ", ".join(f"{k} {pct(true[k])}" for k in names)
    text = (f"σ = {de(f.sigma, 1)}, n = {f.n}, {f.reps} Wiederholungen: wahrer Grad 3 gewählt von {listing}. Zu kleine Modelle wählen BIC in {pct(f.share['BIC'][0])} und AIC in "
            f"{pct(f.share['AIC'][0])}, zu große AIC in {pct(f.share['AIC'][2])} und BIC in {pct(f.share['BIC'][2])} der Wiederholungen. Standardfehler der Anteile bis {de(se, 1)} Punkte.")
    if true[best] >= 90.0:
        return "success", text + f" Nur {best} findet den wahren Grad hier fast immer."
    if true[best] < 50.0:
        return "warning", text + " Kein Kriterium findet den wahren Grad in mehr als der Hälfte der Wiederholungen."
    return "info", text + f" Am häufigsten findet ihn {best}, aber nicht sicher."


# =====================================================================================================================================
# Experiment 3: Signal gegen Schwelle (Formeln; Zellen und Überanpassung aus mwl_frozen.json)
# =====================================================================================================================================

@dataclass
class Signal:
    n: int
    lam: float
    lam_grid: np.ndarray
    curve: dict           # Regel -> P(Grad 3 gewählt) auf dem λ-Gitter
    p_now: dict           # Regel -> P bei dem λ der Regler
    kipp50: dict
    kipp90: dict
    threshold: dict       # Regel -> F-Schwelle


def signal_curve(n: int, sigma: float, beta3: float) -> Signal:
    grid = np.linspace(0.0, 60.0, 241)
    lam = T.lambda_approx(n, beta3, sigma)
    curve, p_now, k50, k90, thr = {}, {}, {}, {}, {}
    for rule in C.RULES:
        curve[rule] = 100.0 * np.asarray(T.p_choose_bigger(n, 4, grid, rule))
        p_now[rule] = 100.0 * float(T.p_choose_bigger(n, 4, lam, rule))
        k50[rule], k90[rule] = T.kipp_lambda(n, 0.5, rule), T.kipp_lambda(n, 0.9, rule)
        thr[rule] = T.f_threshold(n, 4, rule)
    return Signal(n, lam, grid, curve, p_now, k50, k90, thr)


def verdict_signal(s: Signal) -> tuple:
    text = (f"Bei n = {s.n} wird Grad 3 gegen Grad 2 in 50 % der Fälle gewählt ab λ = {de(s.kipp50['AIC'], 2)} (AIC) und {de(s.kipp50['BIC'], 2)} (BIC), in 90 % ab "
            f"{de(s.kipp90['AIC'], 2)} und {de(s.kipp90['BIC'], 2)}. Die Regler geben λ ≈ {de(s.lam, 1)} (Näherung); dort wählt AIC den Kubikterm in {pct(s.p_now['AIC'])}, "
            f"BIC in {pct(s.p_now['BIC'])} der Wiederholungen (Paarvergleich Grad 2 gegen Grad 3).")
    if s.p_now["AIC"] >= 80 and s.p_now["AIC"] - s.p_now["BIC"] >= 10:
        return "warning", text + " Das Signal liegt zwischen den Schwellen: BIC verwirft hier den echten Term in mindestens 10 Prozentpunkten mehr der Wiederholungen als AIC."
    if s.p_now["BIC"] >= 80:
        return "success", text + " Beide Kriterien wählen den Kubikterm in mindestens 80 % der Wiederholungen."
    return "info", text + " Mindestens ein Kriterium wählt den Kubikterm in weniger als 80 % der Wiederholungen, und der Abstand beider beträgt weniger als 10 Prozentpunkte oder AIC liegt unter 80 %."


def verdict_overfit(frozen: dict) -> tuple:
    tab = frozen["pool"]
    ns = sorted(int(k) for k in tab["n"])
    aic_n = {n: 100.0 - tab["n"][str(n)]["AIC"][0] for n in ns}
    bic_n = {n: 100.0 - tab["n"][str(n)]["BIC"][0] for n in ns}
    limit = frozen["grenzwert"]["wert"]
    text = (f"Unter den Graden 3 bis 8 wählt AIC einen zu großen Grad in {pct(aic_n[ns[0]])} der Wiederholungen bei n = {ns[0]}, {pct(aic_n[300])} bei n = 300 und "
            f"{pct(aic_n[ns[-1]], 2)} bei n = {de_int(ns[-1])}; der Grenzwert n → ∞ liegt bei etwa {pct(limit)} (Irrfahrt mit Schritten χ²₁ − 2). BIC fällt auf {pct(bic_n[ns[-1]], 2)}. "
            f"Gilt für diese Kandidatenmenge (3 bis 8), nicht für jede.")
    return "info", text


# =====================================================================================================================================
# Experiment 4: Export-Kurve (kein wahrer Grad)
# =====================================================================================================================================

@dataclass
class ExportResult:
    n: int
    reps: int
    reps_asked: int
    bias2: np.ndarray                 # beste Näherung je Grad (n → ∞), mittlerer quadratischer Abstand auf dem Gitter
    deg_median: dict                  # Kriterium -> (Q25, Median, Q75)
    dist: dict                        # Kriterium -> Anteile der gewählten Grade in %
    hit_best: dict                    # Anteil, in dem der gewählte Grad der im Nachhinein beste ist (%)
    pe_median: dict                   # Median des Prognosefehlers
    ratio: dict                       # Median des Verhältnisses Prognosefehler gewählt / bester Grad im Nachhinein
    best_median: tuple                # (Q25, Median, Q75) des besten Grades im Nachhinein
    pe_best_median: float
    fold_differ: float                # % der Datensätze, in denen zwei zufällige 5-Faltungen verschiedene Grade wählen


def export_bias2() -> np.ndarray:
    """Beste Polynomnäherung jedes Grades an die wahre Kurve (Grenzwert n → ∞): mittlerer quadratischer Abstand auf 361 Gitterpunkten des Vorlaufs 1 bis 10."""
    v = np.linspace(1.0, 10.0, C.GRID_POINTS_EXPORT)
    vg = S.legendre_design(G.export_x(v), C.DEG_MAX)
    f = G.export_truth(v)
    out = []
    for d in range(C.DEG_MAX + 1):
        b = np.linalg.lstsq(vg[:, : d + 1], f, rcond=None)[0]
        out.append(float(np.mean((vg[:, : d + 1] @ b - f) ** 2)))
    return np.array(out)


def export_reps(n: int, reps: int) -> int:
    """Wiederholungen nach dem Rechenbudget (Wiederholungen × n höchstens BUDGET_EXPORT, mindestens 50)."""
    return int(max(50, min(reps, C.BUDGET_EXPORT // n)))


def export_selection(n: int, reps: int, seed: int) -> ExportResult:
    r_eff = reps
    v, y = G.export_data(r_eff, n, seed)
    x = G.export_x(v)
    fam = S.fit_family(x, y, C.DEG_MAX)
    cr = poly_criteria(fam, y, seed)
    vgrid = np.linspace(1.0, 10.0, C.GRID_POINTS_EXPORT)
    pe = S.prediction_error(fam, S.legendre_design(G.export_x(vgrid), C.DEG_MAX), G.export_truth(vgrid))
    best = np.argmin(pe, axis=1)
    pe_best = pe.min(axis=1)
    rows = np.arange(r_eff)
    out = dict(deg_median={}, dist={}, hit_best={}, pe_median={}, ratio={})
    for name in C.CRITERIA_POLY:
        idx = S.choose(cr[name])[0]
        out["deg_median"][name] = tuple(float(q) for q in np.percentile(idx, [25, 50, 75]))
        out["dist"][name] = 100.0 * np.bincount(idx, minlength=C.DEG_MAX + 1) / r_eff
        out["hit_best"][name] = 100.0 * float(np.mean(idx == best))
        err = pe[rows, idx]
        out["pe_median"][name] = float(np.median(err))
        out["ratio"][name] = float(np.median(err / pe_best))
    cv2 = S.cv_family(fam.design, y, R.ranks_array(R.derive_seed(seed, G.T_FOLD2), r_eff, n) % C.FOLDS, C.FOLDS)
    differ = 100.0 * float(np.mean(S.choose(cr["CV5"])[0] != S.choose(cv2)[0]))
    return ExportResult(n, r_eff, reps, export_bias2(), out["deg_median"], out["dist"], out["hit_best"], out["pe_median"], out["ratio"],
                        tuple(float(q) for q in np.percentile(best, [25, 50, 75])), float(np.median(pe_best)), differ)


def verdict_export(e: ExportResult) -> tuple:
    lo = min(e.ratio.values())
    hi = max(e.ratio.values())
    med = e.deg_median
    text = (f"Es gibt keinen wahren Grad (die Kurve log(0,6 + 0,85·Vorlauf) ist kein Polynom; bester Näherungsfehler bei Grad 3: {de_sig(e.bias2[3])}, bei Grad 8: {de_sig(e.bias2[8])}). "
            f"Bei n = {e.n} ({e.reps} Wiederholungen) wählt AIC im Median Grad {de(med['AIC'][1], 0)} (Quartile {de(med['AIC'][0], 0)} bis {de(med['AIC'][2], 0)}), BIC Grad {de(med['BIC'][1], 0)}. "
            f"Der Prognosefehler der Wahl liegt im Median beim {de(lo, 2)}- bis {de(hi, 2)}-Fachen des im Nachhinein besten Grades; der Grad stimmt mit diesem nur in "
            f"{pct(min(e.hit_best.values()), 0)} bis {pct(max(e.hit_best.values()), 0)} der Wiederholungen überein. Zwei zufällige 5-Faltungen desselben Datensatzes wählen in "
            f"{pct(e.fold_differ)} der Datensätze verschiedene Grade.")
    return "info", text


# =====================================================================================================================================
# Experiment 5: Merkmalswahl am Import
# =====================================================================================================================================

@dataclass
class FeatureResult:
    n: int
    k_noise: int
    reps: int
    reps_loo: int
    exact: dict           # Kriterium -> Anteil genau {Zoll, Wochenende} (%)
    zoll_in: dict
    we_in: dict
    noise_mean: dict      # mittlere Zahl aufgenommener Rauschmerkmale
    noise_any: dict       # Anteil mit mindestens einem Rauschmerkmal (%)
    pe_mean: dict         # Prognosefehler (Mittel), auch "wahre Menge" und "alle Merkmale"
    pe_median: dict
    expected_noise: dict  # AIC, BIC: erwartete Zahl aufgenommener Rauschmerkmale aus der Einzelstufe (k_noise · P(Aufnahme))
    expected_none: dict   # AIC, BIC: erwarteter Anteil ohne Rauschmerkmal (%) aus der Einzelstufe


def feature_reps(n: int, reps: int) -> int:
    """Wiederholungen nach dem Rechenbudget (Wiederholungen × n höchstens BUDGET_FEAT, mindestens 50)."""
    return int(max(50, min(reps, C.BUDGET_FEAT // n)))


def feature_selection(n: int, k_noise: int, reps: int, seed: int) -> FeatureResult:
    x, y = G.import_data(reps, n, k_noise, seed)
    subs = S.all_subsets(k_noise)
    ks = np.array([len(s) for s in subs])
    has_zoll = np.array([1 in s for s in subs])
    has_we = np.array([2 in s for s in subs])
    n_noise = np.array([int(np.sum(s >= 3)) for s in subs])
    rss, _, coef = S.subset_tables(x, y, subs)
    cr = S.criteria_from_rss(rss, n, ks)
    reps_loo = int(min(reps, max(20, C.BUDGET_LOO_FEAT // n)))
    _, press, _ = S.subset_tables(x[:reps_loo], y[:reps_loo], subs, with_loo=True)
    beta = G.import_beta(k_noise)
    m = G.import_moments(k_noise)
    true_idx = next(i for i, s in enumerate(subs) if list(s) == [0, 1, 2])
    full_idx = len(subs) - 1

    def error(idx, count):
        d = coef[np.arange(count), idx] - beta
        return np.einsum("rp,pq,rq->r", d, m, d)

    out = FeatureResult(n, k_noise, reps, reps_loo, {}, {}, {}, {}, {}, {}, {}, {}, {})
    for name in C.CRITERIA_FEAT:
        if name == "LOO":
            idx, count = S.choose(press)[0], reps_loo
        else:
            idx, count = S.choose(cr[name])[0], reps
        err = error(idx, count)
        out.exact[name] = 100.0 * float(np.mean(idx == true_idx))
        out.zoll_in[name] = 100.0 * float(np.mean(has_zoll[idx]))
        out.we_in[name] = 100.0 * float(np.mean(has_we[idx]))
        out.noise_mean[name] = float(np.mean(n_noise[idx]))
        out.noise_any[name] = 100.0 * float(np.mean(n_noise[idx] >= 1))
        out.pe_mean[name], out.pe_median[name] = float(err.mean()), float(np.median(err))
    for label, idx in (("wahre Menge", true_idx), ("alle Merkmale", full_idx)):
        err = error(np.full(reps, idx), reps)
        out.pe_mean[label], out.pe_median[label] = float(err.mean()), float(np.median(err))
    for rule in C.RULES:
        p_in = T.p_include_noise(n, 4, rule) if k_noise else 0.0
        out.expected_noise[rule] = k_noise * p_in
        out.expected_none[rule] = 100.0 * (1.0 - p_in) ** k_noise
    return out


def verdict_features(f: FeatureResult) -> tuple:
    text = (f"n = {f.n}, {f.k_noise} Rauschmerkmale, {f.reps} Wiederholungen: genau {{Zoll, Wochenende}} wählen AIC in {pct(f.exact['AIC'])}, AICc in {pct(f.exact['AICc'])} und BIC in "
            f"{pct(f.exact['BIC'])}; im Mittel nehmen AIC {de(f.noise_mean['AIC'], 2)} und BIC {de(f.noise_mean['BIC'], 2)} Rauschmerkmale auf (Einzelstufe: "
            f"{de(f.expected_noise['AIC'], 2)} und {de(f.expected_noise['BIC'], 2)}). Prognosefehler gegen die wahre Mittelfunktion: AIC {de_sig(f.pe_mean['AIC'])}, BIC {de_sig(f.pe_mean['BIC'])}, "
            f"wahre Menge {de_sig(f.pe_mean['wahre Menge'])}.")
    if f.k_noise == 0:
        return "info", text + " Ohne Rauschmerkmale gibt es nichts Falsches aufzunehmen; es bleibt die Frage, ob das Wochenende (Effekt 0,3) gefunden wird."
    if f.pe_mean["BIC"] < f.pe_mean["AIC"]:
        return "success", text + " Am Import ist BIC hier der bessere Vorhersager."
    return "warning", text + " Hier ist BIC nicht der bessere Vorhersager."


# =====================================================================================================================================
# Experiment 6: nach der Wahl
# =====================================================================================================================================

PROCEDURES = ("ohne Wahl (alle 7 Merkmale)", "AIC", "BIC", "AIC, Datenteilung", "BIC, Datenteilung")


@dataclass
class AfterResult:
    n: int
    reps: int
    chosen_we: dict       # Anteil, in dem das Wochenende gewählt wird (%)
    cover_we: dict        # Überdeckung des naiven 95-%-Intervalls für das Wochenende, gegeben gewählt (%)
    noise_mean: dict      # gewählte Rauschmerkmale je Wiederholung
    noise_sig: dict       # Anteil „signifikanter“ (p < 0,05) unter den gewählten Rauschmerkmalen (%)
    any_sig: dict         # Anteil der Wiederholungen mit mindestens einem „signifikanten“ gewählten Rauschmerkmal (%)


def split_halves(reps: int, n: int, seed: int):
    """Datenteilung: je Wiederholung eine zufällige Zerlegung der Zeilen in die Hälfte zur Wahl (n // 2 Zeilen) und den Rest zur Anpassung; die Mengen sind disjunkt und decken alle Zeilen."""
    order = np.argsort(R.ranks_array(R.derive_seed(seed, G.T_SPLIT), reps, n), axis=1, kind="stable")
    return order[:, : n // 2], order[:, n // 2:]


def _postsel(x, y, idx, subs):
    """Für die gewählten Teilmengen `idx` (je Wiederholung ein Index) die naiven Intervalle und Tests; gruppiert nach Teilmenge."""
    r = x.shape[0]
    chosen_we = np.zeros(r, bool)
    cover = np.zeros(r, bool)
    n_noise = np.zeros(r)
    sig_noise = np.zeros(r)
    for i in np.unique(idx):
        s = subs[i]
        sel = np.flatnonzero(idx == i)
        b, se, df = S.fit_subset(x[sel], y[sel], s)
        tcrit = float(stats.t.ppf(0.975, df))
        if 2 in s:
            pos = list(s).index(2)
            chosen_we[sel] = True
            cover[sel] = np.abs(b[:, pos] - G.BETA_WE) < tcrit * se[:, pos]
        noise_pos = [j for j, c in enumerate(s) if c >= 3]
        if noise_pos:
            sig = np.abs(b[:, noise_pos] / se[:, noise_pos]) > tcrit
            n_noise[sel] = len(noise_pos)
            sig_noise[sel] = sig.sum(axis=1)
    return chosen_we, cover, n_noise, sig_noise


def after_selection(n: int, reps: int, seed: int) -> AfterResult:
    k_noise = C.K_NOISE_MAX
    x, y = G.import_data(reps, n, k_noise, seed)
    subs = S.all_subsets(k_noise)
    ks = np.array([len(s) for s in subs])
    full = len(subs) - 1
    res = AfterResult(n, reps, {}, {}, {}, {}, {})

    def record(name, we, cover, n_noise, sig_noise):
        res.chosen_we[name] = 100.0 * float(np.mean(we))
        res.cover_we[name] = 100.0 * float(cover[we].mean()) if we.any() else float("nan")
        res.noise_mean[name] = float(np.mean(n_noise))
        res.noise_sig[name] = 100.0 * float(sig_noise.sum() / n_noise.sum()) if n_noise.sum() > 0 else float("nan")
        res.any_sig[name] = 100.0 * float(np.mean(sig_noise > 0))

    record(PROCEDURES[0], *_postsel(x, y, np.full(reps, full), subs))
    rss, _, _ = S.subset_tables(x, y, subs)
    cr = S.criteria_from_rss(rss, n, ks)
    for rule in C.RULES:
        record(rule, *_postsel(x, y, S.choose(cr[rule])[0], subs))
    # Datenteilung: Wahl auf einer zufälligen Hälfte (mit dem Kriterium für n_a Beobachtungen), Anpassung und Intervall auf der anderen
    n_a = n // 2
    ia, ib = split_halves(reps, n, seed)
    xa, ya = np.take_along_axis(x, ia[..., None], axis=1), np.take_along_axis(y, ia, axis=1)
    xb, yb = np.take_along_axis(x, ib[..., None], axis=1), np.take_along_axis(y, ib, axis=1)
    rss_a, _, _ = S.subset_tables(xa, ya, subs)
    cr_a = S.criteria_from_rss(rss_a, n_a, ks)
    for rule in C.RULES:
        record(f"{rule}, Datenteilung", *_postsel(xb, yb, S.choose(cr_a[rule])[0], subs))
    return res


def coverage_curve(n: int, thetas: np.ndarray) -> dict:
    """Überdeckung des naiven 95-%-Intervalls gegeben die Aufnahme und Aufnahmewahrscheinlichkeit, exakt, für AIC und BIC (Einzelfall, ein Kandidat)."""
    out = {}
    for rule in C.RULES:
        vals = [T.coverage_given_inclusion(float(t), n, rule) for t in thetas]
        out[rule] = (100.0 * np.array([v[0] for v in vals]), 100.0 * np.array([v[1] for v in vals]))
    return out


def verdict_after(a: AfterResult, theta: float, cov: dict, thetas: np.ndarray) -> tuple:
    j = int(np.argmin(np.abs(thetas - theta)))
    text = (f"Einzelfall (ein Kandidat, n = {a.n}, θ = {de(thetas[j], 1)}): das naive 95-%-Intervall überdeckt gegeben die Aufnahme bei AIC in {pct(cov['AIC'][0][j])}, bei BIC in "
            f"{pct(cov['BIC'][0][j])} der Fälle (ohne Auswahl 95 %). Im Vehikel ({a.reps} Wiederholungen) sind unter den gewählten Rauschmerkmalen {pct(a.noise_sig['AIC'])} (AIC) und "
            f"{pct(a.noise_sig['BIC'])} (BIC) „signifikant“ gegen {pct(a.noise_sig[PROCEDURES[0]])} ohne Wahl; mit Datenteilung {pct(a.noise_sig['AIC, Datenteilung'])} und "
            f"{pct(a.noise_sig['BIC, Datenteilung'])}. Der Preis der Teilung: das Wochenende wird in {pct(a.chosen_we['AIC, Datenteilung'], 0)} statt {pct(a.chosen_we['AIC'], 0)} (AIC) "
            f"und {pct(a.chosen_we['BIC, Datenteilung'], 0)} statt {pct(a.chosen_we['BIC'], 0)} (BIC) gewählt.")
    return "warning", text
