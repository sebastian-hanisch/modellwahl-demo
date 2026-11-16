"""Geschlossene Formeln und Rechnungen ohne Daten zur Wahl nach AIC und BIC (Normalfehler, geschachtelte Modelle).

1. Das Kriterium ist eine F-Schwelle. Gegeben ein kleineres Modell und ein größeres mit einem Koeffizienten mehr (k_big Koeffizienten, n - k_big Freiheitsgrade des größeren):
   das größere gewinnt genau dann, wenn  F > (n - k_big)(exp(pen/n) - 1),  pen = 2 (AIC) oder ln n (BIC). (Aus n ln(RSS_klein/RSS_groß) > pen.)
2. Ein reines Rauschmerkmal wird mit der Wahrscheinlichkeit P(F(1, n - k_big) > Schwelle) aufgenommen (zentrales F).
3. Hat das Merkmal die Nichtzentralität λ = β² · ||(I - H) x||² / σ², ist die Aufnahmewahrscheinlichkeit P(F'(1, n - k_big; λ) > Schwelle).
4. Wahl unter den Graden 3 bis 8 bei einem Erzeuger vom Grad 3: Die Entscheidung hängt nur an n. Die Rest-Quadratsumme des Grades 8 ist χ² mit n - 9 Freiheitsgraden, jeder niedrigere
   Grad addiert die Quadrate weiterer Normalkoordinaten (χ² mit einem Freiheitsgrad); σ, β und Design kürzen sich.
5. Überdeckung des naiven 95-%-Intervalls gegeben die Aufnahme (ein Kandidat, orthogonal codiert): Integral über die Verteilung von s².
Nur numpy und scipy."""

import math

import numpy as np
from scipy import integrate, optimize, special, stats

import mwl_rng as R
import mwl_selection as S


def penalty(rule: str, n: int) -> float:
    if rule == "AIC":
        return 2.0
    if rule == "BIC":
        return math.log(n)
    raise ValueError(rule)


def f_threshold(n: int, k_big: int, rule: str) -> float:
    """F-Schwelle, ab der das größere Modell (k_big Koeffizienten) gewählt wird."""
    return (n - k_big) * (math.exp(penalty(rule, n) / n) - 1.0)


def p_include_noise(n: int, k_big: int, rule: str) -> float:
    """Wahrscheinlichkeit, dass ein reines Rauschmerkmal aufgenommen wird."""
    return float(stats.f.sf(f_threshold(n, k_big, rule), 1, n - k_big))


def p_choose_bigger(n: int, k_big: int, lam, rule: str):
    """Wahrscheinlichkeit, dass das größere Modell gewählt wird, bei Nichtzentralität λ (Feld erlaubt)."""
    lam = np.asarray(lam, dtype=float)
    thr, df = f_threshold(n, k_big, rule), n - k_big
    # scipy.stats.ncf liefert bei nc = 0 unbrauchbare Werte (gemessen: -0,84 bei n = 300): ab λ unter 1e-6 gilt das zentrale F
    return np.where(lam < 1e-6, stats.f.sf(thr, 1, df), stats.ncf.sf(thr, 1, df, np.maximum(lam, 1e-6)))


def kipp_lambda(n: int, p: float, rule: str, k_big: int = 4) -> float:
    """λ, ab dem das größere Modell mit Wahrscheinlichkeit p gewählt wird."""
    return float(optimize.brentq(lambda lam: float(p_choose_bigger(n, k_big, lam, rule)) - p, 0.0, 400.0))


def lambda_cubic(x: np.ndarray, beta3: float, sigma: float) -> np.ndarray:
    """Nichtzentralität des Kubikterms je Wiederholung: β3² · Rest-Quadratsumme von x³ nach {1, x, x²} / σ². x: (R, n). Über die QR-Zerlegung der Monome."""
    v = np.stack([x ** j for j in range(4)], axis=-1)
    r44 = np.linalg.qr(v)[1][:, 3, 3]
    with np.errstate(divide="ignore"):
        return (beta3 ** 2) * r44 ** 2 / sigma ** 2


LAMBDA_APPROX_FACTOR = 4.0 / 175.0   # Var(x³ - 3x/5) bei Gleich(-1, 1): Rest von x³ nach {1, x, x²}; λ ≈ n · β3² · 4/175 / σ²


def lambda_approx(n: int, beta3: float, sigma: float) -> float:
    return n * beta3 ** 2 * LAMBDA_APPROX_FACTOR / sigma ** 2


# --- Wahl unter den Graden 3 bis 8 ohne Daten (Quadratsummen von χ²-Größen) ------------------------------------------------------------

POOL_DEGREES = np.arange(3, 9)


def _pool_rss(n: int, reps: int, seed: int, tag: int) -> np.ndarray:
    u = R.uniform_array(R.derive_seed(seed, tag), reps)
    r8 = stats.chi2.ppf(np.clip(u, 1e-300, 1 - 1e-16), n - 9)
    inc = R.normal_matrix(R.derive_seed(seed, tag + 500), (reps, 5)) ** 2
    rss = np.empty((reps, 6))
    rss[:, 5] = r8
    for j in range(4, -1, -1):
        rss[:, j] = rss[:, j + 1] + inc[:, j]
    return rss


def pool_walk(n: int, reps: int, seed: int, chunk: int = 250_000) -> dict:
    """Anteil (in %) der gewählten Grade 3 bis 8 je Kriterium, entwurfsfrei; Rückgabe {Kriterium: Feld (6,)}. Gleiches n, gleiche Zufallszahlen für alle Kriterien."""
    counts = {k: np.zeros(6) for k in ("AIC", "AICc", "BIC")}
    done, tag = 0, 100
    while done < reps:
        m = min(chunk, reps - done)
        rss = _pool_rss(n, m, seed, tag)
        cr = S.criteria_from_rss(rss, n, POOL_DEGREES + 1)
        for name in counts:
            counts[name] += np.bincount(S.choose(cr[name])[0], minlength=6)
        done += m
        tag += 1
    return {k: 100.0 * v / reps for k, v in counts.items()}


def irrfahrt_limit(reps: int, seed: int, chunk: int = 250_000) -> float:
    """Grenzwert n → ∞ der Wahrscheinlichkeit, dass AIC unter den Graden 3 bis 8 einen höheren Grad als 3 wählt: das Maximum einer Irrfahrt mit fünf Schritten χ²₁ - 2 liegt über null (in %)."""
    hits, done, tag = 0, 0, 900
    while done < reps:
        m = min(chunk, reps - done)
        steps = R.normal_matrix(R.derive_seed(seed, tag), (m, 5)) ** 2 - 2.0
        hits += int(np.sum(np.cumsum(steps, axis=1).max(axis=1) > 0))
        done += m
        tag += 1
    return 100.0 * hits / reps


# --- Überdeckung gegeben die Aufnahme -------------------------------------------------------------------------------------------------

def coverage_given_inclusion(theta: float, n: int, rule: str):
    """Ein Kandidat mit Wirkung β, orthogonal zum Achsenabschnitt codiert, θ = β / Standardfehler, n Beobachtungen (n - 2 Freiheitsgrade).
    Rückgabe (Überdeckung des naiven 95-%-t-Intervalls gegeben die Aufnahme, Wahrscheinlichkeit der Aufnahme), beides exakt (Integral über die Verteilung von s²)."""
    df = n - 2
    c = math.sqrt(df * (math.exp(penalty(rule, n) / n) - 1.0))   # Aufnahme genau dann, wenn |t| > c
    tcrit = float(stats.t.ppf(0.975, df))

    def both(v):
        a, b = tcrit * math.sqrt(v), c * math.sqrt(v)
        lo, hi = max(-a, -b - theta), min(a, b - theta)
        overlap = max(special.ndtr(hi) - special.ndtr(lo), 0.0) if hi > lo else 0.0
        return (2 * special.ndtr(a) - 1) - overlap

    def inc(v):
        b = c * math.sqrt(v)
        return 1.0 - (special.ndtr(b - theta) - special.ndtr(-b - theta))

    def dens(v):
        return df * math.exp((df / 2.0 - 1.0) * math.log(df * v) - df * v / 2.0 - (df / 2.0) * math.log(2.0) - math.lgamma(df / 2.0)) if v > 0 else 0.0

    num = integrate.quad(lambda v: both(v) * dens(v), 0, 10, limit=200)[0]
    den = integrate.quad(lambda v: inc(v) * dens(v), 0, 10, limit=200)[0]
    return num / den, den
