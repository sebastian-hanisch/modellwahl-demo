"""Auswahlkriterien für lineare Modelle mit Normalfehlern: AIC, AICc, BIC, Leave-one-out (PRESS) und K-fache Kreuzvalidierung, gestapelt über Wiederholungen.

Konvention (README, Abschnitt „Konventionen“): `k` ist die Zahl der Koeffizienten eines Modells (Achsenabschnitt eingeschlossen), `K = k + 1` zählt die Varianz mit
(abschaltbar mit `count_variance=False`). Mit `RSS` der Rest-Quadratsumme und `n` den Beobachtungen:
  AIC = n ln(RSS/n) + 2K,   AICc = AIC + 2K(K+1)/(n-K-1),   BIC = n ln(RSS/n) + K ln n,   LOO = Σ (e_i/(1-h_i))² / n,   CVk = Σ über k Faltungen der Quadratsummen / n.
Kleiner ist besser. Gleichstände (Abstand höchstens `TIE_TOL`, relativ) gehen an das kleinste Modell; `choose` zählt sie.

Rechnung: Polynomgrade über eine QR-Zerlegung der Legendre-Basis (RSS und Hebel jedes Grades aus den Spalten von Q), Kreuzvalidierung über Gram-Matrizen je Faltung,
Teilmengen von Merkmalen über die Gram-Matrix (Normalgleichung; bei höchstens acht gut skalierten Spalten unkritisch). Nur numpy."""

from dataclasses import dataclass

import numpy as np

TIE_TOL = 1e-9
HAT_ONE = 1.0 - 1e-9          # Hebel ab hier gilt als 1: Leave-one-out ist dort nicht definiert
RSS_ZERO = 1e-12              # RSS unter diesem Anteil von y'y gilt als null (Interpolation): AIC = -inf


def legendre_design(x: np.ndarray, deg_max: int) -> np.ndarray:
    """Legendre-Polynome P0 bis P_deg_max an den Stellen x in [-1, 1]; Form (..., n, deg_max + 1). Gleicher Spaltenraum wie die Monome, aber gut konditioniert."""
    cols = [np.ones_like(x), x]
    for k in range(1, deg_max):
        cols.append(((2 * k + 1) * x * cols[k] - k * cols[k - 1]) / (k + 1))
    return np.stack(cols[: deg_max + 1], axis=-1)


def monomial_design(x: np.ndarray, deg_max: int) -> np.ndarray:
    return np.stack([x ** j for j in range(deg_max + 1)], axis=-1)


def criteria_from_rss(rss, n: int, k, count_variance: bool = True) -> dict:
    """AIC, AICc, BIC aus Rest-Quadratsummen (Form (..., M)) und den Koeffizientenzahlen k (Form (M,)). RSS = 0 gibt -inf, AICc mit n - K - 1 <= 0 gibt +inf."""
    rss = np.asarray(rss, dtype=float)
    kk = np.asarray(k, dtype=float) + (1.0 if count_variance else 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = n * np.log(rss / n)
        aic = base + 2.0 * kk
        corr = n - kk - 1.0
        aicc = np.where(corr > 0, aic + 2.0 * kk * (kk + 1.0) / np.where(corr > 0, corr, 1.0), np.inf)
        bic = base + kk * np.log(n)
    return {"AIC": aic, "AICc": aicc, "BIC": bic}


def choose(crit: np.ndarray, tol: float = TIE_TOL):
    """Index des kleinsten Kriteriumswerts je Zeile; bei Gleichstand der kleinste Index. Rückgabe (Index, Gleichstand ja/nein je Zeile). nan zählt als undefiniert (+inf)."""
    c = np.where(np.isnan(crit), np.inf, crit)
    m = c.min(axis=-1)
    with np.errstate(invalid="ignore"):
        thr = np.where(np.isfinite(m), m + tol * np.maximum(1.0, np.abs(m)), m)
    within = c <= thr[..., None]
    return np.argmax(within, axis=-1), within.sum(axis=-1) > 1


# --- Polynomgrade ---------------------------------------------------------------------------------------------------------------------

@dataclass
class PolyFamily:
    """Anpassung aller Grade 0 bis deg_max an jede Wiederholung."""
    n: int
    rss: np.ndarray       # (R, D) Rest-Quadratsumme je Grad
    press: np.ndarray     # (R, D) Σ (e/(1-h))², +inf bei Hebel 1
    c: np.ndarray         # (R, D) Koordinaten Q'y
    rfac: np.ndarray      # (R, D, D) R-Faktor der QR-Zerlegung (Koeffizienten des Grades d: R[:d+1,:d+1] b = c[:d+1])
    design: np.ndarray    # (R, n, D)

    @property
    def degrees(self) -> int:
        return self.rss.shape[1]


def fit_family(x: np.ndarray, y: np.ndarray, deg_max: int, basis: str = "legendre") -> PolyFamily:
    """x, y: Form (R, n); deg_max <= n - 1 (sonst ist die Zerlegung nicht schmal). Basis 'legendre' (Standard) oder 'monom' (Eigenschaftstest: gleiche Rest-Quadratsummen)."""
    n = x.shape[1]
    if deg_max > n - 1:
        raise ValueError("Grad größer als n - 1")
    v = legendre_design(x, deg_max) if basis == "legendre" else monomial_design(x, deg_max)
    q, rfac = np.linalg.qr(v)
    c = np.einsum("rnd,rn->rd", q, y)
    fitted = np.cumsum(q * c[:, None, :], axis=2)
    e = y[..., None] - fitted
    rss = (e ** 2).sum(axis=1)
    h = np.cumsum(q ** 2, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        press = np.where((h >= HAT_ONE).any(axis=1), np.inf, ((e / (1.0 - h)) ** 2).sum(axis=1))
    yy = (y ** 2).sum(axis=1)
    rss = np.where(rss < RSS_ZERO * np.maximum(yy, 1.0)[:, None], 0.0, rss)
    return PolyFamily(n, rss, press, c, rfac, v)


def prediction_error(fam: PolyFamily, grid_design: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Mittlerer quadratischer Abstand der angepassten Kurve jedes Grades zur wahren Kurve auf dem Gitter; Form (R, D)."""
    r, d_all = fam.rss.shape
    out = np.empty((r, d_all))
    for d in range(d_all):
        b = np.linalg.solve(fam.rfac[:, : d + 1, : d + 1], fam.c[:, : d + 1, None])[..., 0]
        pred = b @ grid_design[:, : d + 1].T
        out[:, d] = ((pred - truth[None, :]) ** 2).mean(axis=1)
    return out


def cv_family(v: np.ndarray, y: np.ndarray, fold_id: np.ndarray, folds: int) -> np.ndarray:
    """K-fache Kreuzvalidierung für alle Grade (führende Spalten von v): mittlerer quadratischer Vorhersagefehler, Form (R, D).
    Je Faltung wird die Trainings-Normalgleichung aus den Gram-Matrizen (gesamt minus Faltung) gelöst. Gilt nur, wenn n >= folds und das Training mehr Punkte als Koeffizienten hat (sonst +inf)."""
    r, n, d_all = v.shape
    out = np.full((r, d_all), np.inf)
    if n < folds:
        return out
    gram = v.transpose(0, 2, 1) @ v
    gy = v.transpose(0, 2, 1) @ y[..., None]
    total = np.zeros((r, d_all))
    eye = np.eye(d_all)[None]
    for f in range(folds):
        m = (fold_id == f).astype(float)
        vf = v * m[..., None]
        gf = vf.transpose(0, 2, 1) @ v
        gyf = vf.transpose(0, 2, 1) @ y[..., None]
        yf2 = ((y ** 2) * m).sum(axis=1)
        size_train = n - m.sum(axis=1)
        for d in range(d_all):
            ok = size_train >= d + 1
            t = np.where(ok[:, None, None], (gram - gf)[:, : d + 1, : d + 1], eye[:, : d + 1, : d + 1])
            b = np.linalg.solve(t, (gy - gyf)[:, : d + 1, :])
            sse = yf2 - 2.0 * (b * gyf[:, : d + 1, :]).sum(axis=(1, 2)) + np.einsum("rik,rij,rjk->r", b, gf[:, : d + 1, : d + 1], b)
            total[:, d] += np.where(ok, sse, np.inf)
    return total / n


# --- Teilmengen von Merkmalen -----------------------------------------------------------------------------------------------------

def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Löst a x = b für einen Stapel; ist eine Gram-Matrix singulär (etwa eine leere Zoll-Spalte in einer kleinen Hälfte der Datenteilung), gilt für den ganzen Stapel die Pseudoinverse."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


def _inv(a: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a)



def all_subsets(k_noise: int) -> list:
    """Alle Teilmengen der Spalten 1 (Zoll), 2 (Wochenende) und der Rauschmerkmale; der Achsenabschnitt (Spalte 0) ist immer dabei. 2^(2 + k_noise) Teilmengen, Reihenfolge nach den Bits."""
    out = []
    for code in range(2 ** (2 + k_noise)):
        out.append(np.array([0] + [j + 1 for j in range(2 + k_noise) if (code >> (1 + k_noise - j)) & 1]))
    return out


def subset_tables(x: np.ndarray, y: np.ndarray, subs: list, with_loo: bool = False):
    """Rest-Quadratsumme, PRESS (auf Wunsch, sonst None) und Koeffizienten (mit Nullen außerhalb der Teilmenge, Form (R, S, p)) jeder Teilmenge für jede Wiederholung. x: (R, n, p), y: (R, n)."""
    r, n, p = x.shape
    gram = x.transpose(0, 2, 1) @ x
    gy = (x.transpose(0, 2, 1) @ y[..., None])[..., 0]
    yy = (y ** 2).sum(axis=1)
    rss = np.empty((r, len(subs)))
    coef = np.zeros((r, len(subs), p))
    press = np.empty((r, len(subs))) if with_loo else None
    for i, s in enumerate(subs):
        gs = gram[:, s[:, None], s[None, :]]
        b = _solve(gs, gy[:, s, None])[..., 0]
        coef[:, i, s] = b
        rss[:, i] = yy - (b * gy[:, s]).sum(axis=1)
        if with_loo:
            xs = x[:, :, s]
            h = np.einsum("rnk,rkn->rn", xs, _solve(gs, xs.transpose(0, 2, 1)))
            e = y - np.einsum("rnk,rk->rn", xs, b)
            press[:, i] = np.where((h >= HAT_ONE).any(axis=1), np.inf, ((e / (1.0 - h)) ** 2).sum(axis=1))
    return rss, press, coef


def fit_subset(x: np.ndarray, y: np.ndarray, s: np.ndarray):
    """OLS in der Teilmenge s für jede Wiederholung: Koeffizienten (R, k), Standardfehler (R, k), Freiheitsgrade n - k."""
    n = x.shape[1]
    xs = x[:, :, s]
    gs = xs.transpose(0, 2, 1) @ xs
    gi = _inv(gs)
    b = np.einsum("rkj,rj->rk", gi, np.einsum("rnk,rn->rk", xs, y))
    e = y - np.einsum("rnk,rk->rn", xs, b)
    s2 = (e ** 2).sum(axis=1) / (n - len(s))
    se = np.sqrt(s2[:, None] * np.diagonal(gi, axis1=1, axis2=2))
    empty = np.diagonal(gs, axis1=1, axis2=2) <= 0                    # Spalte ohne Wirkung auf die Anpassung (nur Nullen): keine Information, Standardfehler unendlich
    return b, np.where(empty, np.inf, se), n - len(s)
