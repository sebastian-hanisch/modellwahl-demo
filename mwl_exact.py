"""Exakte Rechnung für kleine Datensätze: Brüche (`fractions`) für Anpassung, Hebel und Leave-one-out, `decimal` (50 Stellen) für die Logarithmen der Kriterien.

Dient dem Handbeispiel (n = 8, x = -7, -5, ..., 7, y = 3, 0, 8, 3, 5, 6, 9, 6) und als Gegenprobe der gestapelten Gleitkommarechnung in `mwl_selection.py`.
Die Entscheidung zwischen zwei Kriterienwerten wird hier ohne `math.log` getroffen."""

import decimal
from fractions import Fraction as Fr

HAND_X = (-7, -5, -3, -1, 1, 3, 5, 7)
HAND_Y = (3, 0, 8, 3, 5, 6, 9, 6)

_CTX = decimal.Context(prec=50)


def solve(a, b):
    """Gauß-Elimination in Brüchen: löst a x = b."""
    n = len(a)
    m = [[Fr(v) for v in row] + [Fr(bb)] for row, bb in zip(a, b)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        pv = m[c][c]
        m[c] = [v / pv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [p - f * q for p, q in zip(m[r], m[c])]
    return [m[r][n] for r in range(n)]


def design(x, deg):
    return [[Fr(xi) ** j for j in range(deg + 1)] for xi in x]


def gram(x_rows):
    p = len(x_rows[0])
    return [[sum(r[a] * r[b] for r in x_rows) for b in range(p)] for a in range(p)]


def fit(x, y, deg):
    """Koeffizienten (aufsteigend), Residuen, Rest-Quadratsumme des Polynoms vom Grad deg, alles Brüche."""
    rows = design(x, deg)
    p = deg + 1
    g = gram(rows)
    rhs = [sum(r[a] * Fr(yi) for r, yi in zip(rows, y)) for a in range(p)]
    b = solve(g, rhs)
    e = [Fr(yi) - sum(bj * v for bj, v in zip(b, r)) for yi, r in zip(y, rows)]
    return b, e, sum(v * v for v in e)


def hat_diagonal(x, deg):
    """Hebel h_i = x_i' (X'X)^-1 x_i als Brüche."""
    rows = design(x, deg)
    p = deg + 1
    g = gram(rows)
    cols = [solve(g, [Fr(int(i == j)) for i in range(p)]) for j in range(p)]
    out = []
    for r in rows:
        tmp = [sum(cols[j][i] * r[j] for j in range(p)) for i in range(p)]
        out.append(sum(r[i] * tmp[i] for i in range(p)))
    return out


def press(x, y, deg):
    """Leave-one-out-Quadratsumme Σ (e_i / (1 - h_i))² als Bruch; None, wenn ein Hebel 1 ist (nicht definiert)."""
    _, e, _ = fit(x, y, deg)
    h = hat_diagonal(x, deg)
    if any(hi == 1 for hi in h):
        return None
    return sum((ei / (1 - hi)) ** 2 for ei, hi in zip(e, h))


def press_by_refit(x, y, deg):
    """Dieselbe Größe auf dem zweiten Weg: Punkt i weglassen, neu anpassen, den Fehler an Punkt i quadrieren (Brüche)."""
    total = Fr(0)
    for i in range(len(x)):
        xs, ys = list(x[:i]) + list(x[i + 1:]), list(y[:i]) + list(y[i + 1:])
        b, _, _ = fit(xs, ys, deg)
        total += (Fr(y[i]) - sum(bj * Fr(x[i]) ** j for j, bj in enumerate(b))) ** 2
    return total


def _ln(q: Fr) -> decimal.Decimal:
    if q == 0:
        return decimal.Decimal("-Infinity")
    return _CTX.ln(decimal.Decimal(q.numerator) / decimal.Decimal(q.denominator))


def criteria(rss: Fr, n: int, deg: int, count_variance: bool = True) -> dict:
    """AIC, AICc, BIC als Decimal (50 Stellen); k = deg + 1 Koeffizienten, K = k + 1 mit der Varianz. AICc mit n - K - 1 <= 0 ist +Infinity."""
    kk = deg + 1 + (1 if count_variance else 0)
    base = n * _ln(rss / n)
    aic = base + 2 * kk
    corr = n - kk - 1
    aicc = aic + decimal.Decimal(2 * kk * (kk + 1)) / decimal.Decimal(corr) if corr > 0 else decimal.Decimal("Infinity")
    bic = base + kk * _CTX.ln(decimal.Decimal(n))
    return {"AIC": aic, "AICc": aicc, "BIC": bic}


def table(x, y, deg_max, count_variance=True):
    """Eine Zeile je Grad: RSS und PRESS (Brüche), AIC, AICc, BIC (Decimal). Grade ab n - 1 haben Hebel 1 (PRESS None)."""
    n = len(x)
    rows = []
    for d in range(min(deg_max, n - 1) + 1):
        _, _, rss = fit(x, y, d)
        rows.append({"deg": d, "rss": rss, "press": press(x, y, d), **criteria(rss, n, d, count_variance)})
    return rows


def pick(rows, key):
    """Grad mit dem kleinsten Wert von `key` (Gleichstand: kleinster Grad); PRESS None zählt als unendlich."""
    vals = {}
    for r in rows:
        v = r[key]
        vals[r["deg"]] = decimal.Decimal("Infinity") if v is None else (decimal.Decimal(v.numerator) / decimal.Decimal(v.denominator) if isinstance(v, Fr) else v)
    best = min(vals.values())
    return min(d for d, v in vals.items() if v == best)
