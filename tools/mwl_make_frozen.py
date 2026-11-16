"""Rechnet die vorgerechneten Messreihen und schreibt `mwl_frozen.json` (Wurzelordner der Demo).

    python tools/mwl_make_frozen.py            # neu rechnen und die Datei schreiben (rund 30 s)
    python tools/mwl_make_frozen.py --check    # neu rechnen und mit der Datei vergleichen (Band, Exit-Code 1 bei Abweichung)

Inhalt:
  zellen     die vier Zellen σ ∈ {1, 3} × n ∈ {30, 300} des Experiments 2 (1000 Wiederholungen, Zufalls-Seed 35): Anteil „zu klein / wahr / zu groß“ je Kriterium,
             Prognosefehler, mittlere Nichtzentralität des Kubikterms und die Formel (nichtzentrales F) für Grad 3 gegen Grad 2 bei dieser Nichtzentralität.
  pool       Wahl unter den Graden 3 bis 8 ohne Daten (Quadratsummen von χ²-Größen) für n = 30 bis 100 000, 1 Mio. Wiederholungen je n (Experiment 3).
  grenzwert  Irrfahrt-Grenzwert n → ∞ der AIC-Überanpassung unter den Graden 3 bis 8 (2 Mio. Wiederholungen).
Der Test `tests/test_frozen.py` rechnet einen Teil neu und prüft, dass die Datei zur Suite passt."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mwl_constants as C  # noqa: E402
import mwl_evaluation as ev  # noqa: E402
import mwl_generator as G  # noqa: E402
import mwl_theory as T  # noqa: E402

ZELLEN = ((1.0, 30), (1.0, 300), (3.0, 30), (3.0, 300))
ZELLEN_REPS, ZELLEN_SEED = 1000, 35
POOL_NS = (30, 100, 300, 1000, 10000, 100000)
POOL_REPS, POOL_SEED = 1_000_000, 5
LIMIT_REPS, LIMIT_SEED = 2_000_000, 5


def cell_key(sigma: float, n: int) -> str:
    return f"s{int(sigma)}_n{n}"


def compute_cell(sigma: float, n: int) -> dict:
    f = ev.selection_frequency(n, sigma, ZELLEN_REPS, ZELLEN_SEED)
    x, _ = G.poly_data(ZELLEN_REPS, n, sigma, ZELLEN_SEED)
    lam = T.lambda_cubic(x, 3.0, sigma)
    return {
        "sigma": sigma, "n": n, "reps": ZELLEN_REPS, "seed": ZELLEN_SEED,
        "shares": {k: [round(v, 4) for v in f.share[k]] for k in f.share},
        "pe_mean": {k: float(f"{v:.6g}") for k, v in f.pe_mean.items()},
        "pe_median": {k: float(f"{v:.6g}") for k, v in f.pe_median.items()},
        "mean_lambda": round(float(lam.mean()), 4),
        "paar_formel": {rule: round(100.0 * float(np.mean(T.p_choose_bigger(n, 4, lam, rule))), 4) for rule in C.RULES},
    }


def compute_pool() -> dict:
    table = {}
    for n in POOL_NS:
        w = T.pool_walk(n, POOL_REPS, POOL_SEED)
        table[str(n)] = {k: [round(float(v), 5) for v in w[k]] for k in w}
    return {"reps": POOL_REPS, "seed": POOL_SEED, "degrees": [3, 4, 5, 6, 7, 8], "n": table}


def compute() -> dict:
    return {
        "zellen": {cell_key(s, n): compute_cell(s, n) for s, n in ZELLEN},
        "pool": compute_pool(),
        "grenzwert": {"reps": LIMIT_REPS, "seed": LIMIT_SEED, "wert": round(T.irrfahrt_limit(LIMIT_REPS, LIMIT_SEED), 4)},
    }


def differences(a, b, path="", tol_pct=0.35, tol_rel=2e-3):
    """Abweichungen zwischen zwei Messdateien (Anteile in Prozentpunkten absolut, übrige Zahlen relativ); leere Liste = gleich im Band."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: Schlüssel {sorted(set(a) ^ set(b))}"]
        for k in a:
            out += differences(a[k], b[k], f"{path}/{k}", tol_pct, tol_rel)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: Länge"]
        for i, (u, v) in enumerate(zip(a, b)):
            out += differences(u, v, f"{path}[{i}]", tol_pct, tol_rel)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if "shares" in path or "paar_formel" in path or "/pool/" in path or "grenzwert" in path:
            ok = abs(a - b) <= tol_pct
        else:
            ok = abs(a - b) <= tol_rel * max(abs(a), abs(b), 1e-300)
        if not ok:
            out.append(f"{path}: {a} gegen {b}")
    elif a != b:
        out.append(f"{path}: {a!r} gegen {b!r}")
    return out


def main(argv) -> int:
    new = compute()
    path = ROOT / "mwl_frozen.json"
    if "--check" in argv:
        old = json.loads(path.read_text(encoding="utf-8"))
        diff = differences(old, new)
        print("mwl_frozen.json passt zur Neu-Rechnung (Band)." if not diff else "\n".join(diff))
        return 1 if diff else 0
    path.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"geschrieben: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
