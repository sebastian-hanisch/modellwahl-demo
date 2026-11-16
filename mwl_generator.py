"""Erzeuger der Datensätze: ein erfundenes kubisches Polynom und zwei Ausschnitte des Container-Verweilzeit-Vehikels (Import, Export).

Herkunft der Parameter des Vehikels: `vwz_constants.py` und `vwz_generator.py` der Demo verweilzeit-demo (nur gelesen, nichts importiert): zwölf Kundeneffekte
`UK = default_rng(12345).normal(0, 0,3, 12)` (hier als Zahlen eingetragen, `tests/test_generator.py` prüft sie gegen numpy), Import-Median `3,5 · exp(0,7 Zoll + 0,3 Wochenende)`
mit Streuung 0,55 und Kundeneffekt Faktor 1, Export-Median `0,6 + 0,85 Vorlauf` mit Streuung 0,25 und Kundeneffekt Faktor 0,3, Zoll mit Wahrscheinlichkeit 0,25, Wochenende 2/7.
Alles ist erfunden (kein Echtdatenbezug). Die zusätzlichen Rauschmerkmale des Imports (bis zu fünf Normalmerkmale ohne Wirkung) sind ein Zusatz dieser Demo, im Generator der
Verweilzeit-Demo gibt es nur das eine Merkmal „gewicht“. Das Polynom ist kein Verweilzeit-Vehikel: eine Messung „wird der wahre Grad gefunden“ braucht einen Erzeuger mit bekanntem Grad.

Alle Felder sind Stapel von Wiederholungen: Form (Wiederholungen, n, ...). Zufall nur aus `mwl_rng` (SplitMix64); die ersten Wiederholungen hängen nicht von der Zahl der Wiederholungen ab."""

import numpy as np

import mwl_rng as R

POLY_COEF = (0.5, 1.0, -2.0, 3.0)            # wahres Polynom y = 0,5 + x - 2 x^2 + 3 x^3, wahrer Grad 3
UK = np.array([-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
               -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771])
K_KUNDEN = 12
P_ZOLL, P_WE = 0.25, 2.0 / 7.0
LOG_MEDIAN_IMPORT = float(np.log(3.5))
BETA_ZOLL, BETA_WE = 0.7, 0.3
S_IMPORT, S_EXPORT, U_EXPORT = 0.55, 0.25, 0.3
VAR_KUNDEN = float(np.mean((UK - UK.mean()) ** 2))     # Populationsvarianz der zwölf Effekte: 0,1319
SIGMA2_IMPORT = S_IMPORT ** 2 + VAR_KUNDEN              # Fehlervarianz, wenn die Kunden im Modell fehlen
SIGMA2_EXPORT = S_EXPORT ** 2 + U_EXPORT ** 2 * VAR_KUNDEN

# Zwecke der Zufallsströme (derive_seed)
T_X, T_EPS, T_ZOLL, T_WE, T_KUNDE, T_NOISE, T_FOLD1, T_FOLD2, T_SPLIT = 1, 2, 3, 4, 5, 6, 7, 8, 9


def poly_truth(x):
    return POLY_COEF[0] + POLY_COEF[1] * x + POLY_COEF[2] * x ** 2 + POLY_COEF[3] * x ** 3


def poly_data(reps: int, n: int, sigma: float, seed: int):
    """x ~ Gleich(-1, 1), y = wahres Polynom + sigma * Normal. Rückgabe x, y mit Form (reps, n)."""
    x = 2.0 * R.uniform_matrix(R.derive_seed(seed, T_X), (reps, n)) - 1.0
    y = poly_truth(x) + sigma * R.normal_matrix(R.derive_seed(seed, T_EPS), (reps, n))
    return x, y


def import_beta(k_noise: int) -> np.ndarray:
    """Wahre Koeffizienten (Achsenabschnitt, Zoll, Wochenende, Rauschmerkmale = 0); der Achsenabschnitt enthält den Mittelwert der Kundeneffekte."""
    return np.array([LOG_MEDIAN_IMPORT + float(UK.mean()), BETA_ZOLL, BETA_WE] + [0.0] * k_noise)


def import_moments(k_noise: int) -> np.ndarray:
    """E[x x'] der Merkmale (1, Zoll, Wochenende, Rauschmerkmale), unabhängig gezogen: exakt, für den Prognosefehler gegen die wahre Mittelfunktion."""
    p = 3 + k_noise
    mean = np.array([1.0, P_ZOLL, P_WE] + [0.0] * k_noise)
    m = np.outer(mean, mean)
    m[1, 1], m[2, 2] = P_ZOLL, P_WE
    for j in range(3, p):
        m[j, j] = 1.0
    return m


def import_data(reps: int, n: int, k_noise: int, seed: int):
    """Import-Container: log-Verweilzeit = b0 + 0,7 Zoll + 0,3 Wochenende + Kundeneffekt + 0,55 Normal; dazu k_noise Normalmerkmale ohne Wirkung.
    Die Kunden bleiben im Modell weg (wirken als Teil des Fehlers). X hat Form (reps, n, 3 + k_noise), Spalten 1, Zoll, Wochenende, Rauschen."""
    zoll = (R.uniform_matrix(R.derive_seed(seed, T_ZOLL), (reps, n)) < P_ZOLL).astype(float)
    we = (R.uniform_matrix(R.derive_seed(seed, T_WE), (reps, n)) < P_WE).astype(float)
    kunde = np.minimum((R.uniform_matrix(R.derive_seed(seed, T_KUNDE), (reps, n)) * K_KUNDEN).astype(int), K_KUNDEN - 1)
    eps = R.normal_matrix(R.derive_seed(seed, T_EPS), (reps, n))
    noise = R.normal_matrix(R.derive_seed(seed, T_NOISE), (reps, n, k_noise)) if k_noise else np.zeros((reps, n, 0))
    y = LOG_MEDIAN_IMPORT + BETA_ZOLL * zoll + BETA_WE * we + UK[kunde] + S_IMPORT * eps
    X = np.concatenate([np.ones((reps, n, 1)), zoll[..., None], we[..., None], noise], axis=2)
    return X, y


def export_x(v):
    """Vorlauf 1 bis 10 Tage auf [-1, 1] abgebildet (für die Polynombasis)."""
    return (v - 5.5) / 4.5


def export_truth(v):
    """Wahre Mittelfunktion der log-Verweilzeit am Export: log(0,6 + 0,85 v) plus der mittlere Kundeneffekt. Kein Polynom."""
    return np.log(0.6 + 0.85 * v) + U_EXPORT * float(UK.mean())


def export_data(reps: int, n: int, seed: int):
    """Export-Container: log-Verweilzeit = log(0,6 + 0,85 Vorlauf) + 0,3 Kundeneffekt + 0,25 Normal; Vorlauf ~ Gleich(1, 10). Rückgabe Vorlauf, y mit Form (reps, n)."""
    v = 1.0 + 9.0 * R.uniform_matrix(R.derive_seed(seed, T_X), (reps, n))
    kunde = np.minimum((R.uniform_matrix(R.derive_seed(seed, T_KUNDE), (reps, n)) * K_KUNDEN).astype(int), K_KUNDEN - 1)
    eps = R.normal_matrix(R.derive_seed(seed, T_EPS), (reps, n))
    y = np.log(0.6 + 0.85 * v) + U_EXPORT * UK[kunde] + S_EXPORT * eps
    return v, y
