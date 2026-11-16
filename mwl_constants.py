"""Konstanten: Regler-Grenzen, Vorgaben der Experimente, Farben, Presets."""

# --- Zufalls-Seed ---------------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- Modell ---------------------------------------------------------------------------------------------------------------------------
DEG_MAX = 8                                   # höchster Kandidat bei den Polynomen
TRUE_DEGREE = 3                               # Grad des erfundenen Erzeugers
POLY_LABEL = "y = 0,5 + x − 2x² + 3x³ + σ·ε,  x ~ Gleich(−1, 1)"
CRITERIA_POLY = ("AIC", "AICc", "BIC", "CV5", "LOO")
CRITERIA_FEAT = ("AIC", "AICc", "BIC", "LOO")
RULES = ("AIC", "BIC")
GRID_POINTS_POLY, GRID_POINTS_EXPORT = 201, 361
FOLDS = 5
K_NOISE_MAX = 5

# --- Regler: Bereich (min, max), Standardwert, Schrittweite --------------------------------------------------------------------------
DEG1_RANGE, DEG1_DEFAULT = (1, 7), 2
N1_RANGE, N1_DEFAULT = (12, 60), 20
SIGMA1_RANGE, SIGMA1_DEFAULT = (0.5, 3.0), 1.0
SIGMA2_RANGE, SIGMA2_DEFAULT = (1.0, 3.0), 2.0
N2_RANGE, N2_DEFAULT = (30, 300), 100
REPS2_RANGE, REPS2_DEFAULT = (100, 1000), 500
N3_RANGE, N3_DEFAULT = (30, 1000), 300
SIGMA3_RANGE, SIGMA3_DEFAULT = (1.0, 3.0), 3.0
BETA3_RANGE, BETA3_DEFAULT = (0.0, 3.0), 3.0
N4_OPTIONS, N4_DEFAULT = (30, 100, 300, 1000, 3000), 100
REPS4_RANGE, REPS4_DEFAULT = (100, 500), 200
KRIT4_DEFAULT = "AIC"
N5_OPTIONS, N5_DEFAULT = (50, 100, 300, 1000), 100
NOISE5_RANGE, NOISE5_DEFAULT = (0, 5), 5
KRIT5_DEFAULT = "AIC"
REPS5_RANGE, REPS5_DEFAULT = (100, 1000), 300
N6_OPTIONS, N6_DEFAULT = (50, 100, 200, 300), 100
THETA6_RANGE, THETA6_DEFAULT = (0.0, 5.0), 1.0
REPS6_RANGE, REPS6_DEFAULT = (200, 1000), 500

# Rechenbudget der Live-Rechnungen (Beobachtungen insgesamt je Experiment; Hunderte Wiederholungen bei n = 3000 wären zu langsam)
BUDGET_EXPORT = 600_000        # Wiederholungen × n beim Export (Experiment 4)
BUDGET_LOO_FEAT = 60_000       # Wiederholungen × n für Leave-one-out bei 128 Teilmengen (Experiment 5)
BUDGET_FEAT = 300_000          # Wiederholungen × n für Experiment 5 insgesamt

# --- Farben ---------------------------------------------------------------------------------------------------------------------------
COLOR = {"AIC": "#c77700", "AICc": "#a8508a", "BIC": "#2a6fb0", "CV5": "#2e7d4f", "LOO": "#7a3fb0"}
COLOR_TRUE = "#5b6470"
COLOR_DATA = "#2F6B65"
COLOR_BAND = "rgba(76,120,168,0.18)"

# --- Presets (Schnellstart): jedes setzt alle Regler; „Von Hand“ ist die Voreinstellung -----------------------------------------------
_BASE = dict(deg1=DEG1_DEFAULT, vz=True, n1=N1_DEFAULT, sigma1=SIGMA1_DEFAULT, sigma2=SIGMA2_DEFAULT, n2=N2_DEFAULT, reps2=REPS2_DEFAULT,
             n3=N3_DEFAULT, sigma3=SIGMA3_DEFAULT, beta3=BETA3_DEFAULT, n4=N4_DEFAULT, reps4=REPS4_DEFAULT, krit4=KRIT4_DEFAULT,
             n5=N5_DEFAULT, noise5=NOISE5_DEFAULT, krit5=KRIT5_DEFAULT, reps5=REPS5_DEFAULT, n6=N6_DEFAULT, theta6=THETA6_DEFAULT, reps6=REPS6_DEFAULT)
PRESETS = {
    "Von Hand (n = 8)": dict(_BASE),
    "Viele Daten, wenig Rauschen": dict(_BASE, sigma2=1.0, n2=300),
    "Schwaches Signal": dict(_BASE, sigma2=3.0, n2=300),
    "Wenig Daten": dict(_BASE, sigma2=3.0, n2=30),
    "Kein wahrer Grad (Export)": dict(_BASE, n4=3000),
    "Rauschmerkmale (Import)": dict(_BASE, n5=1000),
}
PRESET_HELP = {
    "Von Hand (n = 8)": "Voreinstellung. Das Handbeispiel aus acht Punkten in Experiment 1: vier Kriterien, drei Grade, jede Zahl in Brüchen nachgerechnet.",
    "Viele Daten, wenig Rauschen": "Experiment 2 mit σ = 1 und n = 300: BIC 98,2 %, AIC 72,3 % (wahrer Grad gewählt, 1000 Wiederholungen).",
    "Schwaches Signal": "Experiment 2 mit σ = 3 und n = 300: BIC 56,9 % (41,8 % zu klein), AIC 63,4 % (wahrer Grad gewählt, 1000 Wiederholungen).",
    "Wenig Daten": "Experiment 2 mit σ = 3 und n = 30: jedes Kriterium wählt den wahren Grad in 9 bis 16 % der Wiederholungen (1000 Wiederholungen).",
    "Kein wahrer Grad (Export)": "Experiment 4 mit n = 3000: die Kurve der Export-Container ist kein Polynom, der gewählte Grad wächst mit n (AIC im Median Grad 4, 200 Wiederholungen).",
    "Rauschmerkmale (Import)": "Experiment 5 mit n = 1000: BIC wählt genau {Zoll, Wochenende} in 96,3 %, AIC in 43,5 % (1000 Wiederholungen; die App rechnet bei n = 1000 höchstens 300).",
}
