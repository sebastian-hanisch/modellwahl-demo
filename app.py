"""Welches Modell? AIC, BIC, Kreuzvalidierung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Misst die Wahl des Modells selbst: wie oft ein Kriterium das erzeugende Modell findet, was die Wahl für die Vorhersage kostet und was nach der Wahl für Koeffizienten und
Intervalle gilt. Vehikel: ein erfundenes Polynom vom Grad 3 und zwei Ausschnitte des Container-Verweilzeit-Vehikels (Import, Export). Aufbau nach DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import mwl_constants as C
import mwl_evaluation as ev
import mwl_exact as E
import mwl_presets as P
from mwl_evaluation import de, de_int, de_sig, pct
from mwl_layout import dunkles_design, kopf_css
from mwl_visualization import (
    build_bias2,
    build_coverage,
    build_degree_distribution,
    build_feature_shares,
    build_fit,
    build_overfit,
    build_shares,
    build_signal,
)

st.set_page_config(page_title="Welches Modell? AIC, BIC, Kreuzvalidierung – Sebastian Hanisch", layout="wide")

THETAS = np.linspace(0.0, 5.0, 11)


@st.cache_data(show_spinner=False)
def _hand(deg_max, vz):
    return ev.exact_rows(deg_max, vz)


@st.cache_data(show_spinner=False)
def _drawn(n, sigma, deg_max, vz, seed):
    return ev.drawn_rows(n, sigma, deg_max, vz, seed)


@st.cache_data(show_spinner="Rechne die Wiederholungen …")
def _freq(n, sigma, reps, seed):
    return ev.selection_frequency(n, sigma, reps, seed)


@st.cache_data(show_spinner=False)
def _signal(n, sigma, beta3):
    return ev.signal_curve(n, sigma, beta3)


@st.cache_data(show_spinner="Rechne die Wiederholungen …")
def _export(n, reps, seed):
    return ev.export_selection(n, reps, seed)


@st.cache_data(show_spinner="Rechne alle Teilmengen …")
def _features(n, k_noise, reps, seed):
    return ev.feature_selection(n, k_noise, reps, seed)


@st.cache_data(show_spinner="Rechne die Wiederholungen …")
def _after(n, reps, seed):
    return ev.after_selection(n, reps, seed)


@st.cache_data(show_spinner=False)
def _coverage(n):
    return ev.coverage_curve(n, THETAS)


@st.cache_data(show_spinner=False)
def _frozen():
    return ev.load_frozen()


def _show(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _table(df):
    st.dataframe(df, hide_index=True, width="stretch")


def _fmt(x, nd=2):
    """Zahl mit Dezimalkomma; −∞ (RSS = 0) und nicht definierte Werte (+∞, None) sichtbar."""
    if x is None or (isinstance(x, float) and np.isposinf(x)):
        return "—"
    if isinstance(x, float) and np.isneginf(x):
        return "−∞"
    return de(x, nd)


def settings_from_state() -> ev.Settings:
    """Alle Regler aus dem Session State (Standardwerte vor dem ersten Zeichnen), damit die Kernfragen oben schon die Ergebnisse der Experimente nennen können."""
    values = {}
    for field, state_key in P.SETTINGS_FIELDS.items():
        values[field] = st.session_state.get(state_key, P.SETTING_SPECS[state_key].default)
    return ev.Settings(**values)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mwl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Welches Modell? AIC, BIC, Kreuzvalidierung")
st.markdown(
    """
In den Nachbar-Demos steht das Modell vorher fest. Hier wird die **Wahl selbst** gemessen: Wie oft findet ein Kriterium (**AIC, AICc, BIC, Leave-one-out, 5-fache Kreuzvalidierung**) das Modell,
das die Daten erzeugt hat? Was kostet die Wahl für die Vorhersage? Und was gilt danach für Koeffizienten und Intervalle des gewählten Modells? Erst ein **Handbeispiel in Brüchen**, dann Wiederholungen
mit bekannter Wahrheit: ein erfundenes Polynom vom Grad 3 und zwei Ausschnitte des **Container-Verweilzeit-Vehikels**. Zähleinheit ist die **Wiederholung** (ein Datensatz).

Weiter unten: sechs Experimente, die **❓ Kernfragen**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Wahl nach Kriterium", expanded=True):
    st.markdown(
        """
1. **Kandidaten anpassen:** jedes Modell (Polynomgrad 0 bis 8, oder eine Teilmenge der Merkmale) wird mit kleinsten Quadraten angepasst. Die Rest-Quadratsumme $\\mathrm{RSS}$ sinkt mit jedem Parameter.
2. **Einen Wert je Kandidat:** AIC, AICc und BIC addieren zu $n\\ln(\\mathrm{RSS}/n)$ eine Strafe für die Parameter (2 je Parameter bei AIC, $\\ln n$ bei BIC). Leave-one-out und die 5-fache Kreuzvalidierung
   messen den Fehler an Daten, die bei der Anpassung weggelassen wurden.
3. **Der kleinste Wert gewinnt.** Bei geschachtelten Modellen mit Normalfehlern ist das genau ein F-Test mit fester Schwelle: das größere Modell gewinnt, wenn $F > (n-k)(e^{2/n}-1)$ (AIC),
   mit $\\ln n$ statt 2 bei BIC. Ein Rauschmerkmal wird damit mit fester Wahrscheinlichkeit aufgenommen.
4. **Was hier gemessen wird:** nicht, ob „das richtige Modell“ gewählt wurde, sondern der Anteil der Wiederholungen, in denen es gewählt wird, und was die Wahl danach kostet.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

P.load_permalink_settings()
P.init_session_state_defaults()

with st.sidebar:
    P.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *P.bounds("seed_input"), key="seed_input", step=1,
                           help="Alle Wiederholungen aller Experimente stammen aus diesem Zufalls-Seed (SplitMix64). Gleicher Zufalls-Seed, gleiche Zahlen.")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=P.randomize_seed)
    st.caption("Die Regler der Experimente stehen bei den Experimenten.")

cfg = settings_from_state()

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ein Datensatz, vier Kriterien")
st.markdown(
    "Erst ein Datensatz von Hand: **acht Punkte** ($x = -7, -5, \\dots, 7$; $y = 3, 0, 8, 3, 5, 6, 9, 6$). Die Rest-Quadratsummen und der Leave-one-out-Fehler (PRESS) stehen als Brüche, "
    "die Logarithmen der Kriterien in 50 Stellen. Das Beispiel ist erfunden und gehört nicht zum Verweilzeit-Vehikel."
)
c1, c2 = st.columns(2)
with c1:
    deg1 = st.slider("Höchster Grad der Kandidaten", *P.bounds("deg1_slider"), key="deg1_slider", step=1,
                     help="Kandidaten sind die Polynome vom Grad 0 bis zu diesem Grad (höchstens n − 1). Bei Grad n − 1 verläuft die Kurve durch alle Punkte (RSS = 0).")
with c2:
    vz = st.toggle("Varianz zählt als Parameter (K = k + 1)", key="vz_toggle",
                   help="Zählt die Varianz mit, verschiebt das AIC in allen Modellen um 2 (die Wahl ändert sich nicht), den Zuschlag von AICc aber nicht: AICc kann anders wählen.")
rows_hand, picks_hand = _hand(deg1, vz)
df_hand = pd.DataFrame([{
    "Grad": r["deg"], "RSS": str(r["rss"]) + f" = {de(r['RSS'], 2)}",
    "AIC": _fmt(r["AIC"]) + (" ✓" if picks_hand["AIC"] == r["deg"] else ""), "AICc": _fmt(r["AICc"]) + (" ✓" if picks_hand["AICc"] == r["deg"] else ""),
    "BIC": _fmt(r["BIC"]) + (" ✓" if picks_hand["BIC"] == r["deg"] else ""),
    "PRESS (Leave-one-out)": ("nicht definiert (Hebel 1)" if r["press"] is None else de(r["LOO"], 2)) + (" ✓" if picks_hand["LOO"] == r["deg"] else ""),
} for r in rows_hand])
_table(df_hand)
st.caption("✓ = Wahl des Kriteriums (kleinster Wert; Gleichstand: kleinster Grad). PRESS = Σ (Residuum / (1 − Hebel))², gleich der Summe der quadrierten Fehler beim Weglassen je eines Punktes.")
grid_hand = np.linspace(-8.0, 8.0, 161)
curves_hand = [np.polyval(np.array([float(c) for c in E.fit(E.HAND_X, E.HAND_Y, d)[0]])[::-1], grid_hand) for d in range(len(rows_hand))]
st.plotly_chart(build_fit(E.HAND_X, E.HAND_Y, grid_hand, curves_hand, picks_hand), width="stretch", key="chart_hand")
_show(ev.verdict_hand(picks_hand, min(deg1, 7), vz))

st.markdown("**Ein gezogener Datensatz** (Polynom vom Grad 3 mit Normalfehlern; neuer Zufalls-Seed in der Seitenleiste, andere Daten):")
c1, c2 = st.columns(2)
with c1:
    n1 = st.slider("Beobachtungen n", *P.bounds("n1_slider"), key="n1_slider", step=1)
with c2:
    sigma1 = st.slider("Streuung σ der Fehler", *P.bounds("sigma1_slider"), key="sigma1_slider", step=0.5)
xd, yd, rows_d, picks_d = _drawn(n1, sigma1, deg1, vz, cfg.seed)
df_d = pd.DataFrame([{
    "Grad": r["deg"], "RSS": de(r["RSS"], 2),
    **{k: _fmt(r[k]) + (" ✓" if picks_d[k] == r["deg"] else "") for k in ("AIC", "AICc", "BIC")},
    "5-fach-CV": _fmt(r["CV5"], 3) + (" ✓" if picks_d["CV5"] == r["deg"] else ""), "LOO (je Punkt)": _fmt(r["LOO"], 3) + (" ✓" if picks_d["LOO"] == r["deg"] else ""),
} for r in rows_d])
_table(df_d)
grid_d = np.linspace(-1.0, 1.0, 161)
st.plotly_chart(build_fit(xd, yd, grid_d, ev.monomial_curves(xd, yd, len(rows_d) - 1, grid_d), picks_d), width="stretch", key="chart_drawn")
st.caption("Der Erzeuger hat den Grad 3. Die Kurven der Grade 0 bis 8 sind grau, die gewählten Grade farbig.")

st.markdown("---")

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------

f2 = _freq(cfg.n2, cfg.sigma2, cfg.reps2, cfg.seed)
r5 = _features(cfg.n5, cfg.noise5, ev.feature_reps(cfg.n5, cfg.reps5), cfg.seed)
a6 = _after(cfg.n6, cfg.reps6, cfg.seed)
st.markdown("## ❓ Kernfragen")
st.markdown(
    f"1. **Wie oft findet ein Kriterium das erzeugende Modell?** Beim Polynom (σ = {de(cfg.sigma2, 1)}, n = {cfg.n2}, {cfg.reps2} Wiederholungen): BIC in {pct(f2.share['BIC'][1])}, AIC in {pct(f2.share['AIC'][1])} "
    f"(Experiment 2).\n"
    f"2. **Was kostet die Wahl für die Vorhersage?** Am Import (n = {cfg.n5}, {cfg.noise5} Rauschmerkmale): Prognosefehler {de_sig(r5.pe_mean['BIC'])} mit BIC gegen {de_sig(r5.pe_mean['AIC'])} mit AIC, "
    f"{de_sig(r5.pe_mean['wahre Menge'])} bei bekannter wahrer Menge (Experiment 5); beim Polynom ist es in manchen Zellen umgekehrt (Experiment 2).\n"
    f"3. **Was gilt nach der Wahl?** Unter den gewählten Rauschmerkmalen sind {pct(a6.noise_sig['AIC'], 0)} (AIC) und {pct(a6.noise_sig['BIC'], 0)} (BIC) „signifikant“, ohne Wahl {pct(a6.noise_sig[ev.PROCEDURES[0]], 0)} "
    f"(Experiment 6, n = {cfg.n6})."
)

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie oft findet ein Kriterium den wahren Grad?")
st.markdown(
    f"Erzeuger: {C.POLY_LABEL} (wahrer Grad 3), Kandidaten Grad 0 bis 8. Je Wiederholung ein neuer Datensatz; jedes Kriterium wählt einen Grad. Prognosefehler = mittlerer quadratischer Abstand der "
    "angepassten zur wahren Kurve auf 201 Punkten in $[-1, 1]$."
)
c1, c2, c3 = st.columns(3)
with c1:
    sigma2 = st.slider("Streuung σ", *P.bounds("sigma2_slider"), key="sigma2_slider", step=0.5)
with c2:
    n2 = st.slider("Beobachtungen n", *P.bounds("n2_slider"), key="n2_slider", step=10)
with c3:
    reps2 = st.slider("Wiederholungen", *P.bounds("reps2_slider"), key="reps2_slider", step=100)
f2 = _freq(n2, sigma2, reps2, cfg.seed)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_shares(f2), width="stretch", key="chart_shares")
with c2:
    st.plotly_chart(build_degree_distribution(f2.dist), width="stretch", key="chart_dist2")
df2 = pd.DataFrame([{"Kriterium": k, "wahrer Grad gewählt": pct(f2.share[k][1]), "zu klein": pct(f2.share[k][0]), "zu groß": pct(f2.share[k][2]),
                     "Prognosefehler Mittel": de_sig(f2.pe_mean[k]), "Prognosefehler Median": de_sig(f2.pe_median[k])} for k in C.CRITERIA_POLY]
                   + [{"Kriterium": "Grad 3 fest (Wahrheit bekannt)", "wahrer Grad gewählt": "—", "zu klein": "—", "zu groß": "—",
                       "Prognosefehler Mittel": de_sig(f2.pe_mean["Grad 3 fest"]), "Prognosefehler Median": de_sig(f2.pe_median["Grad 3 fest"])}])
_table(df2)
st.caption("Das Mittel des Prognosefehlers wird von wenigen extremen Werten bestimmt (zu hoher Grad bei kleinem n): der Median steht daneben.")
_show(ev.verdict_frequency(f2))

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Signal gegen Schwelle")
st.markdown(
    "Ob ein echter Term gefunden wird, entscheidet die **Nichtzentralität** $\\lambda = \\beta_3^2\\,\\lVert(I-H)x^3\\rVert^2/\\sigma^2$ gegen die Schwelle (2 bei AIC, $\\ln n$ bei BIC). "
    "Die Kurve gilt für den Paarvergleich Grad 2 gegen Grad 3 (nichtzentrales F); die Rauten sind die vier Zellen aus Experiment 2 (alle Kandidaten, vorgerechnet)."
)
c1, c2, c3 = st.columns(3)
with c1:
    n3 = st.slider("Beobachtungen n", *P.bounds("n3_slider"), key="n3_slider", step=10)
with c2:
    sigma3 = st.slider("Streuung σ", *P.bounds("sigma3_slider"), key="sigma3_slider", step=0.5)
with c3:
    beta3 = st.slider("Koeffizient β₃ des Kubikterms", *P.bounds("beta3_slider"), key="beta3_slider", step=0.5)
frozen = _frozen()
sig = _signal(n3, sigma3, beta3)
cells = [c for c in frozen["zellen"].values() if c["n"] == n3]
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_signal(sig, cells), width="stretch", key="chart_signal")
    st.caption("Rauten nur bei n = 30 und n = 300 (dort liegen die vorgerechneten Zellen); ihr λ ist der Mittelwert über die 1000 Datensätze der Zelle." if not cells else
               "Rauten: Anteil „Grad ≥ 3 gewählt“ unter allen Kandidaten 0 bis 8, 1000 Wiederholungen, vorgerechnet (`mwl_frozen.json`).")
with c2:
    st.plotly_chart(build_overfit(frozen), width="stretch", key="chart_overfit")
    st.caption(f"Vorgerechnet: {de_int(frozen['pool']['reps'])} Wiederholungen je n, Kandidaten Grad 3 bis 8 (der Erzeuger ist enthalten), ohne Daten aus Quadratsummen von χ²-Größen.")
_show(ev.verdict_signal(sig))
_show(ev.verdict_overfit(frozen))
with st.expander("Tabelle: Overfitting gegen n (vorgerechnet)"):
    tab = frozen["pool"]["n"]
    _table(pd.DataFrame([{"n": de_int(int(n)), **{k: pct(100.0 - tab[n][k][0], 2) for k in ("AIC", "AICc", "BIC")}} for n in sorted(tab, key=int)]))
    st.caption(f"Anteil „Grad > 3“ unter den Graden 3 bis 8. Grenzwert n → ∞ für AIC: {pct(frozen['grenzwert']['wert'], 2)} ({de_int(frozen['grenzwert']['reps'])} Wiederholungen der Irrfahrt).")

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wenn es keinen wahren Grad gibt (Export-Vorlauf)")
st.markdown(
    "Vehikel Container-Verweilzeit, Export: log-Verweilzeit gegen den Vorlauf (1 bis 10 Tage). Die wahre Kurve $\\log(0{,}6 + 0{,}85\\,v)$ ist **kein Polynom**; jeder höhere Grad nähert sie besser an. "
    "Das Kriterium kann also nur einen Grad wählen, der zur Datenmenge passt."
)
c1, c2, c3 = st.columns(3)
with c1:
    n4 = st.select_slider("Beobachtungen n", options=list(C.N4_OPTIONS), key="n4_select")
with c2:
    reps4 = st.slider("Wiederholungen", *P.bounds("reps4_slider"), key="reps4_slider", step=100)
with c3:
    krit4 = st.radio("Kriterium im Diagramm", list(C.CRITERIA_POLY), key="krit4_radio", horizontal=True)
e4 = _export(n4, ev.export_reps(n4, reps4), cfg.seed)
if e4.reps < reps4:
    st.caption(f"Bei n = {n4} werden {e4.reps} statt {reps4} Wiederholungen gerechnet (Rechenbudget der Live-Rechnung).")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_bias2(e4.bias2), width="stretch", key="chart_bias2")
with c2:
    st.plotly_chart(build_degree_distribution(e4.dist, names=[krit4]), width="stretch", key="chart_dist4")
_table(pd.DataFrame([{"Kriterium": k, "gewählter Grad (Median, Q25–Q75)": f"{de(e4.deg_median[k][1], 0)} ({de(e4.deg_median[k][0], 0)}–{de(e4.deg_median[k][2], 0)})",
                      "Grad = bester im Nachhinein": pct(e4.hit_best[k]), "Prognosefehler (Median)": de_sig(e4.pe_median[k]), "Verhältnis zum besten Grad (Median)": de(e4.ratio[k], 2)}
                     for k in C.CRITERIA_POLY]))
st.caption(f"Bester Grad im Nachhinein: Median {de(e4.best_median[1], 0)} (Q25–Q75: {de(e4.best_median[0], 0)}–{de(e4.best_median[2], 0)}), Prognosefehler im Median {de_sig(e4.pe_best_median)}. "
           "Der Kundeneffekt wirkt als Teil des Fehlers (die Kunden stehen nicht im Modell).")
_show(ev.verdict_export(e4))

st.markdown("---")

# --- Experiment 5 ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Welche Merkmale? (Import)")
st.markdown(
    "Vehikel Container-Verweilzeit, Import: log-Verweilzeit $= b_0 + 0{,}7\\,\\text{Zoll} + 0{,}3\\,\\text{Wochenende}$ plus Fehler (Kundeneffekt und Streuung 0,55; die Kunden stehen nicht im Modell). "
    "Dazu bis zu fünf **Rauschmerkmale ohne Wirkung** (Zusatz dieser Demo). Es werden **alle** Teilmengen angepasst (bis 128), der Achsenabschnitt ist immer dabei."
)
c1, c2, c3, c4 = st.columns(4)
with c1:
    n5 = st.select_slider("Beobachtungen n", options=list(C.N5_OPTIONS), key="n5_select")
with c2:
    noise5 = st.slider("Rauschmerkmale", *P.bounds("noise5_slider"), key="noise5_slider", step=1)
with c3:
    reps5 = st.slider("Wiederholungen", *P.bounds("reps5_slider"), key="reps5_slider", step=100)
with c4:
    krit5 = st.radio("Kriterium (Kennzahlen)", list(C.CRITERIA_FEAT), key="krit5_radio")
f5 = _features(n5, noise5, ev.feature_reps(n5, reps5), cfg.seed)
if f5.reps < reps5:
    st.caption(f"Bei n = {n5} werden {f5.reps} statt {reps5} Wiederholungen gerechnet (Rechenbudget der Live-Rechnung).")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{krit5}: genau {{Zoll, Wochenende}}", pct(f5.exact[krit5]))
m2.metric("Wochenende gewählt", pct(f5.we_in[krit5]))
m3.metric("Rauschmerkmale im Mittel", de(f5.noise_mean[krit5], 2))
m4.metric("Prognosefehler (Mittel)", de_sig(f5.pe_mean[krit5]))
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(build_feature_shares(f5), width="stretch", key="chart_feat")
with c2:
    _table(pd.DataFrame([{"Kriterium": k, "Prognosefehler Mittel": de_sig(f5.pe_mean[k]), "Median": de_sig(f5.pe_median[k])} for k in C.CRITERIA_FEAT]
                        + [{"Kriterium": "wahre Menge {Zoll, Wochenende}", "Prognosefehler Mittel": de_sig(f5.pe_mean["wahre Menge"]), "Median": de_sig(f5.pe_median["wahre Menge"])},
                           {"Kriterium": "alle Merkmale", "Prognosefehler Mittel": de_sig(f5.pe_mean["alle Merkmale"]), "Median": de_sig(f5.pe_median["alle Merkmale"])}]))
st.caption(f"Prognosefehler = mittlerer quadratischer Abstand der Vorhersage zur wahren Mittelfunktion, exakt über die Momentenmatrix der Merkmale. Leave-one-out ist teuer (128 Modelle × n Hebel) und "
           f"nutzt nur die ersten {f5.reps_loo} der {f5.reps} Wiederholungen (dieselben Datensätze). Erwartete Rauschmerkmale aus der Einzelstufe: AIC {de(f5.expected_noise['AIC'], 2)}, BIC {de(f5.expected_noise['BIC'], 2)}.")
_show(ev.verdict_features(f5))

st.markdown("---")

# --- Experiment 6 ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Was gilt nach der Wahl?")
st.markdown(
    "Nach der Wahl sieht das gewählte Modell aus wie ein fest vorgegebenes, und die üblichen Intervalle und p-Werte werden ausgegeben. Zwei Messungen: der **Einzelfall** (ein Kandidat, exakt gerechnet) und das "
    "**Import-Vehikel** (fünf Rauschmerkmale, alle Teilmengen). Gegenmittel im Vehikel: **Datenteilung** (Wahl auf einer zufälligen Hälfte, Anpassung und Intervall auf der anderen)."
)
c1, c2, c3 = st.columns(3)
with c1:
    n6 = st.select_slider("Beobachtungen n", options=list(C.N6_OPTIONS), key="n6_select")
with c2:
    theta6 = st.slider("Effektstärke θ im Einzelfall", *P.bounds("theta6_slider"), key="theta6_slider", step=0.5,
                       help="θ = Effekt / Standardfehler. θ = 0: reines Rauschen; θ ≥ 3: der Effekt wird meist aufgenommen (n = 100, θ = 3: AIC 94,4 %, BIC 80,1 %).")
with c3:
    reps6 = st.slider("Wiederholungen im Vehikel", *P.bounds("reps6_slider"), key="reps6_slider", step=100)
a6 = _after(n6, reps6, cfg.seed)
cov6 = _coverage(n6)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_coverage(n6, THETAS, cov6, theta6), width="stretch", key="chart_cov")
    st.caption("Einzelfall: ein Kandidat, orthogonal zum Achsenabschnitt, nur die Fälle mit Aufnahme zählen. Exakt gerechnet (Integral über die Verteilung von s²).")
with c2:
    _table(pd.DataFrame([{"Verfahren": p, "Wochenende gewählt": pct(a6.chosen_we[p]), "Überdeckung Wochenende (gegeben gewählt)": pct(a6.cover_we[p]),
                          "Rauschmerkmale gewählt (je Wdh.)": de(a6.noise_mean[p], 2), "davon „signifikant“": pct(a6.noise_sig[p]),
                          "mind. 1 „signifikant“": pct(a6.any_sig[p])} for p in ev.PROCEDURES]))
    st.caption("Naives 95-%-Intervall und p < 0,05 im gewählten Modell. Das Wochenende (wahr 0,3) ist ein echter Effekt, die Rauschmerkmale haben die Wirkung 0.")
_show(ev.verdict_after(a6, theta6, cov6, THETAS))

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Normalfehler, gleiche Streuung** (alle Zahlen gelten nur für das Gauß-Kriterium) | Wächst die Streuung mit dem Merkmal, gilt die Gauß-Rechnung nicht mehr; hier nicht gemessen. | [streuung-wls-demo](https://github.com/sebastian-hanisch/streuung-wls-demo) |
| **Unabhängige Beobachtungen** (die Kunden stehen nicht im Modell und wirken als Teil des Fehlers) | Mit Kundeneffekt im Modell ändert sich die Wahl der Zufallsstruktur; hier nicht gemessen. | [cluster-gemischt-demo](https://github.com/sebastian-hanisch/cluster-gemischt-demo) |
| **Das erzeugende Modell steht unter den Kandidaten** (Experimente 2, 3, 5, 6) | Im Export-Vorlauf gibt es keinen wahren Grad: der gewählte Grad wächst mit n (Experiment 4). | – |
| **Wahl nach Kriterium über alle Teilmengen** | Wahl über einen Pfad (Lasso) und ihre Stabilität sind etwas anderes. | [kollinearitaet-auswahl-demo](https://github.com/sebastian-hanisch/kollinearitaet-auswahl-demo) |
| **Lineares Modell** | Die Wahl der Link-Funktion in verallgemeinerten linearen Modellen ist ein eigener Fall. | [glm-familie-demo](https://github.com/sebastian-hanisch/glm-familie-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben: $n$ Beobachtungen $(x_i, y_i)$ und eine Menge von Kandidatenmodellen (Polynomgrade oder Teilmengen von Merkmalen); gesucht: ein Wert je Kandidat, dessen Minimum das Modell wählt.
Das Modell $y = X\beta + \varepsilon$ mit $\varepsilon \sim N(0, \sigma^2 I)$ wird mit kleinsten Quadraten angepasst.

| Symbol | Bedeutung |
|---|---|
| $n$ | Zahl der Beobachtungen |
| $k$ | Zahl der Koeffizienten eines Modells, Achsenabschnitt eingeschlossen (Grad $d$ hat $k = d + 1$) |
| $K$ | Zahl der Parameter mit Varianz, $K = k + 1$ (abschaltbar: $K = k$) |
| $\mathrm{RSS}$ | Rest-Quadratsumme des Modells |
| $e_i$, $h_i$ | Residuum und Hebel der Beobachtung $i$ |
| $\lambda$ | Nichtzentralität eines Terms, $\lambda = \beta^2\lVert(I-H)x\rVert^2/\sigma^2$ |
| $\theta$ | Effekt geteilt durch seinen Standardfehler |

**Kriterien** (kleiner ist besser).
"""
    )
    st.latex(r"\mathrm{AIC} = n\ln(\mathrm{RSS}/n) + 2K")
    st.latex(r"\mathrm{AICc} = \mathrm{AIC} + \frac{2K(K+1)}{n-K-1}")
    st.latex(r"\mathrm{BIC} = n\ln(\mathrm{RSS}/n) + K\ln n")
    st.latex(r"\mathrm{LOO} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{e_i}{1-h_i}\right)^2")
    st.markdown(
        r"""
Der Leave-one-out-Fehler ist gleich der Neuanpassung ohne den Punkt $i$ (Tabelle in Experiment 1). Die 5-fache Kreuzvalidierung summiert die quadrierten Fehler der fünf Prüffaltungen und teilt durch $n$; sie hängt an der
zufälligen Faltung.

**Konventionen, die die Wahl ändern können.** (1) Konstanten: $-2\,\mathrm{logLik} + 2K$ unterscheidet sich von der Kurzform um $n(\ln 2\pi + 1)$ in allen Modellen gleich, die Wahl nicht. (2) Zählung der Varianz: verschiebt AIC in allen
Modellen um 2, nicht aber AICc (der Zuschlag hängt an $K$). (3) BIC bestraft erst ab $n \ge 8$ stärker als AIC ($\ln 8 = 2{,}079 > 2 > \ln 7$). Konventionen anderer Software sind hier nicht geprüft.

**Das Kriterium ist eine F-Schwelle.** Ein Modell mit einem Koeffizienten mehr (größeres Modell mit $k$ Koeffizienten, $n-k$ Freiheitsgrade) wird gewählt genau dann, wenn
"""
    )
    st.latex(r"F > (n-k)\left(e^{2/n}-1\right)\quad(\mathrm{AIC}),\qquad F > (n-k)\left(e^{\ln n/n}-1\right)\quad(\mathrm{BIC}).")
    st.markdown(r"Ein reines Rauschmerkmal wird also mit der Wahrscheinlichkeit $P\big(F_{1,\,n-k} > \text{Schwelle}\big)$ aufgenommen; hat es die Nichtzentralität $\lambda$, gilt das nichtzentrale $F'_{1,\,n-k}(\lambda)$.")
    st.markdown(r"**Wahl unter den Graden 3 bis 8, wenn der Erzeuger Grad 3 hat.** Die Entscheidung hängt nur an $n$: $\mathrm{RSS}_8/\sigma^2 \sim \chi^2_{n-9}$, jeder niedrigere Grad addiert das Quadrat einer weiteren Normalkoordinate. Für AIC und $n \to \infty$ ist die Wahl eine Irrfahrt mit Schritten $\chi^2_1 - 2$ (fünf Stufen).")
    st.markdown(r"**Überdeckung gegeben die Aufnahme** (ein Kandidat, orthogonal codiert, $t$ mit $n-2$ Freiheitsgraden): das naive 95-%-Intervall wird nur in den Wiederholungen ausgewertet, in denen $|t|$ über der Schwelle liegt. Der Wert folgt aus einem Integral über die Verteilung von $s^2$.")
    st.markdown(
        r"""
**Literatur** (jeweils in dem Umfang gelesen, der in Klammern steht).
- Hastie, Tibshirani, Friedman: *The Elements of Statistical Learning*, 2. Aufl., Abschn. 7.7, S. 233 bis 235 (Wortlaut gelesen): BIC sei als Auswahlkriterium asymptotisch konsistent, AIC neige zu zu komplexen Modellen, BIC wähle in endlichen Stichproben oft zu einfache.
- Tibshirani 1996, *Regression shrinkage and selection via the lasso*, J. Roy. Statist. Soc. B 58, Abschn. 7.1 (Wortlaut gelesen, Scan): Kreuzvalidierung zur Auswahl könne inkonsistent sein, wenn die Prüfmenge nicht angemessen wächst.
- Berk, Brown, Buja, Zhang, Zhao 2013, *Valid post-selection inference*, Ann. Statist. 41 (Wortlaut gelesen) und Lee, Sun, Sun, Taylor 2016, Ann. Statist. 44 (nur Zusammenfassung gelesen): Intervalle nach datengetriebener Auswahl. Die dort vorgeschlagenen Verfahren sind hier nicht gerechnet.
- Bates, Hastie, Tibshirani (arXiv 2104.00673) und Varoquaux (arXiv 1706.07581), nur Zusammenfassungen gelesen: Streuung der Kreuzvalidierung bei kleinen Stichproben.
- Heinze, Wallisch, Dunkler 2018, Biometrical Journal 60 (nur Suchtreffer gelesen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Regression und GLM: wann die Gleichung trügt](https://sebastianhanisch.net/konzepte-regression.html)."
)

P.sync_query_params({key: st.session_state[key] for key in P.SETTING_SPECS})
