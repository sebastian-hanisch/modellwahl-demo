"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import mwl_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_fit(x, y, grid, curves, picks: dict, x_title="x"):
    """Datenpunkte, die Kurven aller Grade (dünn, grau) und die gewählten Grade (dick, farbig; Kriterien mit gleicher Wahl teilen eine Kurve)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="markers", marker=dict(size=9, color=C.COLOR_DATA), name="Daten", hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"))
    for d, c in enumerate(curves):
        fig.add_trace(go.Scatter(x=list(grid), y=list(c), mode="lines", line=dict(width=1, color="rgba(150,150,150,0.5)"), name=f"Grad {d}", showlegend=False,
                                 hovertemplate=f"Grad {d}: %{{y:.2f}}<extra></extra>"))
    by_deg = {}
    for name, d in picks.items():
        by_deg.setdefault(d, []).append(name)
    for d, names in sorted(by_deg.items()):
        fig.add_trace(go.Scatter(x=list(grid), y=list(curves[d]), mode="lines", line=dict(width=4, color=C.COLOR[names[0]]), name=f"Grad {d}: {', '.join(names)}",
                                 hovertemplate=f"Grad {d}: %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="y", range=[min(min(y) - 2, -1), max(max(y) + 2, 1)] if len(y) else None)
    return _base(fig, 380)


def build_shares(freq):
    """Anteil zu kleiner, wahrer und zu großer Grade je Kriterium (gestapelt, in % der Wiederholungen)."""
    names = list(C.CRITERIA_POLY)
    fig = go.Figure()
    for label, i, color in (("zu klein (Grad < 3)", 0, "rgba(76,120,168,0.85)"), ("wahr (Grad 3)", 1, C.COLOR_DATA), ("zu groß (Grad > 3)", 2, "rgba(199,119,0,0.85)")):
        fig.add_trace(go.Bar(x=names, y=[freq.share[k][i] for k in names], name=label, marker_color=color, text=[f"{freq.share[k][i]:.1f}".replace(".", ",") for k in names],
                             textposition="inside", hovertemplate="%{x}: %{y:.1f} %<extra>" + label + "</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Anteil der Wiederholungen (%)", range=[0, 100])
    return _base(fig)


def build_degree_distribution(dist: dict, title="Anteil der Wiederholungen (%)", names=None):
    """Verteilung der gewählten Grade 0 bis 8 je Kriterium (gruppierte Balken)."""
    names = names or list(dist)
    fig = go.Figure()
    for k in names:
        fig.add_trace(go.Bar(x=[str(d) for d in range(len(dist[k]))], y=list(dist[k]), name=k, marker_color=C.COLOR[k], hovertemplate="Grad %{x}: %{y:.1f} %<extra>" + k + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="gewählter Grad", type="category")
    fig.update_yaxes(title_text=title)
    return _base(fig)


def build_signal(sig, cells=None):
    """P(Grad 3 gegen Grad 2 gewählt) gegen die Nichtzentralität λ für AIC und BIC; senkrechte Linie bei λ der Regler; Punkte: Zellen aus Experiment 2 (alle Kandidaten)."""
    fig = go.Figure()
    for rule in C.RULES:
        fig.add_trace(go.Scatter(x=list(sig.lam_grid), y=list(sig.curve[rule]), mode="lines", line=dict(width=3, color=C.COLOR[rule]), name=f"{rule} (Formel)",
                                 hovertemplate="λ = %{x:.1f}: %{y:.1f} %<extra>" + rule + "</extra>"))
    fig.add_vline(x=sig.lam, line=dict(color=C.COLOR_TRUE, dash="dash", width=2), annotation_text=f"λ ≈ {sig.lam:.1f}".replace(".", ","), annotation_position="top right")
    for rule in C.RULES:
        fig.add_hline(y=50, line=dict(color="rgba(128,128,128,0.35)", dash="dot", width=1))
    if cells:
        for rule in C.RULES:
            pts = [(c["mean_lambda"], 100.0 - c["shares"][rule][0], c["sigma"]) for c in cells]
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", marker=dict(size=11, symbol="diamond", color=C.COLOR[rule], line=dict(width=1, color="white")),
                                     name=f"{rule}: Zelle aus Experiment 2", text=[f"σ = {p[2]:g}".replace(".", ",") for p in pts],
                                     hovertemplate="%{text}, mittleres λ = %{x:.1f}: Grad ≥ 3 in %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Nichtzentralität λ des Kubikterms", range=[0, 65])
    fig.update_yaxes(title_text="Grad 3 gewählt (%)", range=[0, 101])
    return _base(fig, 400)


def build_overfit(frozen: dict):
    """Anteil „Grad > 3“ unter den Graden 3 bis 8 gegen n (logarithmische Achse) für AIC, AICc, BIC und der Grenzwert der Irrfahrt für AIC."""
    tab = frozen["pool"]["n"]
    ns = sorted(int(k) for k in tab)
    fig = go.Figure()
    for rule in ("AIC", "AICc", "BIC"):
        fig.add_trace(go.Scatter(x=ns, y=[100.0 - tab[str(n)][rule][0] for n in ns], mode="lines+markers", line=dict(width=3, color=C.COLOR[rule]), name=rule,
                                 hovertemplate="n = %{x}: %{y:.2f} %<extra>" + rule + "</extra>"))
    fig.add_hline(y=frozen["grenzwert"]["wert"], line=dict(color=C.COLOR_TRUE, dash="dash", width=2), annotation_text="Grenzwert AIC", annotation_position="top left")
    fig.update_xaxes(title_text="Beobachtungen n", type="log")
    fig.update_yaxes(title_text="Grad > 3 gewählt (%)", range=[0, 45])
    return _base(fig)


def build_bias2(bias2):
    """Bester Näherungsfehler je Grad (n → ∞), logarithmische Achse."""
    fig = go.Figure(go.Bar(x=[str(d) for d in range(len(bias2))], y=list(bias2), marker_color=C.COLOR_TRUE, hovertemplate="Grad %{x}: %{y:.2e}<extra></extra>"))
    fig.update_xaxes(title_text="Grad", type="category")
    fig.update_yaxes(title_text="bester Näherungsfehler (Bias², log)", type="log")
    return _base(fig, 320)


def build_feature_shares(f):
    """Je Kriterium: genau {Zoll, Wochenende}, Wochenende gewählt, mindestens ein Rauschmerkmal gewählt (in % der Wiederholungen)."""
    names = list(C.CRITERIA_FEAT)
    fig = go.Figure()
    for label, src, color in (("genau {Zoll, Wochenende}", f.exact, C.COLOR_DATA), ("Wochenende gewählt", f.we_in, "rgba(76,120,168,0.85)"),
                              ("mindestens ein Rauschmerkmal", f.noise_any, "rgba(199,119,0,0.85)")):
        fig.add_trace(go.Bar(x=names, y=[src[k] for k in names], name=label, marker_color=color, hovertemplate="%{x}: %{y:.1f} %<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anteil der Wiederholungen (%)", range=[0, 100])
    return _base(fig)


def build_coverage(n: int, thetas, cov: dict, theta: float):
    """Überdeckung des naiven 95-%-Intervalls gegeben die Aufnahme gegen θ = Effekt / Standardfehler (exakt, ein Kandidat)."""
    fig = go.Figure()
    for rule in C.RULES:
        fig.add_trace(go.Scatter(x=list(thetas), y=list(cov[rule][0]), mode="lines+markers", line=dict(width=3, color=C.COLOR[rule]), name=rule,
                                 hovertemplate="θ = %{x:.1f}: %{y:.1f} %<extra>" + rule + "</extra>"))
    fig.add_hline(y=95, line=dict(color=C.COLOR_TRUE, dash="dash", width=2), annotation_text="95 % ohne Auswahl", annotation_position="bottom right")
    fig.add_vline(x=theta, line=dict(color="rgba(128,128,128,0.6)", dash="dot", width=2))
    fig.update_xaxes(title_text="θ = Effekt / Standardfehler", range=[min(thetas), max(thetas)])
    fig.update_yaxes(title_text="Überdeckung gegeben aufgenommen (%)", range=[0, 100])
    return _base(fig)
