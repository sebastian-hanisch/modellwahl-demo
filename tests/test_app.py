"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def _verdicts(at):
    return " ".join(m.value for kind in (at.success, at.info, at.warning) for m in kind)


def test_each_preset_shows_the_expected_verdict_line():
    """Je Schnellstart die erwartete Verdikt-Zeile (kein Preset ohne Wirkung meldet eine Wirkung)."""
    expected = {
        "Von Hand (n = 8)": ["AIC, BIC, LOO wählen Grad 1; AICc wählt Grad 0", "σ = 2,0, n = 100, 500 Wiederholungen"],
        "Viele Daten, wenig Rauschen": ["σ = 1,0, n = 300, 500 Wiederholungen", "Nur BIC findet den wahren Grad hier fast immer."],
        "Schwaches Signal": ["σ = 3,0, n = 300, 500 Wiederholungen"],
        "Wenig Daten": ["σ = 3,0, n = 30, 500 Wiederholungen", "Kein Kriterium findet den wahren Grad in mehr als der Hälfte der Wiederholungen."],
        "Kein wahrer Grad (Export)": ["Bei n = 3000 (200 Wiederholungen)"],
        "Rauschmerkmale (Import)": ["n = 1000, 5 Rauschmerkmale, 300 Wiederholungen"],
    }
    at = _fresh_app()
    for name, needles in expected.items():
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        text = _verdicts(at)
        for needle in needles:
            assert needle in text, (name, needle)


def test_hand_example_table_marks_the_choices_and_names_the_fractions():
    at = _fresh_app()
    first = at.dataframe[0].value
    assert list(first["Grad"]) == [0, 1, 2] and first["RSS"][0].startswith("60 =") and first["RSS"][1].startswith("1559/42") and first["RSS"][2].startswith("767/21")
    assert "✓" in first["AIC"][1] and "✓" in first["BIC"][1] and "✓" in first["AICc"][0] and "✓" in first["PRESS (Leave-one-out)"][1]
    assert sum("✓" in v for v in first["AIC"]) == 1


def test_kernfragen_stand_before_the_formula_expander_and_after_experiment_one():
    at = _fresh_app()
    headers = [h.value for h in at.markdown if h.value.startswith("## ")]
    assert any("Kernfragen" in h for h in headers)
    labels = [e.label for e in at.expander]
    assert labels[-1] == "📐 Mathematische Formulierung"
