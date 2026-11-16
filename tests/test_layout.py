"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt sicher, dass wirklich JEDER Regler (Seitenleiste
und Hauptbereich) dort steht: sonst bliebe das Preset trotz Änderung markiert. Allgemeine Einstellungen (Zufalls-Seed) stehen in der Seitenleiste, die Regler eines Experiments im
Hauptbereich beim Experiment."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mwl_constants as C
import mwl_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "select_slider", "number_input", "selectbox", "radio", "toggle", "checkbox", "text_input")
SIDEBAR_REGLER = {"seed_input"}


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for ort in (at.sidebar, at.main):
        for art in REGLER_ARTEN:
            try:
                return getattr(ort, art)(key=key)
            except KeyError:
                continue
    raise AssertionError(f"Regler {key} nicht gefunden")


def _andern(regler, spec):
    """Einen Wert wählen, der vom jetzigen verschieden ist."""
    if isinstance(regler.value, bool):
        regler.set_value(not regler.value)
    elif getattr(regler, "options", None):
        regler.set_value(next(o for o in regler.options if o != regler.value))
    else:
        regler.set_value(spec.hi if regler.value != spec.hi else spec.lo)


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_steht_in_setting_specs_und_ort_stimmt():
    at = _app()
    seite = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    haupt = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key}
    assert seite == SIDEBAR_REGLER                                              # allgemeine Einstellungen in der Seitenleiste
    assert haupt | seite == set(P.SETTING_SPECS)                                # kein Regler fehlt in SETTING_SPECS, keiner ist nur dort


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    _andern(_regler(at, state_key), P.SETTING_SPECS[state_key])
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    name = next(iter(C.PRESETS))                                                # erneutes Anklicken markiert wieder
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_ein_preset_setzt_die_regler_der_experimente():
    at = _app()
    at.sidebar.button(key="preset_Schwaches Signal").click().run()
    assert _regler(at, "sigma2_slider").value == 3.0 and _regler(at, "n2_slider").value == 300
    at.sidebar.button(key="preset_Kein wahrer Grad (Export)").click().run()
    assert _regler(at, "n4_select").value == 3000 and _regler(at, "sigma2_slider").value == C.SIGMA2_DEFAULT
    at.sidebar.button(key="preset_Rauschmerkmale (Import)").click().run()
    assert _regler(at, "n5_select").value == 1000
