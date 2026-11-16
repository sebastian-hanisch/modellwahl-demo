"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in SETTING_SPECS: der Zufalls-Seed in der Seitenleiste, die Regler eines Experiments im Hauptbereich beim Experiment. Jedes Preset setzt alle Regler
(`mwl_constants.PRESETS`); das aktive Preset ist hervorgehoben, solange jeder Regler seinem Wert entspricht. Kein Regler wird ausgeblendet (kein KEPT-Muster nötig)."""

import random
from dataclasses import dataclass
from typing import Callable

import streamlit as st

import mwl_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _near_int(options):
    """Ganzzahl, auf die nächste erlaubte Stufe eingerastet."""
    def cast(value):
        value = int(float(value))
        return min(options, key=lambda o: abs(o - value))
    return cast


def _bool(value):
    if str(value).lower() in ("1", "true", "ja"):
        return True
    if str(value).lower() in ("0", "false", "nein"):
        return False
    raise ValueError(value)


def _spec(param, caster, rng, default, step=None):
    return SettingSpec(param, caster, default, rng[0], rng[1], step)


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "deg1_slider": _spec("deg1", int, C.DEG1_RANGE, C.DEG1_DEFAULT, 1),
    "vz_toggle": SettingSpec("vz", _bool, True),
    "n1_slider": _spec("n1", int, C.N1_RANGE, C.N1_DEFAULT, 1),
    "sigma1_slider": _spec("s1", float, C.SIGMA1_RANGE, C.SIGMA1_DEFAULT, 0.5),
    "sigma2_slider": _spec("s2", float, C.SIGMA2_RANGE, C.SIGMA2_DEFAULT, 0.5),
    "n2_slider": _spec("n2", int, C.N2_RANGE, C.N2_DEFAULT, 10),
    "reps2_slider": _spec("r2", int, C.REPS2_RANGE, C.REPS2_DEFAULT, 100),
    "n3_slider": _spec("n3", int, C.N3_RANGE, C.N3_DEFAULT, 10),
    "sigma3_slider": _spec("s3", float, C.SIGMA3_RANGE, C.SIGMA3_DEFAULT, 0.5),
    "beta3_slider": _spec("b3", float, C.BETA3_RANGE, C.BETA3_DEFAULT, 0.5),
    "n4_select": SettingSpec("n4", _near_int(C.N4_OPTIONS), C.N4_DEFAULT, min(C.N4_OPTIONS), max(C.N4_OPTIONS)),
    "reps4_slider": _spec("r4", int, C.REPS4_RANGE, C.REPS4_DEFAULT, 100),
    "krit4_radio": SettingSpec("k4", _choice(C.CRITERIA_POLY), C.KRIT4_DEFAULT),
    "n5_select": SettingSpec("n5", _near_int(C.N5_OPTIONS), C.N5_DEFAULT, min(C.N5_OPTIONS), max(C.N5_OPTIONS)),
    "noise5_slider": _spec("z5", int, C.NOISE5_RANGE, C.NOISE5_DEFAULT, 1),
    "krit5_radio": SettingSpec("k5", _choice(C.CRITERIA_FEAT), C.KRIT5_DEFAULT),
    "reps5_slider": _spec("r5", int, C.REPS5_RANGE, C.REPS5_DEFAULT, 100),
    "n6_select": SettingSpec("n6", _near_int(C.N6_OPTIONS), C.N6_DEFAULT, min(C.N6_OPTIONS), max(C.N6_OPTIONS)),
    "theta6_slider": _spec("t6", float, C.THETA6_RANGE, C.THETA6_DEFAULT, 0.5),
    "reps6_slider": _spec("r6", int, C.REPS6_RANGE, C.REPS6_DEFAULT, 100),
}
# Preset-Schlüssel (Settings-Feld) -> Zustandsschlüssel des Reglers
PRESET_KEYS = {"deg1": "deg1_slider", "vz": "vz_toggle", "n1": "n1_slider", "sigma1": "sigma1_slider", "sigma2": "sigma2_slider", "n2": "n2_slider",
               "reps2": "reps2_slider", "n3": "n3_slider", "sigma3": "sigma3_slider", "beta3": "beta3_slider", "n4": "n4_select", "reps4": "reps4_slider",
               "krit4": "krit4_radio", "n5": "n5_select", "noise5": "noise5_slider", "krit5": "krit5_radio", "reps5": "reps5_slider", "n6": "n6_select",
               "theta6": "theta6_slider", "reps6": "reps6_slider"}
SETTINGS_FIELDS = {"seed": "seed_input", **PRESET_KEYS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def step_of(state_key):
    return SETTING_SPECS[state_key].step


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value)) if isinstance(value, bool) else str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt die Regler des Presets und den Zufalls-Seed auf seinen Standard (die Zahlen der README gelten für den Standard-Zufalls-Seed)."""
    st.session_state["seed_input"] = C.DEFAULT_SEED
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag meint den Standardwert; der Zufalls-Seed gehört keinem Preset und gilt als Standard)
    mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(_gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
               for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt die Regler der Experimente):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
                  type="primary" if aktiv == name else "secondary")


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
