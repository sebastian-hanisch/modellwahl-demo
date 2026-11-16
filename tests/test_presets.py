"""Presets (Schnellstart): Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink begrenzt und rastet ein, Zufalls-Seed gehört keinem Preset."""

import dataclasses

import mwl_constants as C
import mwl_evaluation as ev
import mwl_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 6
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_keys_are_exactly_the_settings_fields_without_the_seed():
    fields = {f.name for f in dataclasses.fields(ev.Settings)}
    assert set(P.PRESET_KEYS) == fields - {"seed"} and set(P.SETTINGS_FIELDS) == fields
    assert set(P.SETTING_SPECS) == set(P.SETTINGS_FIELDS.values())


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if isinstance(value, bool) or isinstance(value, str):
                continue
            assert spec.lo <= value <= spec.hi, (name, key)
        assert preset["n4"] in C.N4_OPTIONS and preset["n5"] in C.N5_OPTIONS and preset["n6"] in C.N6_OPTIONS
        assert preset["krit4"] in C.CRITERIA_POLY and preset["krit5"] in C.CRITERIA_FEAT


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Von Hand (n = 8)"]) == ev.Settings()


def test_presets_differ_from_each_other_in_at_least_one_regler():
    seen = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(seen)) == len(seen)


def test_each_non_default_preset_changes_only_the_regler_it_names():
    base = C.PRESETS["Von Hand (n = 8)"]
    changed = {name: {k for k in p if p[k] != base[k]} for name, p in C.PRESETS.items() if name != "Von Hand (n = 8)"}
    assert changed == {"Viele Daten, wenig Rauschen": {"sigma2", "n2"}, "Schwaches Signal": {"sigma2", "n2"}, "Wenig Daten": {"sigma2", "n2"},
                       "Kein wahrer Grad (Export)": {"n4"}, "Rauschmerkmale (Import)": {"n5"}}


def test_defaults_lie_inside_their_bounds():
    for key, spec in P.SETTING_SPECS.items():
        if isinstance(spec.default, (int, float)) and not isinstance(spec.default, bool):
            assert spec.lo <= spec.default <= spec.hi, key


def test_permalink_casters_snap_to_options_and_reject_garbage():
    assert P.SETTING_SPECS["n4_select"].caster("250") == 300 and P.SETTING_SPECS["n4_select"].caster("99999") == 3000
    assert P.SETTING_SPECS["krit4_radio"].caster("BIC") == "BIC"
    for bad in ("foo", "aic"):
        try:
            P.SETTING_SPECS["krit5_radio"].caster(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert P.SETTING_SPECS["vz_toggle"].caster("0") is False and P.SETTING_SPECS["vz_toggle"].caster("true") is True
