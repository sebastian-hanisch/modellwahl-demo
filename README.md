# Welches Modell? AIC, BIC, Kreuzvalidierung – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-modellwahl-demo.streamlit.app/)** -->

Demo zu Regression und verallgemeinerten linearen Modellen aus dem Bereich „Konzepte“ der Website „Sebastian Hanisch – Operations Research und Machine Learning“. In den Nachbar-Demos zu diesem Thema steht das Modell vorher fest (welche Merkmale, welcher Grad, welche Strukturen). Hier wird die **Wahl selbst** gemessen: Wie oft findet ein Kriterium das Modell, das die Daten erzeugt hat, was kostet die Wahl für die Vorhersage, und was gilt danach für Koeffizienten und Intervalle des gewählten Modells? Die Kriterien sind AIC, AICc, BIC, Leave-one-out (LOO) und die 5-fache Kreuzvalidierung (CV5), alle selbst gerechnet. Vehikel: ein erfundenes kubisches Polynom (Experimente 1 bis 3) und zwei Ausschnitte des Container-Verweilzeit-Vehikels der Demo `verweilzeit-demo`: der Export-Vorlauf (Experiment 4, die Kurve ist kein Polynom) und der Import mit Rauschmerkmalen (Experimente 5 und 6). Zähleinheit überall: die **Wiederholung** (ein Datensatz mit n Beobachtungen), wenn nichts anderes dabeisteht. Nichts hier ist wissenschaftlich neu; neu ist die Vorführbarkeit mit Zähleinheiten und Gegenproben.

## Warum dieses Problem

Aus einer angepassten Gleichung lässt sich nur schließen, wenn man weiß, wie sie zustande kam. Wer unter mehreren Modellen nach einem Kriterium wählt, hat das Modell aus denselben Daten gewählt, mit denen er es danach beurteilt. Die Kernfragen: (1) Wie oft findet ein Kriterium das erzeugende Modell? (2) Was kostet die Wahl für die Vorhersage? (3) Was gilt nach der Wahl noch? Die Antworten gelten für diese Modelle, diese Parameter und diese Zähleinheiten; sie sind eigene Nachrechnung, keine Aussage über die Methodenklasse.

## Modell

Lineares Modell mit Normalfehlern, kleinste Quadrate. `RSS` ist die Rest-Quadratsumme, `n` die Zahl der Beobachtungen, `k` die Zahl der Koeffizienten (Achsenabschnitt eingeschlossen), `K = k + 1` zählt die Varianz mit:

`AIC = n ln(RSS/n) + 2K`, `AICc = AIC + 2K(K+1)/(n−K−1)`, `BIC = n ln(RSS/n) + K ln n`, `LOO = Σ (e_i/(1−h_i))² / n` (Hebel `h_i`, gleich der Neuanpassung ohne Punkt `i`), `CV5` = Summe der quadrierten Fehler der fünf Prüffaltungen geteilt durch `n`. Kleiner ist besser.

**Konventionen, die die Wahl ändern können:** (1) Konstanten: `−2·logLik + 2K` unterscheidet sich von der Kurzform um `n(ln 2π + 1)` in allen Modellen gleich, die Wahl nicht. (2) Zählung der Varianz verschiebt AIC in allen Modellen um 2, **aber nicht AICc** (der Zuschlag hängt an `K`): im Handbeispiel wählt AICc mit `K = k + 1` den Grad 0, mit `K = k` den Grad 1. (3) BIC bestraft pro Parameter erst ab n ≥ 8 stärker als AIC (`ln 8 = 2,079 > 2 > ln 7`). Die Konventionen anderer Software sind nicht geprüft und werden nicht genannt. Die Formeln stehen im Expander „📐 Mathematische Formulierung“ der App.

**Das Kriterium ist eine F-Schwelle:** ein Modell mit einem Koeffizienten mehr (größeres Modell mit `k` Koeffizienten, `n − k` Freiheitsgrade) wird genau dann gewählt, wenn `F > (n−k)(e^{2/n} − 1)` (AIC) beziehungsweise mit `ln n` statt 2 (BIC). Ein reines Rauschmerkmal wird damit mit der Wahrscheinlichkeit `P(F(1, n−k) > Schwelle)` aufgenommen, ein echter Term mit der Nichtzentralität `λ = β²·‖(I−H)x‖²/σ²` mit dem nichtzentralen F.

**Erzeuger.** Experimente 1 bis 3: `y = 0,5 + x − 2x² + 3x³ + σ·ε`, `x` gleichverteilt in [−1, 1], wahrer Grad 3, Kandidaten Grad 0 bis 8 (erfunden, kein Verweilzeit-Vehikel: eine Messung „wird der wahre Grad gefunden“ braucht einen Erzeuger mit bekanntem Grad). Experiment 4 (Export): log-Verweilzeit gegen den Vorlauf 1 bis 10, wahre Kurve `log(0,6 + 0,85 v)` plus Kundeneffekt, kein Polynom. Experimente 5 und 6 (Import): log-Verweilzeit `= b₀ + 0,7·Zoll + 0,3·Wochenende` plus Fehler, dazu bis zu fünf Rauschmerkmale ohne Wirkung (Zusatz dieser Demo; die Verweilzeit-Demo hat nur ein Rauschmerkmal). Alle Parameter und die zwölf Kundeneffekte stammen aus `vwz_constants.py` und `vwz_generator.py` der Verweilzeit-Demo (nur gelesen, als Zahlen eingetragen; `tests/test_generator.py` prüft sie gegen numpy). Die Kunden stehen **nicht** im Modell und wirken als Teil des Fehlers (Fehlervarianz 0,55² + 0,1319 = 0,4344, Streuung 0,659 im Import). Alles ist erfunden.

## Methodik

- **Experiment 1:** Handbeispiel mit acht Punkten in Brüchen (`fractions`) und 50-stelligen Logarithmen (`decimal`), danach ein gezogener Datensatz.
- **Experiment 2:** je Wiederholung ein Datensatz, jedes Kriterium wählt einen Grad 0 bis 8; Anteile „zu klein / wahr / zu groß“ und Prognosefehler (mittlerer quadratischer Abstand zur wahren Kurve auf 201 Punkten; Mittel und Median).
- **Experiment 3:** Kurve `P(Grad 3 gegen Grad 2 gewählt)` gegen `λ` aus dem nichtzentralen F; Überanpassung gegen `n` unter den Graden 3 bis 8 ohne Daten (Quadratsummen von χ²-Größen), **vorgerechnet** (`mwl_frozen.json`, Skript `tools/mwl_make_frozen.py`).
- **Experiment 4:** Export-Kurve, gewählter Grad, Prognosefehler gegen den im Nachhinein besten Grad, zwei zufällige 5-Faltungen desselben Datensatzes.
- **Experiment 5:** alle Teilmengen von Zoll, Wochenende und bis zu fünf Rauschmerkmalen (bis 128 Modelle), Anteil genau {Zoll, Wochenende}, Zahl aufgenommener Rauschmerkmale, Prognosefehler exakt über die Momentenmatrix der Merkmale.
- **Experiment 6:** Einzelfall (ein Kandidat) exakt als Integral über die Verteilung von `s²`; im Import-Vehikel die Überdeckung des naiven 95-%-Intervalls für das Wochenende, die „Signifikanz“ gewählter Rauschmerkmale und die Datenteilung (Wahl auf einer zufälligen Hälfte, Anpassung und Intervall auf der anderen).
- **Zufall:** nur ganzzahliger SplitMix64 (`mwl_rng.py`), Normalzahlen nach Box-Muller aus Grundrechenarten und Wurzel ohne C-Bibliothek; ein Zufalls-Seed in der Seitenleiste, je Verwendungszweck ein eigener Strom. Die ersten Wiederholungen hängen nicht von der Zahl der Wiederholungen ab.
- **Rechenbudget der App:** Experiment 4 rechnet höchstens 600 000 Beobachtungen (Wiederholungen × n; bei n = 3000 also 200 Wiederholungen), Experiment 5 höchstens 300 000, Leave-one-out dort nur die ersten Wiederholungen bis 60 000 Beobachtungen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Zufalls-Seed 35, Wiederholungszahl bei jeder Zeile genannt; Anteile aus Wiederholungen haben einen Standardfehler bis 1,6 Punkte (1000 Wiederholungen). Die Spalte „Test“ der Tabelle weiter unten nennt, wo die Zahl geprüft wird.

**Handbeispiel (Experiment 1; n = 8, x = −7, −5, …, 7, y = 3, 0, 8, 3, 5, 6, 9, 6; erfunden, kein Verweilzeit-Vehikel).** Rest-Quadratsummen 60, 1559/42, 767/21 für Grad 0, 1, 2; Leave-one-out (PRESS) 3840/49 = 78,37, 64,76 und 90,02. AIC 20,12 / 18,28 / 20,15, BIC 20,28 / 18,52 / 20,47, AICc 22,52 / 24,28 / 33,48. **AIC, BIC und LOO wählen Grad 1, AICc wählt Grad 0** (Abstand zum Zweitplatzierten jeweils über 1,5). Zählt die Varianz bei AICc nicht (`K = k`), wählt AICc ebenfalls Grad 1: die Konvention entscheidet. Grad 2 gegen Grad 1: `F = 125/1534 = 0,0815`, Schwellen 1,420 (AIC) und 1,484 (BIC): die Parabel wird nicht aufgenommen. Ein reines Rauschmerkmal wird bei n = 8 mit 28,7 % (AIC) beziehungsweise 27,7 % (BIC) aufgenommen (3 Koeffizienten im größeren Modell). Nimmt man Kandidaten bis Grad 6 hinzu, wählen AIC und BIC wegen der fast verschwindenden RSS den Grad 6 (Überanpassung bei n = 8), LOO bleibt bei Grad 1.

**Wie oft findet ein Kriterium den wahren Grad (Experiment 2; Polynom, Kandidaten 0 bis 8, 1000 Wiederholungen je Zelle):**

| σ, n | AIC | AICc | BIC | CV5 | LOO |
|---|---|---|---|---|---|
| 1, 30 | 47,4 % | 58,5 % | 55,4 % | 48,4 % | 52,0 % |
| 1, 300 | 72,3 % | 74,7 % | **98,2 %** | 69,0 % | 72,8 % |
| 3, 30 | 14,9 % | 12,8 % | 9,4 % | 15,2 % | 16,4 % |
| 3, 300 | 63,4 % | 65,1 % | 56,9 % | 58,4 % | 63,2 % |

Der wahre Grad wird also in 9,4 bis 98,2 % der Wiederholungen gewählt; bei σ = 3 und n = 30 von keinem Kriterium in mehr als 16,4 %. Bei σ = 3 und n = 300 wählt BIC in 41,8 % einen zu kleinen Grad (AIC: 10,3 %). Bei σ = 1 und n = 300 wählt AIC in 27,7 % einen zu großen Grad (BIC: 1,8 %).

**Was kostet die Wahl für die Vorhersage (Experiment 2, Mittel und Median des Prognosefehlers)?** Beim Polynom mit σ = 1, n = 300 ist BIC besser als AIC (Mittel 0,0148 gegen 0,0210, Median 0,0123 gegen 0,0158; Grad 3 fest: 0,0143 und 0,0120). **Beim Polynom mit σ = 3, n = 300 ist BIC schlechter** (Mittel 0,233 gegen 0,200, Median 0,217 gegen 0,156; Grad 3 fest: 0,128 und 0,108), weil BIC den echten Kubikterm in 41,8 % der Wiederholungen verwirft. Bei n = 30 und σ = 1 wird das Mittel von wenigen extremen Werten bestimmt (AIC: Mittel 9,51 gegen Median 0,262): darum steht der Median immer daneben.

**Signal gegen Schwelle (Experiment 3).** Ob ein echter Term gefunden wird, entscheidet `λ` gegen die Schwelle. Bei n = 300 wird Grad 3 gegen Grad 2 in 50 % der Fälle gewählt ab `λ` = 1,96 (AIC) und 5,67 (BIC), in 90 % ab 7,23 und 13,46. Die Näherung `λ ≈ n·β₃²·(4/175)/σ²` gibt für σ = 3, n = 300 den Wert 6,86; das mittlere `λ` über die 1000 Datensätze der Zelle ist 6,74 (4/175 ist die Varianz des Rests von x³ nach {1, x, x²} bei Gleichverteilung). Dort liegt der Paarvergleich bei 88,1 % (AIC) und 58,3 % (BIC). Die Zelle selbst (alle Kandidaten 0 bis 8, andere Zähleinheit) wählt Grad 3 oder höher in 89,7 % (AIC) und 58,2 % (BIC). **Überanpassung gegen n** (Wahl unter den Graden 3 bis 8, der Erzeuger ist enthalten, 1 Mio. Wiederholungen je Zeile, vorgerechnet): Anteil „Grad > 3“ bei AIC 39,44 / 29,72 / 27,54 / 26,83 / 26,56 / 26,54 % für n = 30 / 100 / 300 / 1000 / 10 000 / 100 000, bei AICc 15,96 / 23,38 / 25,49 / 26,21 / 26,50 / 26,53 %, bei BIC 15,17 / 4,48 / 2,01 / 0,94 / 0,25 / 0,07 %. Der Grenzwert n → ∞ ist bei AIC etwa 26,5 % (Irrfahrt mit fünf Schritten χ²₁ − 2, 2 Mio. Wiederholungen); eine Stufe allein ergäbe `P(χ²₁ > 2)` = 15,73 %. **AIC findet den wahren Grad also auch mit unendlich vielen Daten nicht sicher** (höchstens 73,5 % unter den Graden 3 bis 8); das gilt für diese Kandidatenmenge und Normalfehler, nicht für jede Kandidatenmenge (bei nur zwei Kandidaten wäre der Grenzwert 15,73 %). BIC erreicht in derselben Rechnung bei n = 100 000 einen zu großen Grad nur in 0,07 %, wählt aber bei schwachem Signal zu klein (siehe oben).

**Wenn es keinen wahren Grad gibt (Experiment 4; Export-Vorlauf).** Bester Näherungsfehler je Grad 0 bis 8 (n → ∞): 2,41·10⁻¹, 1,20·10⁻², 1,02·10⁻³, 1,09·10⁻⁴, 1,31·10⁻⁵, 1,71·10⁻⁶, 2,35·10⁻⁷, 3,35·10⁻⁸, 4,94·10⁻⁹; jeder höhere Grad passt besser, einen wahren Grad gibt es nicht. Der gewählte Grad wächst mit n: AIC wählt im Median Grad 2 / 3 / 3 / 3 / 4 bei n = 30 / 100 / 300 / 1000 / 3000, BIC Grad 2 / 2 / 2 / 3 / 3 (1000, 1000, 1000, 600 und 200 Wiederholungen). Der Prognosefehler der Wahl liegt im Median zwischen dem 1,2- und dem 1,6-Fachen des im Nachhinein besten Grades (alle fünf Kriterien, n = 30 bis 3000; der größte Wert ist BIC bei n = 3000 mit 1,58). Der gewählte Grad ist der im Nachhinein beste nur in etwa 10 bis 26 % der Wiederholungen. **Die Faltung zählt:** zwei zufällige 5-Faltungen desselben Datensatzes wählen verschiedene Grade in etwa 38 bis 43 % der Datensätze (n = 30 bis 3000).

**Welche Merkmale (Experiment 5; Import, fünf Rauschmerkmale, alle 128 Teilmengen, 1000 Wiederholungen je n):**

| n | Kriterium | genau {Zoll, Wochenende} | Wochenende drin | Rauschmerkmale im Mittel | Prognosefehler (Mittel) |
|---|---|---|---|---|---|
| 50 | AIC / BIC | 20,9 / 22,1 % | 54,2 / 33,2 % | 0,93 / 0,31 | 0,0683 / 0,0616 |
| 100 | AIC / BIC | 33,0 / 42,4 % | 74,7 / 48,6 % | 0,86 / 0,16 | 0,0308 / 0,0261 |
| 300 | AIC / BIC | 41,3 / 80,5 % | 98,3 / 86,6 % | 0,81 / 0,08 | 0,00892 / 0,00725 |
| 1000 | AIC / BIC | 43,5 / **96,3 %** | 100 / 100 % | 0,78 / 0,04 | 0,00246 / 0,00141 |

Zum Vergleich der Prognosefehler: wahre Menge {Zoll, Wochenende} 0,0271 / 0,0134 / 0,00444 / 0,00126, alle sieben Merkmale 0,0817 / 0,0384 / 0,0119 / 0,00341. **Am Import ist BIC bei jedem der vier n der bessere Vorhersager** (im Mittel; nur für diese vier Zellen). Die Zahl aufgenommener Rauschmerkmale folgt der Einzelstufe: erwartet aus `5·P(F(1, n−4) > Schwelle)` bei AIC 0,89 / 0,83 / 0,80 / 0,79, bei BIC 0,30 / 0,18 / 0,09 / 0,04 (Näherung: unabhängige Rauschspalten). AICc wählt genau {Zoll, Wochenende} um 0,4 bis 3,1 Punkte häufiger als AIC (23,8 / 36,1 / 44,1 / 43,9 %). Leave-one-out nutzt in der App nur die ersten Wiederholungen (Rechenbudget); sein Anteil genau {Zoll, Wochenende} ist 21,3 % bei n = 50 (1000 Wiederholungen), 32,5 % bei n = 100 (600), 46,5 % bei n = 300 (200) und 43,3 % bei n = 1000 (60).

**Was gilt nach der Wahl (Experiment 6).** *Einzelfall exakt* (n = 100, ein Kandidat, orthogonal codiert, θ = Effekt / Standardfehler): Überdeckung des naiven 95-%-Intervalls **gegeben die Aufnahme**: AIC bei θ = 0, 1, 2, 3, 4: 69,2 / 90,5 / 96,5 / 97,3 / 95,5 %; BIC: **0,0** / 80,1 / 94,4 / 96,9 / 97,0 %. Bei BIC und θ = 0 liegt die Schwelle `|t| > 2,15` über dem Kritischwert 1,98: das Intervall der aufgenommenen schließt die Null nie ein. Aufnahmewahrscheinlichkeit AIC 16,3 / 35,2 / 72,4 / 94,4 / 99,5 %, BIC 3,4 / 13,0 / 44,4 / 80,1 / 96,7 %. Die Verletzung sitzt bei **schwachen** Effekten (θ ≤ 1); bei θ ≥ 2 liegt die Überdeckung in allen sechs Werten zwischen 94,4 und 97,3 %, in fünf davon über 95 %. *Import-Vehikel* (2000 Wiederholungen, Wochenende wahr 0,3, fünf Rauschmerkmale):

| n | Verfahren | Wochenende gewählt | Überdeckung Wochenende | Rauschmerkmale gewählt | davon „signifikant“ (p < 0,05) | mind. 1 „signifikant“ |
|---|---|---|---|---|---|---|
| 100 | ohne Wahl (alle 7) | 100 % | 94,9 % | 5,00 | 5,1 % | 22,6 % |
| 100 | AIC | 73,5 % | 96,2 % | 0,86 | 32,0 % | 24,2 % |
| 100 | BIC | 47,0 % | 94,5 % | 0,17 | **100,0 %** | 15,9 % |
| 100 | AIC, Datenteilung | 52,6 % | 94,9 % | 0,92 | 4,9 % | 4,5 % |
| 100 | BIC, Datenteilung | 30,5 % | 95,9 % | 0,30 | 6,4 % | 1,9 % |
| 300 | ohne Wahl (alle 7) | 100 % | 95,7 % | 5,00 | 4,9 % | 21,6 % |
| 300 | AIC | 98,2 % | 97,0 % | 0,79 | 32,8 % | 22,6 % |
| 300 | BIC | 87,4 % | 97,2 % | 0,09 | **100,0 %** | 8,2 % |
| 300 | AIC, Datenteilung | 85,2 % | 95,6 % | 0,83 | 6,0 % | 4,8 % |
| 300 | BIC, Datenteilung | 58,4 % | 95,2 % | 0,13 | 4,2 % | 0,5 % |

**Negativbefund:** für das Wochenende (θ ≈ 2 bei n = 100, ≈ 3,5 bei n = 300) ist die Überdeckung nach der Wahl nicht verletzt: in allen zehn Zeilen zwischen 94,5 und 97,2 %. Verletzt ist die „Signifikanz“ der gewählten Rauschmerkmale: 32,0 und 32,8 % bei AIC, 100,0 % bei BIC (jedes gewählte Rauschmerkmal besteht den Test, weil die BIC-Schwelle über dem Kritischwert liegt) gegen etwa 5 % ohne Wahl. Die Datenteilung stellt die 5 % her (4,2 bis 6,4 %), **kostet aber Treffer**: das Wochenende wird bei n = 100 in 52,6 % (AIC) und 30,5 % (BIC) gewählt statt in 73,5 und 47,0 %.

### Zahl → Test

Jede Zahl dieses Abschnitts steht in `tests/test_claims.py`: Zufalls-Seed 35, dieselben Auswertungsfunktionen wie in der App, Neu-Rechnungen als Band (Anteile aus Wiederholungen: Rundung auf die genannte Stelle nach Neu-Rechnung, bei Plattformabweichung höchstens drei Zählungen; Floats relativ 1e-9), exakt nur, was in Brüchen oder `fractions`/`decimal` gerechnet ist.

| Zahl | Test | Art |
|---|---|---|
| Handbeispiel: RSS, PRESS, Kriterien, Wahlen, F, Schwellen, Aufnahme n = 8 | `test_b1_*` | exakt (Brüche, `decimal`) |
| Zelltabelle Experiment 2 und Prognosefehler | `test_b2_*` | Band, aus `mwl_frozen.json` und neu gerechnet |
| Kipp-λ, λ-Näherung, Paarformel | `test_b3_*` | Band (scipy) |
| Überanpassung gegen n, Grenzwert | `test_b3_*` | Band, vorgerechnet |
| Export: Näherungsfehler, gewählte Grade, Verhältnisse, Faltung | `test_b4_*` | Band, neu gerechnet |
| Import: Tabelle Experiment 5, Einzelstufe | `test_b5_*` | Band, neu gerechnet |
| Einzelfall und Import nach der Wahl | `test_b6_*` | Einzelfall exakt (Integral), Vehikel Band |
| Presets: Hilfetexte nennen gemessene Zahlen | `test_presets_help_*` | Band |

## Befunde und Korrekturen gegenüber der ersten Rechnung

Die erste Rechnung (frühere Messung mit numpy-Zufall, Zufalls-Seeds 2101 bis 3101) und diese Demo (eigener SplitMix64-Strom) stimmen in der Größenordnung überein, jede Zahl wurde neu gemessen. Abweichungen:

- **Prognosefehler im Export:** erste Rechnung 1,14- bis 1,67-fach, hier 1,2- bis 1,6-fach (die Zelle n = 3000 hat in der App nur 200 statt 300 Wiederholungen, Rechenbudget). Faltungsanteil dort 37 bis 45 %, hier 38 bis 43 %; Treffer des im Nachhinein besten Grades dort 12,3 bis 25,9 %, hier 10 bis 26 %. Der gewählte AIC-Grad bei n = 100 ist hier im Median 3 (dort 2).
- **Wahl bei σ = 3, n = 300:** BIC wählt den wahren Grad hier in 56,9 % (erste Rechnung 54,5 %), AIC in 63,4 % (62,8 %); innerhalb des Binomialfehlers.
- **„BIC schlechter für die Vorhersage“** gilt nur für das Polynom mit σ = 3 (n = 300); am Import ist BIC bei jedem gemessenen n besser. Die Aussage „BIC ist der bessere Vorhersager“ steht in der README nur für die vier Zellen von Experiment 5.
- **`scipy.stats.ncf` liefert bei `λ = 0` unbrauchbare Werte** (gemessen: −0,84 bei n = 300, eine negative Wahrscheinlichkeit). Ein Eigenschaftstest fand es (`tests/test_eigenschaften_modellwahl.py`); `mwl_theory.p_choose_bigger` weicht unter `λ < 1e-6` auf das zentrale F aus. Betroffen war nur der Anfang der Kurve in Experiment 3.
- **Singuläre Gram-Matrix bei der Datenteilung mit n = 50:** in einer Hälfte (25 Zeilen) fehlt manchmal jeder Zollhalt. Das brach beim ersten Bau die App (Seitenleisten-Test, Zufalls-Seed 35). Jetzt gilt in diesem Fall die Pseudoinverse; die leere Spalte erhält den Standardfehler unendlich (kein Test, keine Information, das Intervall überdeckt). Seit der Korrektur bricht nichts mehr (`tests/test_edge_cases.py`).
- **Kante zu `cluster-gemischt-demo`:** dort bleibt die Wahl der Zufallsstruktur ausdrücklich draußen, hier fehlen die Kunden im Modell. Die Kante ist geprüft, nichts wird doppelt gemessen.

## Ehrliche Grenzen

- **Normalfehler, gleiche Streuung, unabhängige Beobachtungen**; nur das Gauß-Kriterium. Keine Echtdaten, alles erfunden. Keine Kunden im Modell, keine Heteroskedastizität, keine verallgemeinerten linearen Modelle.
- **Das erzeugende Modell steht unter den Kandidaten** (Experimente 2, 3, 5, 6); nur Experiment 4 hat keinen wahren Grad. Die Rechnung ohne Daten (Experiment 3, Quadratsummen) gilt für geschachtelte Polynomgrade, die den Erzeuger enthalten; für Teilmengen anderer Struktur sind die Entscheidungen nicht entwurfsfrei (dort misst Experiment 5 direkt). Die Näherung „Produkt über Rauschmerkmale“ ist eine Näherung (an vier n geprüft, ±0,25).
- **Kreuzvalidierung:** nur die 5-fache; nicht gemessen sind die 10-fache, die wiederholte, die Ein-Standardfehler-Regel und die verschachtelte Kreuzvalidierung sowie die Konsistenz bei wachsender Prüffaltung (Tibshirani 1996, Abschn. 7.1, nennt die Möglichkeit der Inkonsistenz; hier nicht gemessen).
- **Datenteilung:** nur eine zufällige Hälfte gegen eine Hälfte; kein Mehrfach-Teilen.
- **Die Verfahren für Intervalle nach der Wahl** (simultane Inferenz über alle Untermodelle nach Berk u. a. 2013, bedingte Inferenz nach Lee u. a. 2016) sind **nicht gerechnet**, nur gelesen (Berk u. a. im Wortlaut, Lee u. a. nur die Zusammenfassung). Gemessen sind die Überdeckung gegeben die Aufnahme (exakt, ein Kandidat), die „Signifikanz“ gewählter Rauschmerkmale und die Datenteilung.
- **Zähler:** Zoll und Wochenende sind unabhängig gezogen, nicht aus der Ereignisfolge der Verweilzeit-Demo. Die realisierte Streuung der zwölf Kundeneffekte ist 0,363 (Nenner n) beziehungsweise 0,379 (Nenner n − 1).
- **Standardfehler der Anteile** bis 1,6 Punkte (1000 Wiederholungen); Zahlen aus anderen Zufalls-Seeds weichen in diesem Band ab. Beim Leave-one-out im Import (Rechenbudget) ist der Fehler größer (60 bis 1000 Wiederholungen).
- **Rechenzeit:** jede Eingabe in der App rechnet unter etwa zwei Sekunden (gemessen am Rechner des Bauers, nicht auf der Streamlit Cloud).
- **Quellenstufen:** Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2. Aufl., Abschn. 7.7, S. 233 bis 235 (Wortlaut gelesen): BIC sei als Auswahlkriterium asymptotisch konsistent, AIC neige zu zu komplexen Modellen, BIC wähle in endlichen Stichproben oft zu einfache. Tibshirani 1996, J. Roy. Statist. Soc. B 58, Abschn. 7.1 (Wortlaut gelesen, Scan): Kreuzvalidierung zur Auswahl könne inkonsistent sein. Berk, Brown, Buja, Zhang, Zhao 2013, Ann. Statist. 41 (Wortlaut gelesen) und Lee, Sun, Sun, Taylor 2016, Ann. Statist. 44 (nur Zusammenfassung): Intervalle nach datengetriebener Auswahl. Bates, Hastie, Tibshirani (arXiv 2104.00673) und Varoquaux (arXiv 1706.07581): nur Zusammenfassungen, Streuung der Kreuzvalidierung bei kleinen Stichproben. Heinze, Wallisch, Dunkler 2018, Biometrical Journal 60: nur Suchtreffer. Die Aussagen über AIC und BIC der Autoren Burnham/Anderson und Shao werden nicht als Beleg benutzt. Alle Zahlen dieser README sind eigene Nachrechnung in diesem Modell bei diesen Parametern.
- **Nachbar-Demos, deren Zahlen hier nicht wiederholt werden:** [arima-demo](https://github.com/sebastian-hanisch/arima-demo) und [exponential-smoothing-demo](https://github.com/sebastian-hanisch/exponential-smoothing-demo) wählen Modelle nach AICc in Prognosesettings ohne wahren Grad (die Wahrheit ist dort nicht gemessen); [naive-forecast-demo](https://github.com/sebastian-hanisch/naive-forecast-demo) misst, wie oft ein einzelner Testzeitraum den falschen Gewinner kürt (andere Zähleinheit: Ursprünge, hier zwei Faltungen); [multiple-testing-demo](https://github.com/sebastian-hanisch/multiple-testing-demo) behandelt den Gewinnerfluch bei mehreren Tests (verwandt: die Aufnahme eines Rauschmerkmals ist ein Selektionseffekt, hier ohne Mehrfachtest-Korrektur); [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo) und [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) wählen die Stärke einer Regularisierung ohne wahres Modell (Leave-one-out ist die lineare Verwandte der dortigen verallgemeinerten Kreuzvalidierung, hier nicht gemessen); [newsvendor-direkt-demo](https://github.com/sebastian-hanisch/newsvendor-direkt-demo) wählt die Stärke einer L1-Strafe per 5-facher Kreuzvalidierung (die Faltungsabhängigkeit der Gradwahl hier ist ein Befund anderer Art, keine Bestätigung der dortigen Deutung). Ähnlicher Name, anderes Gebiet: [modellwahl-netze-demo](https://github.com/sebastian-hanisch/modellwahl-netze-demo) vergleicht Netzfamilien, nicht lineare Modelle. Die Stabilität der Wahl über einen Lasso-Pfad liegt in [kollinearitaet-auswahl-demo](https://github.com/sebastian-hanisch/kollinearitaet-auswahl-demo), die Wahl der Link-Funktion in [glm-familie-demo](https://github.com/sebastian-hanisch/glm-familie-demo).

## Verifikation

- **Kern gegen Handrechnung:** das Handbeispiel in Brüchen (`tests/test_exact.py`): RSS, Hebel, PRESS, Kriterien, F; Hebelmuster und AICc-Zuschlag von Hand.
- **Orakel-Tests** (`tests/test_oracle_modellwahl.py`, anderer Rechenweg): Kriterium gegen F-Schwelle in 400 Zufallsinstanzen mit `decimal`; gestapelte Rechnung gegen Brüche (RSS, PRESS) und `lstsq`; Legendre-Basis gegen `numpy.polynomial.legendre`; Leave-one-out und 5-fache Kreuzvalidierung gegen scikit-learn (`LeaveOneOut`, `PredefinedSplit`); Aufnahme eines Rauschmerkmals gegen `scipy.stats.f`; Wahl unter den Graden 3 bis 8 auf drei Wegen (direkte Anpassungen, Koordinaten, χ²-Summen); Paarwahl gegen `scipy.stats.ncf`; Kipp-λ als Umkehrung; Irrfahrt-Grenzwert 26,53 % ±0,4; Überdeckung nach der Aufnahme gegen die Simulation mit Anpassung; Näherungsfehler der Export-Kurve gegen Monome; Prognosefehler der wahren Menge gegen `σ²·3/n`.
- **Eigenschaften** (`tests/test_eigenschaften_modellwahl.py`): Skalierung und Verschiebung von y, Umordnen, Basiswechsel, affine Änderung von x, Verdopplung aller Beobachtungen, AICc → AIC, BIC-Strafe ≥ AIC-Strafe genau ab n ≥ 8, BIC wählt nie ein größeres geschachteltes Modell als AIC (ab n ≥ 8), K-fache Kreuzvalidierung mit K = n gleich LOO, Faltungsabhängigkeit (nur als Zählung), disjunkte Hälften der Datenteilung, RSS fällt mit dem Grad, Stetigkeit des nichtzentralen F bei `λ → 0`.
- **Randfälle** (`tests/test_edge_cases.py`): n ≤ K + 1 (AICc undefiniert), RSS = 0, konstantes y, σ = 0, Hebel 1 (Anscombe-IV-Muster), n kleiner als die Faltungen, Grad über n − 1, leere Spalte in der Hälfte, Regler an den Rändern.
- **Erzeuger und Zufall** (`tests/test_generator.py`, `tests/test_rng.py`): Kundeneffekte gegen numpy, Momente, Präfix-Eigenschaft, Normalzahlen gleich der exakten Fassung auf 1e-13.
- **Vorgerechnet** (`tests/test_frozen.py`): vier Zellen voll neu gerechnet, Wahl ohne Daten und Grenzwert mit weniger Wiederholungen im statistischen Band; vollständiger Abgleich mit `python tools/mwl_make_frozen.py --check`.
- **Verdikte** (`tests/test_verdicts.py`): Schwellen und jede Zahl im Text an handgesetzten Werten; **Presets** (`tests/test_presets.py`); **Oberfläche** (`tests/test_app.py`, `tests/test_layout.py`): jeder Regler am Minimum und Maximum, jedes Schnellstart-Szenario mit der erwarteten Verdikt-Zeile, jede Änderung eines Reglers hebt die Hervorhebung auf; Footer-Test.
- **Plattform:** Neu-Rechnen-Tests als Band, nie bitgleich; Zufall ohne C-Bibliothek.

- **Testzahl:** 190 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: sechs Experimente, Kernfragen, Annahmen, Formeln |
| `mwl_constants.py` | Regler-Grenzen, Rechenbudget, Farben, Schnellstart-Szenarien |
| `mwl_rng.py` | SplitMix64 (Kopie von `pst_rng.py` der Demo plan-stoerung-demo), Normalzahlen ohne C-Bibliothek |
| `mwl_generator.py` | Erzeuger: Polynom, Import, Export (Parameter der Verweilzeit-Demo als Zahlen) |
| `mwl_selection.py` | AIC, AICc, BIC, LOO, K-fache Kreuzvalidierung, Teilmengen, gestapelt über Wiederholungen |
| `mwl_exact.py` | Handbeispiel in Brüchen und `decimal` |
| `mwl_theory.py` | F-Schwelle, Aufnahme, nichtzentrales F, Wahl ohne Daten, Überdeckung nach der Aufnahme |
| `mwl_evaluation.py` | Experimente, Kennzahlen, Verdikte (reine Funktionen) |
| `mwl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mwl_layout.py` | Fester Kopf (Stylesheet) |
| `mwl_presets.py` | Permalink, Schnellstart in der Seitenleiste, Hervorhebung, Zufalls-Seed-Button |
| `mwl_frozen.json` | Vorgerechnet: vier Zellen, Wahl ohne Daten für n = 30 bis 100 000, Irrfahrt-Grenzwert |
| `tools/mwl_make_frozen.py` | Rechnet `mwl_frozen.json` neu oder gleicht sie ab (`--check`) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, pandas, plotly |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur für die Gegenproben) |
| `tests/` | Handbeispiel, Orakel, Eigenschaften, Randfälle, Behauptungen, Vorgerechnetes, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **PDF-Export** der Ergebnisse (die Experimente sind Messreihen, kein Einzelfall zum Mitnehmen).
- **Echtdaten**, 10-fache und wiederholte Kreuzvalidierung, die Ein-Standardfehler-Regel, verschachtelte Kreuzvalidierung, Regularisierungsparameter (Verweis auf die Nachbar-Demos), simultane und bedingte Inferenz nach der Wahl (nur als Literaturverweis mit Quellenstufe).
- **Kunden im Modell** und die Wahl der Zufallsstruktur (Sache von `cluster-gemischt-demo`).
- **Wahl der Link-Funktion** und verallgemeinerte lineare Modelle (Sache von `glm-familie-demo`).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas; die Gegenproben benutzen scikit-learn.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
