# Brag Plan: 3ML — The Multi-Mission Maximum Likelihood framework

## What is this app?
3ML is a Python framework that lets you model one astrophysical source using data from
many different telescopes at once — and switch between Maximum Likelihood and Bayesian
inference by changing a single line of code, without rewriting your model or your data.

## The angle
A gamma-ray burst goes off. Four different instruments see it, each in its own band, each
with its own ad-hoc data format. Normally that means four incompatible pipelines. 3ML lets
you drop them all into **one `DataList`**, fit **one physical model**, and then flip the
entire statistical engine — frequentist → Bayesian — with **one line**. The logo is literally
four colored puzzle pieces being assembled: that's the whole pitch. This is not a parody. The
tool is genuinely this good, and the video plays it completely straight, like a trailer.

## Hook (first 2-3 seconds)
Deep space, then a hard line of light: **"A gamma-ray burst goes off."** → **"Every telescope
sees it differently."** Four faint colored signals (the logo's red / blue / green / orange)
arrive at slightly different times. Real event on screen: **GRB 080916009** — the dataset
that actually ships in this repo.

## Key moments (the middle)
- Four instruments name-drop in the four logo colors and converge: **Fermi-GBM NaI**,
  **Fermi-GBM BGO**, **Fermi-LAT**, **HAWC** — "Four instruments. One source."
- Real code reveals: `DataList(nai3, bgo0, lat)` + `Powerlaw() + Blackbody()` +
  `Model(PointSource(...))`. One model, all the data. Each `DataList` argument tinted to its
  instrument color.
- **The one-line swap (the payoff):** `JointLikelihood(model, data).fit()` morphs to
  `BayesianAnalysis(model, data).sample()`. Same model. Same data. "Change one line."

## Outro / punchline
Pull back to the cosmos. The four puzzle pieces of the 3ML logo SNAP together on a strong
beat. **"3ML"** / **"The Multi-Mission Maximum Likelihood framework"** / tagline:
**"Multi-wavelength. Multi-messenger. One likelihood."** Small trust line: *Fermi-LAT · HAWC · POLAR*.

## User flow worth showing
The real analysis flow from `examples/Flux_Calculations.ipynb` (entry → key action → result):
1. **Entry:** combine multi-instrument GRB data into one `DataList` and define one `Model`.
2. **Key action:** fit it — `JointLikelihood(model, data).fit()`.
3. **Result / the flex:** swap that one line to `BayesianAnalysis(model, data).sample()` —
   identical model and data, different inference engine. This swap IS the centerpiece.

## Tone
- Preset: cinematic
- Creative direction: a multi-messenger astrophysics trailer for a real research instrument
- Interpretation: Big type, deep-space scale, dramatic but restrained reveals. The code is the
  hero, not decoration — hold it long enough to read. Confidence, never hype. Few, weighty SFX.

## Format: landscape — 1920x1080
## Duration: 24.5s (v2 — deeper: real 3ML plots + plugin architecture)

## v2 revision note
v1 was too surface-level. v2 shows the actual science: the plugin/energy-axis architecture
(optical→TeV, each instrument through its own official software → DataList), and FOUR faithful
reconstructions of real 3ML plots — a GBM light curve with background + source selection, a folded
count spectrum with a residual panel, a νFν SED (keV→GeV) with component decomposition and a green
credible band, and a posterior corner plot — using 3ML's real plotting palette
(#2CBDFE cyan, #F3A0F2 pink, #47DBCD teal, #F5B14C orange), inverted onto dark cinematic panels.
The frequentist↔Bayesian one-line swap stays as the climax, now tied to the corner-plot result.
Scenes: Hook → Energy-axis/plugins → Light curve → Count spectrum+residuals → νFν SED →
Swap+corner → Outro. Beat-locked to vol-12 strong cues (6.0, 8.74, 13.11, 17.47, 19.66, 22.93s).

## Visual identity (from the project)
- Background: deep-space near-black navy `#0a0e1a`
- Accent (four logo puzzle colors): red `#d8362a`, blue `#2a9be0`, green `#5bbf3a`, orange `#f5a623`
- Text: near-white `#f5f7fa`
- Display font: a clean geometric sans (Inter / system) at large scale
- Body/code font: monospace (JetBrains Mono / ui-monospace) for the real Python
- Strongest visual element: the four-color puzzle logo assembling, mirrored by four colored
  data signals converging into one fit

## Share copy (draft)
One gamma-ray burst. Four telescopes. One model. And you flip from Maximum Likelihood to full
Bayesian inference by changing a single line. That's 3ML.

## Audio direction
- Role: cinematic support — a low, steady bed that swells under the final logo assembly
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady, clean; cinematic pick)
- Music treatment: start at 0, volume ~0.30, gentle fade-in, swell into the outro, fade out under
  the logo over the last ~1.5s
- Music cue guidance: bundled preset read (vol-12, ~110 BPM). Strong cues at **8.74s** (code reveal),
  **13.11s** (the one-line swap — primary beat-lock), **17.47s / 18.56s** (logo snap). Beat grid
  available for the instrument-label convergence (~3–6s window).
- Audio-reactive treatment: subtle; use music RMS/bass to let the starfield depth and the assembled
  logo's glow breathe. No waveform/equalizer visuals.
- SFX posture: sparse, cinematic, motion-matched. Soft impacts for code reveals, light card/drop for
  the converging instrument labels, one resonant bell for the logo snap.
- Audio-coupled moments: instrument labels arriving one by one; code lines revealing; the
  JointLikelihood→BayesianAnalysis swap; the final puzzle snap.
- Restraint rule: no dense SFX stacks, no strobing, no pulsing that hurts code legibility. The code
  must always be readable.

## Storyboard

### Scene 1 — Hook: a burst goes off — 3s
Deep space. "A gamma-ray burst goes off." slams in (hold ~1.4s), then "Every telescope sees it
differently." Four faint colored signal streaks (red/blue/green/orange) sweep in from the edges at
staggered arrival times. Small label: GRB 080916009.
Sequential/interaction: yes — four colored signals arrive one by one across ~3–6s, staggered.
Audio intent: a low cinematic swell rising from silence; a soft impact as the first line lands.
Audio-coupled idea: soft impact on the hook line; faint drops as each signal arrives.
Music: low bed, just entering.
Transition mood: dramatic → Scene 2

### Scene 2 — Four instruments, one source — 3.5s
Four instrument names appear in their logo colors and slide toward center: Fermi-GBM NaI (blue),
Fermi-GBM BGO (green), Fermi-LAT (orange), HAWC (red). Line: "Four instruments. One source."
Sequential/interaction: yes — four labels reveal one per beat on the beat grid (~3.8–5.3s), each
held until the full set is on screen; they converge together at scene end.
Audio intent: each label gets a soft card/drop; momentum building toward the code.
Audio-coupled idea: card-place/drop per label, beat-grid aligned.
Music: bed building.
Transition mood: dramatic wipe → Scene 3

### Scene 3 — One model, all the data — 4s
A real syntax-highlighted code block reveals (lines appear quickly, then HOLD ~2s):
`data = DataList(nai3, bgo0, lat)` / `spectrum = Powerlaw() + Blackbody()` /
`model = Model(PointSource("GRB080916009", ra, dec, spectrum))`.
The three DataList arguments are tinted blue/green/orange to echo the instruments. Caption:
"One model. All the data."
Sequential/interaction: yes — 3 code lines reveal ~0.5s apart then hold the full block ~2s.
Audio intent: soft impact as the block lands; a code reveal anchored to the 8.74s strong cue.
Audio-coupled idea: impactSoft on block reveal; very light key tick per line (restrained).
Music: steady bed. // beat-locked: ~8.74s
Transition mood: clean → Scene 4

### Scene 4 — The one-line swap (payoff) — 4.5s
The fit line appears: `results = JointLikelihood(model, data).fit()` (hold ~1.5s). Then on the
strong beat, `JointLikelihood(...).fit()` morphs into `BayesianAnalysis(model, data).sample()`
— same indentation, same model/data, only the engine changes; the changed token glows.
Caption: "Frequentist or Bayesian — change one line."
Sequential/interaction: yes — the swap is the simulated "edit"; the before and after each hold
long enough to read.
Audio intent: a weighty, satisfying accent exactly on the swap.
Audio-coupled idea: impactBell or firm impact on the swap. // beat-locked: ~13.11s
Music: bed, lifting toward the outro.
Transition mood: dramatic → Scene 5

### Scene 5 — Logo assembly / outro — 4s
Pull back to the cosmos. The four colored puzzle pieces of the 3ML logo fly in and SNAP together
on a strong beat. "3ML" large, then "The Multi-Mission Maximum Likelihood framework", tagline
"Multi-wavelength. Multi-messenger. One likelihood.", small trust line "Fermi-LAT · HAWC · POLAR".
Assembled logo glow breathes subtly with the music.
Sequential/interaction: yes — four pieces converge and lock together as one beat-locked event.
Audio intent: music swells, then a resonant bell as the pieces lock; fade out under the hold.
Audio-coupled idea: impactBell_heavy on the snap. // beat-locked: ~17.47s (snap), logo glow audio-reactive
Music: swell then fade out over final ~1.5s.
Transition mood: final hold

**Music mood for this video:** cinematic
**Audio summary:** A low cinematic bed rises from silence, builds through the converging instruments
and the real code, lands a weighty accent on the one-line frequentist→Bayesian swap, then swells and
resolves with a single resonant bell as the four-piece logo snaps together and fades to a held title.
