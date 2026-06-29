# Hyperframes Composition Brief: 3ML

## Objective
Create a short, cinematic launch-style brag video for **3ML — the Multi-Mission Maximum
Likelihood framework**: a Python tool that fits one astrophysical source with data from many
telescopes at once, and switches frequentist↔Bayesian inference with one line of code.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 19s

## Source Material
- Project root: `/Users/jburgess/coding/projects/threeML`
- Primary files read: `README.md`, `examples/Flux_Calculations.ipynb`,
  `examples/fermi_grb_full_demo.ipynb`, `logo/logo_sq.png`, `pyproject.toml`
- Product name: 3ML (The Multi-Mission Maximum Likelihood framework)
- Tagline / strongest claim: "Model one source with every instrument's data — and switch from
  Maximum Likelihood to Bayesian inference by changing one line."
- Key UI/visual moment to recreate: the **real Python API flow** (a syntax-highlighted code
  block) and the **four-color puzzle logo** assembling.
- Copy that must appear verbatim (the code is real, from the repo's examples):
  - `data = DataList(nai3, bgo0, lat)`
  - `spectrum = Powerlaw() + Blackbody()`
  - `model = Model(PointSource("GRB080916009", ra, dec, spectrum))`
  - `results = JointLikelihood(model, data).fit()`
  - `results = BayesianAnalysis(model, data).sample()`
  - Display lines: "A gamma-ray burst goes off." / "Every telescope sees it differently." /
    "Four instruments. One source." / "One model. All the data." /
    "Frequentist or Bayesian — change one line." /
    "Multi-wavelength. Multi-messenger. One likelihood."
  - Instrument labels: "Fermi-GBM NaI", "Fermi-GBM BGO", "Fermi-LAT", "HAWC"

## Creative Direction
- Tone preset: cinematic
- Creative direction: a multi-messenger astrophysics trailer for a real research instrument
- Interpretation: big type, deep-space scale, dramatic-but-restrained reveals. The real code is
  the hero — hold every code line long enough to read it. Confidence, not hype.
- Angle: A gamma-ray burst is seen by four instruments in four bands with four incompatible data
  formats. 3ML drops them all into one `DataList`, fits one physical model, then flips the entire
  statistical engine with one line. The four-piece puzzle logo IS the pitch.
- Hook: deep space → "A gamma-ray burst goes off." → "Every telescope sees it differently." with
  four faint colored signals arriving staggered.
- Outro / punchline: four puzzle pieces snap into the 3ML logo; "Multi-wavelength. Multi-messenger.
  One likelihood."
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals (no random particles standing in for nothing)
  - Unrelated visual redesign — stay in the deep-space + real-code world
  - Waveform/equalizer graphics

## Visual Identity
- Background: deep-space near-black navy `#0a0e1a` (subtle starfield / depth ok)
- Text: near-white `#f5f7fa`
- Accent (the four logo puzzle colors): blue `#2a9be0`, green `#5bbf3a`, orange `#f5a623`,
  red `#d8362a`. Map instruments to colors consistently (NaI=blue, BGO=green, LAT=orange, HAWC=red).
- Display font: clean geometric sans (Inter / system-ui), large scale
- Code font: monospace (JetBrains Mono / ui-monospace), syntax-highlighted Python
- Visual references from the project: the four-color puzzle logo (`logo/logo_sq.png`), the real
  Python API names, GRB 080916009 as the on-screen event.

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. Hook: a burst goes off — 3s — "A gamma-ray burst goes off." / "Every telescope sees it
   differently." + four staggered colored signals; label GRB 080916009.
2. Four instruments, one source — 3.5s — four colored instrument labels reveal one per beat and
   converge; "Four instruments. One source."
3. One model, all the data — 4s — real code block (DataList + Powerlaw()+Blackbody() + Model);
   args tinted to instrument colors; "One model. All the data."
4. The one-line swap (payoff) — 4.5s — `JointLikelihood(...).fit()` morphs to
   `BayesianAnalysis(...).sample()`; "Frequentist or Bayesian — change one line."
5. Logo assembly / outro — 4s — four puzzle pieces snap into 3ML logo; tagline + trust line.

## Audio
- Audio role: cinematic support — low steady bed that swells under the outro.
- Audio arc: rises from near-silence at the hook → builds through instruments + code → weighty
  accent on the one-line swap → swell + single bell on the logo snap → fade out under final hold.
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
- Music treatment: data-start 0, volume ~0.30, gentle fade-in; fade out over the last ~1.5s under
  the logo. Never above 0.4.
- Music cue guidance: bundled preset at
  `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`
  (~110 BPM). Strong cues: **8.74s** (code block reveal), **13.11s** (the swap — primary lock),
  **17.47s / 18.56s** (logo snap). Beat grid (3.82/4.39/4.91/5.34s) for the instrument labels.
  Use only 1–3 strong locks; readability of code wins over snapping.
- Audio-reactive treatment: subtle — drive starfield depth and the assembled logo's glow from
  RMS/bass. No waveform/equalizer/note graphics, no strobing.
- Audio-coupled moments:
  - Scene 1 — four signals arrive — faint drops, staggered
  - Scene 2 — four instrument labels — soft card/drop per label, beat-grid aligned
  - Scene 3 — code block reveal — one soft impact on the block (beat-lock ~8.74s); optional very
    light key tick per line
  - Scene 4 — the swap — one weighty accent exactly on the morph (beat-lock ~13.11s)
  - Scene 5 — logo snap — one resonant bell (beat-lock ~17.47s); music swell then fade
- SFX selection guidance (cinematic, restrained): `impact/impactSoft_medium_*` for code/reveals,
  `casino/card-place-*` or `interface/drop_*` for converging labels, `impact/impactBell_heavy_000`
  or `_004` for the logo snap and the swap accent. Keep SFX at 0.55–0.75. Few, weighty, motion-matched.
- SFX analysis guidance: read `~/.claude/skills/brag/assets/sfx/sfx-analysis.md`; prefer low/medium
  high-frequency-risk files for the repeated/polished code reveals.
- Exact SFX choice: Hyperframes chooses filenames, timestamps, density, and volume from the
  implemented animation.
- Audio files: copy the chosen music and any selected SFX into `brag-output/composition/assets/`.

## Hyperframes Instructions
Use the current `hyperframes` skill and CLI workflow. Prefer native Hyperframes conventions.

Requirements:
- Show at least one real element from the project: the real Python API code block and the puzzle logo.
- Keep all code and text readable in the final render — hold code lines to the reading-time floor.
- Keep the video within 15–25 seconds (target 19s).
- Include the planned music + restrained SFX layer.
- Treat audio notes as guidance; choose SFX after the animation exists.
- Lock 1–3 strong cues (≈8.74 / 13.11 / 17.47s) within ±0.15s; do not snap if it hurts code legibility.
- Wire at least one subtle audio-reactive element (starfield depth or logo glow) or document why not.
- Use local relative asset paths only.
- Run `npx hyperframes lint` and `validate` before render.
