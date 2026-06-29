# threeML (3ML) — The Multi-Mission Maximum Likelihood Framework

Scientific Python library for joint likelihood/Bayesian analysis across astrophysics
missions (Fermi, HAWC, HESS, ...). Pairs with the `astromodels` package for models.

## Environment

- **Use the `3mldev` virtualenv, not system Python:** `/Users/jburgess/.environs/3mldev/bin/python`
- Python `>=3.9`.

## Common commands

Run them with the venv interpreter/tools above (e.g. `/Users/jburgess/.environs/3mldev/bin/pytest`).

- **Tests:** `pytest threeML/test`
- **Fast tests only:** `pytest threeML/test -m "not slow"` (`slow` is the one custom marker)
- **Type check:** `pyright` (config in `pyrightconfig.json`; checks `threeML/`, excludes tests/data)
- **Format:** `black .` then `isort .` — line length **88**, isort uses `profile = "black"`
- **Lint:** `flake8` (max-line-length 88; config in `.flake8`)

Edited `.py` files are auto-formatted with black + isort by the PostToolUse hook in
`.claude/hooks/format-python.sh`, so manual formatting is usually unnecessary.

## Layout

- `threeML/` — package source (~200 modules). Tests live in `threeML/test/`.
- `threeML/plugins/` — per-instrument data plugins (e.g. `FermipyLike.py`, `XYFluxLike.py`).
- `docs/`, `examples/` — Sphinx docs (with nbsphinx notebooks) and example notebooks.
- `ci/`, `.github/workflows/` — GitHub Actions: build/test, conda build, docs, xspec.

## Don't touch

- `versioneer.py` and `threeML/_version.py` — versioning is managed by versioneer.
- `build/`, `dist/`, `*.egg-info/` — generated artifacts.
- Binary test fixtures: `glg_*.fit.gz`, `*.pha`, `*.rsp2`, `_analysis_set_test.*` — do not
  edit or overwrite; they are real instrument data used by the test suite.

## Conventions

- Match the surrounding code style; keep changes surgical and scoped to the request.
- Run the relevant tests before claiming a change works.
