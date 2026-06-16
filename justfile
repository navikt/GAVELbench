# https://just.systems

# Hvis ingen kommando vis alle tilgjengelige oppskrifter
default:
    @just --list

# ── Pipeline ──────────────────────────────────────────────────────────────────

# Run the full pipeline: fetch → generate → evaluate → report
# Pass extra flags with: just pipeline --skip-bq-fetch, --skip-generation, --only-report, --n-per-overkategori N
pipeline *ARGS:
    uv run python src/main.py {{ARGS}}

# Skip BQ fetch, reuse existing data; run generate → evaluate → report
skip-fetch *ARGS:
    uv run python src/main.py --skip-bq-fetch {{ARGS}}

# Skip fetch and generation; re-evaluate existing generated answers and write report
eval *ARGS:
    uv run python src/main.py --skip-generation {{ARGS}}

# Only regenerate report artifacts from existing JSON results in data/results/
report:
    uv run python src/main.py --only-report

# ── Tests ─────────────────────────────────────────────────────────────────────

# Smoke-test API access to all models defined in models.yaml
test-api:
    uv run python tests/test_api_access.py

# ── Code quality ──────────────────────────────────────────────────────────────

# Klargjør prosjektet ved å installere `prek` og oppdatere avhengigheter fra malen
prepare:
    uv run --only-dev prek install
    uv lock --upgrade

# Fiks feil og formater kode med ruff
fix:
    uv run --only-dev ruff check --fix .
    uv run --only-dev ruff format .

# Sjekk at alt koden ser bra ut og er klar for å legges til i git
lint:
    uv run --only-dev prek run --all-files --color always

# ── Quarto ────────────────────────────────────────────────────────────────────

# Lag et preview med Quarto
preview:
    uv run --group quarto quarto preview .

# Bygg Quarto-prosjektet
render:
    uv run --group quarto quarto render .

# ── Misc ──────────────────────────────────────────────────────────────────────

# Sjekk etter sårbarheter i Python-avhengigheter
audit:
    uv run --all-groups --with pip-audit pip-audit --local

# Oppdater Python og pre-commit avhengigheter
update:
    uv lock --upgrade
    uv run prek auto-update
