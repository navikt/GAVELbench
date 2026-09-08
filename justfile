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

# Sample fresh unanswered questions and generate answers (no evaluate or report)
generate *ARGS:
    uv run python src/main.py --skip-eval {{ARGS}}

# Skip fetch and generation; re-evaluate existing generated answers and write report
evaluate *ARGS:
    uv run python src/main.py --skip-generation {{ARGS}}

# Only regenerate report artifacts from existing JSON results in data/results/
report:
    uv run python src/main.py --only-report

# Create aggregate and per-overcategory leaderboard CSV files
leaderboard *ARGS:
    uv run python src/create_leaderboard.py {{ARGS}}

# Start the standalone interactive Dash leaderboard
leaderboard-app:
    uv run python src/leaderboard_app.py

# ── Tests ─────────────────────────────────────────────────────────────────────

# Smoke-test API access to all models defined in models.yaml
test-api:
    uv run python tests/test_api_access.py

# Smoke-test GCS bucket connectivity defined in src/config.yml
test-bucket:
    uv run python tests/test_bucket_access.py

# Unit-test sampling / resampling of unanswered questions (no services needed)
test-sampling:
    uv run python tests/test_sampling.py

# Upload all current data artifacts to the GCS bucket
upload-data *ARGS:
    uv run python scripts/upload_data.py {{ARGS}}

# List which models are available in the Vertex AI project/location vs. models.yaml
list-models *ARGS:
    uv run python scripts/list_available_models.py {{ARGS}}


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
    uv run --group quarto quarto preview quarto

# Bygg Quarto-prosjektet
render:
    uv run --group quarto quarto render quarto

# ── Misc ──────────────────────────────────────────────────────────────────────

# Sjekk etter sårbarheter i Python-avhengigheter
audit:
    uv run --all-groups --with pip-audit pip-audit --local

# Oppdater Python og pre-commit avhengigheter
update:
    uv lock --upgrade
    uv run prek auto-update
