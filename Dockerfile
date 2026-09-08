FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY data/results/ data/results/

ENV PORT=8080
EXPOSE 8080

CMD ["uv", "run", "--no-sync", "python", "src/leaderboard_app.py"]
