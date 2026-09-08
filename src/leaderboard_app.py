"""Standalone Dash application for the interactive model leaderboard."""

import json
import os
from pathlib import Path

from dash import Dash, Input, Output, dash_table, dcc, html

from create_leaderboard import create_leaderboard_config, process_leaderboard_data
from models import filter_by_leaderboard_models

RESULTS_PATH = Path("data/results/evaluation_report.json")
CATEGORY_RESULTS_PATH = Path(
    "data/results/evaluation_report_scores_per_overkategori.json"
)

_COLUMNS = [
    ("model_id", "Modell"),
    ("composite_score", "Total score"),
    ("rouge_l", "ROUGE-L"),
    ("semantic_sim", "Semantisk likhet"),
    ("factual_corr", "Faktakorrekthet"),
    ("answer_quality", "Svarkvalitet"),
    ("jsd_divergence", "JSD (lavere er bedre)"),
    ("status", "Status"),
]

_COLUMN_STYLES = {
    "composite_score": "#fff3cd",
    "rouge_l": "#dbeeff",
    "semantic_sim": "#f2e0f7",
    "factual_corr": "#dff3e2",
    "answer_quality": "#d9f1ee",
    "jsd_divergence": "#ffe8cc",
}


def _score_records(models_data: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    """Scores and filters raw model metrics for display."""
    visible = filter_by_leaderboard_models(models_data, warn=False)
    leaderboard = process_leaderboard_data(visible, create_leaderboard_config())
    return leaderboard[[key for key, _ in _COLUMNS]].to_dict("records")


def load_leaderboard_datasets(
    results_path: Path = RESULTS_PATH,
    category_results_path: Path = CATEGORY_RESULTS_PATH,
) -> dict[str, list[dict[str, object]]]:
    """Loads aggregate and per-overcategory leaderboard rows from result JSON."""
    with results_path.open(encoding="utf-8") as f:
        datasets = {"Samlet resultat": _score_records(json.load(f)["models"])}
    with category_results_path.open(encoding="utf-8") as f:
        scores_by_model = json.load(f).get("scores", {})

    by_category: dict[str, dict[str, dict[str, float]]] = {}
    for model_id, category_scores in scores_by_model.items():
        for category, scores in category_scores.items():
            by_category.setdefault(category, {})[model_id] = scores
    datasets.update(
        {
            category: _score_records(scores)
            for category, scores in sorted(by_category.items())
        }
    )
    return datasets


def _table(rows: list[dict[str, object]]) -> dash_table.DataTable:
    """Creates the sortable, color-coded Dash table."""
    return dash_table.DataTable(
        id="leaderboard-table",
        columns=[
            {"name": label, "id": key, "type": "numeric"} for key, label in _COLUMNS
        ],
        data=rows,
        sort_action="native",
        sort_by=[{"column_id": "composite_score", "direction": "desc"}],
        style_table={"overflowX": "auto"},
        style_cell={"padding": "0.6rem", "textAlign": "left"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"column_id": column}, "backgroundColor": color}
            for column, color in _COLUMN_STYLES.items()
        ],
    )


def create_app(datasets: dict[str, list[dict[str, object]]] | None = None) -> Dash:
    """Creates the Dash application."""
    datasets = datasets or load_leaderboard_datasets()
    categories = list(datasets)
    app = Dash(__name__)
    app.layout = html.Main(
        [
            html.H1("Leaderboard"),
            html.P("Klikk på en kolonneoverskrift for å sortere."),
            dcc.Dropdown(
                id="leaderboard-category",
                options=[
                    {"label": category, "value": category} for category in categories
                ],
                value=categories[0],
                clearable=False,
            ),
            html.Div(id="leaderboard-table-container"),
        ]
    )

    @app.callback(
        Output("leaderboard-table-container", "children"),
        Input("leaderboard-category", "value"),
    )
    def update_table(category: str) -> dash_table.DataTable:
        return _table(datasets[category])

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
