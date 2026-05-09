import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional


RESULTS_DIR = Path("benchmark_results")
PLOTS_DIR = RESULTS_DIR / "plots"

BENCHMARK_PATTERN = re.compile(r"benchmark_results_ef_search_(\d+)_hnsw\.json$")
COMPARE_PATTERN = re.compile(r"compare_ef_search_(\d+)_flat_vs_hnsw\.json$")
RERANKED_PATTERN = re.compile(r"compare_reranked_flat_vs_hnsw_(\d+)_top_5\.json$")

SVG_WIDTH = 1200
SVG_HEIGHT = 760

MARGIN_LEFT = 100
MARGIN_RIGHT = 110
MARGIN_TOP = 155
MARGIN_BOTTOM = 105

FONT_FAMILY = "Inter, Helvetica, Arial, sans-serif"
BACKGROUND = "#ffffff"
TEXT = "#111827"
MUTED_TEXT = "#4b5563"
GRID = "#e5e7eb"
AXIS = "#374151"

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def extract_ef_search(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.match(path.name)
    if not match:
        raise ValueError(f"Could not extract ef_search from {path}")
    return int(match.group(1))


def index_files(results_dir: Path, pattern: re.Pattern[str]) -> Dict[int, Path]:
    indexed: Dict[int, Path] = {}
    for path in results_dir.iterdir():
        if path.is_file() and pattern.match(path.name):
            indexed[extract_ef_search(path, pattern)] = path
    return indexed


def collect_metrics(results_dir: Path) -> List[Dict[str, float]]:
    benchmark_files = index_files(results_dir, BENCHMARK_PATTERN)
    compare_files = index_files(results_dir, COMPARE_PATTERN)
    reranked_files = index_files(results_dir, RERANKED_PATTERN)

    common_ef_search_values = sorted(
        set(benchmark_files) & set(compare_files) & set(reranked_files)
    )

    if not common_ef_search_values:
        raise ValueError(f"No matching ef_search triplets found under {results_dir}")

    metrics: List[Dict[str, float]] = []

    for ef_search in common_ef_search_values:
        benchmark_summary = load_json(benchmark_files[ef_search])["summary"]
        compare_summary = load_json(compare_files[ef_search])["summary"]
        reranked_summary = load_json(reranked_files[ef_search])["summary"]

        metrics.append(
            {
                "ef_search": float(ef_search),
                "p95_latency_ms": float(
                    benchmark_summary["retrieval_p95_across_queries_ms"]
                ),
                "avg_retrieval_recall_at_5": float(compare_summary["avg_recall@5"]),
                "min_retrieval_recall_at_5": float(compare_summary["min_recall@5"]),
                "avg_reranked_recall_at_5": float(
                    reranked_summary["avg_reranked_recall@5"]
                ),
                "min_reranked_recall_at_5": float(
                    reranked_summary["min_reranked_recall@5"]
                ),
            }
        )

    return metrics


def format_number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def scale_x(value: float, x_min: float, x_max: float, plot_width: float) -> float:
    if x_max == x_min:
        return MARGIN_LEFT + plot_width / 2
    return MARGIN_LEFT + ((value - x_min) / (x_max - x_min)) * plot_width


def scale_y(value: float, y_min: float, y_max: float, plot_height: float) -> float:
    if y_max == y_min:
        return MARGIN_TOP + plot_height / 2
    ratio = (value - y_min) / (y_max - y_min)
    return MARGIN_TOP + plot_height - ratio * plot_height


def svg_line(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        stroke_width: float = 1.0,
        dasharray: Optional[str] = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"{dash_attr} />'
    )


def svg_text(
        x: float,
        y: float,
        text: str,
        size: int = 14,
        fill: str = TEXT,
        anchor: str = "middle",
        weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" '
        f'font-family="{FONT_FAMILY}" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(text)}</text>'
    )


def svg_circle(x: float, y: float, radius: float, fill: str) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" />'


def svg_polyline(points: List[str], stroke: str) -> str:
    return (
        f'<polyline points="{" ".join(points)}" fill="none" '
        f'stroke="{stroke}" stroke-width="3.5" stroke-linejoin="round" '
        f'stroke-linecap="round" />'
    )


def render_line_chart_svg(
        title: str,
        subtitle: str,
        x_values: List[float],
        left_series: List[Dict[str, Any]],
        output_path: Path,
        left_axis_label: str,
        left_y_min: float,
        left_y_max: float,
        right_series: Optional[List[Dict[str, Any]]] = None,
        right_axis_label: Optional[str] = None,
        right_y_min: Optional[float] = None,
        right_y_max: Optional[float] = None,
) -> None:
    plot_width = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    x_min = min(x_values)
    x_max = max(x_values)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        f'<rect width="100%" height="100%" fill="{BACKGROUND}" />',
        f'<rect x="28" y="24" width="{SVG_WIDTH - 56}" height="{SVG_HEIGHT - 48}" '
        f'rx="24" fill="#ffffff" stroke="#e5e7eb" stroke-width="1" />',
        svg_text(SVG_WIDTH / 2, 54, title, size=26, weight="bold", fill=TEXT),
        svg_text(SVG_WIDTH / 2, 82, subtitle, size=14, fill=MUTED_TEXT),
    ]

    legend_x = MARGIN_LEFT
    legend_y = 112
    legend_entries = list(left_series) + list(right_series or [])

    for series in legend_entries:
        svg_parts.append(svg_line(legend_x, legend_y, legend_x + 28, legend_y, series["color"], 3.5))
        svg_parts.append(svg_circle(legend_x + 14, legend_y, 4.5, series["color"]))
        svg_parts.append(
            svg_text(
                legend_x + 40,
                legend_y + 5,
                series["label"],
                size=13,
                fill=MUTED_TEXT,
                anchor="start",
                )
        )
        legend_x += series.get("legend_width", 300)

    svg_parts.append(
        svg_line(MARGIN_LEFT, legend_y + 28, SVG_WIDTH - MARGIN_RIGHT, legend_y + 28, "#f1f5f9", 1)
    )

    tick_count = 6
    for tick_index in range(tick_count):
        tick_value = left_y_min + (
                (left_y_max - left_y_min) * tick_index / (tick_count - 1)
        )
        y = scale_y(tick_value, left_y_min, left_y_max, plot_height)

        svg_parts.append(
            svg_line(MARGIN_LEFT, y, SVG_WIDTH - MARGIN_RIGHT, y, GRID, 1, "4 6")
        )
        svg_parts.append(
            svg_text(
                MARGIN_LEFT - 14,
                y + 5,
                format_number(tick_value),
                size=12,
                fill=MUTED_TEXT,
                anchor="end",
                )
        )

        if right_series and right_y_min is not None and right_y_max is not None:
            right_tick_value = right_y_min + (
                    (right_y_max - right_y_min) * tick_index / (tick_count - 1)
            )
            svg_parts.append(
                svg_text(
                    SVG_WIDTH - MARGIN_RIGHT + 14,
                    y + 5,
                    format_number(right_tick_value),
                    size=12,
                    fill=MUTED_TEXT,
                    anchor="start",
                    )
            )

    svg_parts.extend(
        [
            svg_line(MARGIN_LEFT, MARGIN_TOP, MARGIN_LEFT, SVG_HEIGHT - MARGIN_BOTTOM, AXIS, 1.3),
            svg_line(SVG_WIDTH - MARGIN_RIGHT, MARGIN_TOP, SVG_WIDTH - MARGIN_RIGHT, SVG_HEIGHT - MARGIN_BOTTOM, AXIS, 1.3),
            svg_line(MARGIN_LEFT, SVG_HEIGHT - MARGIN_BOTTOM, SVG_WIDTH - MARGIN_RIGHT, SVG_HEIGHT - MARGIN_BOTTOM, AXIS, 1.3),
            svg_text(MARGIN_LEFT - 62, MARGIN_TOP - 22, left_axis_label, size=14, anchor="start", weight="bold", fill=TEXT),
        ]
    )

    if right_series and right_axis_label:
        svg_parts.append(
            svg_text(
                SVG_WIDTH - MARGIN_RIGHT + 4,
                MARGIN_TOP - 22,
                right_axis_label,
                size=14,
                anchor="end",
                weight="bold",
                fill=TEXT,
                )
        )

    for x_value in x_values:
        x = scale_x(x_value, x_min, x_max, plot_width)
        svg_parts.append(
            svg_line(x, SVG_HEIGHT - MARGIN_BOTTOM, x, SVG_HEIGHT - MARGIN_BOTTOM + 7, AXIS, 1.2)
        )
        svg_parts.append(
            svg_text(x, SVG_HEIGHT - MARGIN_BOTTOM + 29, str(int(x_value)), size=12, fill=MUTED_TEXT)
        )

    svg_parts.append(
        svg_text(SVG_WIDTH / 2, SVG_HEIGHT - 34, "ef_search", size=14, weight="bold", fill=TEXT)
    )

    for series in left_series:
        points: List[str] = []
        for x_value, y_value in zip(x_values, series["values"]):
            x = scale_x(x_value, x_min, x_max, plot_width)
            y = scale_y(y_value, left_y_min, left_y_max, plot_height)
            points.append(f"{x:.2f},{y:.2f}")
        svg_parts.append(svg_polyline(points, series["color"]))

        for x_value, y_value in zip(x_values, series["values"]):
            x = scale_x(x_value, x_min, x_max, plot_width)
            y = scale_y(y_value, left_y_min, left_y_max, plot_height)
            svg_parts.append(svg_circle(x, y, 5, series["color"]))

    if right_series and right_y_min is not None and right_y_max is not None:
        for series in right_series:
            points = []
            for x_value, y_value in zip(x_values, series["values"]):
                x = scale_x(x_value, x_min, x_max, plot_width)
                y = scale_y(y_value, right_y_min, right_y_max, plot_height)
                points.append(f"{x:.2f},{y:.2f}")
            svg_parts.append(svg_polyline(points, series["color"]))

            for x_value, y_value in zip(x_values, series["values"]):
                x = scale_x(x_value, x_min, x_max, plot_width)
                y = scale_y(y_value, right_y_min, right_y_max, plot_height)
                svg_parts.append(svg_circle(x, y, 5, series["color"]))

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts))


def plot_reranked_recall_and_latency(metrics: List[Dict[str, float]], output_path: Path) -> None:
    x_values = [row["ef_search"] for row in metrics]
    p95_values = [row["p95_latency_ms"] for row in metrics]
    latency_min = min(p95_values)
    latency_max = max(p95_values)
    latency_padding = max((latency_max - latency_min) * 0.12, 1.0)

    render_line_chart_svg(
        title="Finding the ef_search Sweet Spot",
        subtitle="Final reranked quality improves quickly, while retrieval p95 latency shows the cost of higher search effort",
        x_values=x_values,
        left_series=[
            {
                "label": "Avg reranked recall@5",
                "color": BLUE,
                "values": [row["avg_reranked_recall_at_5"] for row in metrics],
                "legend_width": 300,
            },
            {
                "label": "Min reranked recall@5",
                "color": GREEN,
                "values": [row["min_reranked_recall_at_5"] for row in metrics],
                "legend_width": 300,
            },
        ],
        output_path=output_path,
        left_axis_label="Recall@5",
        left_y_min=0.0,
        left_y_max=1.0,
        right_series=[
            {
                "label": "Retrieval p95 latency",
                "color": RED,
                "values": p95_values,
                "legend_width": 260,
            }
        ],
        right_axis_label="Retrieval p95 latency (ms)",
        right_y_min=latency_min - latency_padding,
        right_y_max=latency_max + latency_padding,
    )


def plot_retrieval_vs_reranked_recall(metrics: List[Dict[str, float]], output_path: Path) -> None:
    render_line_chart_svg(
        title="Retrieval Quality vs Final Reranked Quality",
        subtitle="Reranking quality depends on whether retrieval first captures the right candidate set",
        x_values=[row["ef_search"] for row in metrics],
        left_series=[
            {
                "label": "Avg retrieval recall@5",
                "color": PURPLE,
                "values": [row["avg_retrieval_recall_at_5"] for row in metrics],
                "legend_width": 320,
            },
            {
                "label": "Avg reranked recall@5",
                "color": ORANGE,
                "values": [row["avg_reranked_recall_at_5"] for row in metrics],
                "legend_width": 320,
            },
        ],
        output_path=output_path,
        left_axis_label="Recall@5",
        left_y_min=0.70,
        left_y_max=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ef_search benchmark metrics from benchmark_results JSON files."
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        type=Path,
        help="Directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=PLOTS_DIR,
        type=Path,
        help="Directory where plots will be written.",
    )
    args = parser.parse_args()

    metrics = collect_metrics(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_reranked_recall_and_latency(
        metrics,
        args.output_dir / "ef_search_vs_reranked_recall_and_p95_latency.svg",
        )
    plot_retrieval_vs_reranked_recall(
        metrics,
        args.output_dir / "ef_search_vs_retrieval_and_reranked_recall.svg",
        )

    print("Loaded ef_search values:", [int(row["ef_search"]) for row in metrics])
    print("Wrote plots to:", args.output_dir)


if __name__ == "__main__":
    main()