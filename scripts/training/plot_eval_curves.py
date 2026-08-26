#!/usr/bin/env python3
"""Render one or more eval-curve CSV files as a dependency-free SVG."""

import argparse
import csv
from pathlib import Path


def read_curve(path: Path) -> tuple[str, list[tuple[int, float]]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows[0]["run"], [(int(row["step"]), float(row["eval_loss"])) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    curves = [read_curve(path) for path in args.csv]
    points = [point for _, curve in curves for point in curve]
    min_step, max_step = min(p[0] for p in points), max(p[0] for p in points)
    min_loss, max_loss = min(p[1] for p in points), max(p[1] for p in points)
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 35, 70

    def x(step: int) -> float:
        return left + (step - min_step) / (max_step - min_step) * (width - left - right)

    def y(loss: float) -> float:
        return top + (max_loss - loss) / (max_loss - min_loss) * (height - top - bottom)

    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#111827;font-size:13px}.grid{stroke:#e5e7eb}.axis{stroke:#111827;stroke-width:1.5}</style>',
    ]
    for index in range(6):
        loss = min_loss + (max_loss - min_loss) * index / 5
        yy = y(loss)
        svg.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>')
        svg.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{loss:.3f}</text>')
    for step in range(10000, 50001, 10000):
        xx = x(step)
        svg.append(f'<line class="grid" x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}"/>')
        svg.append(f'<text x="{xx:.1f}" y="{height-bottom+24}" text-anchor="middle">{step//1000}k</text>')
    svg.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle">training step</text>',
        f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle">eval loss</text>',
    ])
    for index, (name, curve) in enumerate(curves):
        color = colors[index % len(colors)]
        polyline = " ".join(f"{x(step):.1f},{y(loss):.1f}" for step, loss in curve)
        svg.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = 28 + index * 22
        svg.append(f'<line x1="{width-330}" y1="{legend_y-4}" x2="{width-300}" y2="{legend_y-4}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{width-292}" y="{legend_y}">{name}</text>')
    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
