#!/usr/bin/env python3
"""Extract eval_loss entries from a LeRobot training log as public CSV."""

import argparse
import csv
import re
from pathlib import Path

PATTERN = re.compile(r"step (\d+): eval_loss=([0-9.]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    values = {int(step): float(loss) for step, loss in PATTERN.findall(args.log.read_text(errors="replace"))}
    if not values:
        raise SystemExit(f"No eval_loss entries found in {args.log}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("run", "step", "eval_loss", "checkpoint_saved"))
        for step in sorted(values):
            writer.writerow((args.run, step, f"{values[step]:.4f}", str(step % 5000 == 0).lower()))


if __name__ == "__main__":
    main()
