"""Score a saved prediction CSV without a fitted model."""

import argparse
from pathlib import Path

from f1_lap_time.evaluation import read_predictions, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, help="New CSV path (default: stdout)")
    args = parser.parse_args()
    try:
        scores = score_predictions(
            read_predictions(args.predictions), model=args.model, run_id=args.run_id
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            scores.to_csv(args.output, index=False, mode="x", lineterminator="\n")
        print(scores.to_csv(index=False, lineterminator="\n"), end="")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot score predictions: {exc}\n")


if __name__ == "__main__":
    main()
