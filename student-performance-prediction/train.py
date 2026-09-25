"""Train, evaluate and save the complete pass/fail experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
from src.config import DATA_PATH, ROOT
from src.data import load_data
from src.evaluation import run_experiment
from src.reporting import statistical_analysis, write_reports


def train(path: Path = DATA_PATH, schema: str = "uci", pass_mark: float | None = None,
          output: Path = ROOT) -> dict:
    frame, y, metadata = load_data(path, schema, pass_mark)
    with threadpool_limits(limits=2):
        bundle = run_experiment(frame, y, schema)
    metadata["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
    bundle["metadata"] = metadata
    bundle["schema_version"] = 2
    # Only aggregated statistics and trained models are serialized, no student rows.
    train_indices = bundle["train_indices"]
    bundle["statistics"] = statistical_analysis(frame.iloc[train_indices], y.iloc[train_indices], schema)
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    write_reports(bundle, output / "reports")
    print(f"Selected: {bundle['model_name']} | {metadata['target_definition']}")
    print(bundle["test_metrics"].to_string(index=False))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--schema", choices=["uci", "custom"], default="uci")
    parser.add_argument("--pass-mark", type=float)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Use a separate folder for private institutional experiments.")
    args = parser.parse_args()
    train(args.csv, args.schema, args.pass_mark, args.output)


if __name__ == "__main__":
    main()
