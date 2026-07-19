"""CLI entrypoint for reconciliation baseline experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconciliation.experiment_runner import load_config, run_reconciliation_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reconciliation_config.yaml",
        help="Path to reconciliation config.",
    )
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path if config_path.exists() else None)
    config["project_root"] = str(PROJECT_ROOT)
    outputs = run_reconciliation_baselines(config)
    print("Reconciliation baselines completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
