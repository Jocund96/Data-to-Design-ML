"""Run the S1 Linear Family workflows in dependency order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKFLOWS = {
    3: [["scripts/run_week03_linear.py"]],
    4: [["scripts/run_week04_representation.py"]],
    5: [],
    6: [["scripts/run_week06_semantic_missingness.py"]],
    7: [
        ["scripts/run_week07_preprocess_shared_uhpc.py"],
        ["scripts/run_week07_linear_experiments.py"],
    ],
    8: [["-m", "s1_linear.week08.runner"]],
    9: [["-m", "s1_linear.week09.runner"]],
    10: [["-m", "s1_linear.week10.runner"]],
}

NOTEBOOKS = {
    3: "week03_linear_family.ipynb",
    4: "week04_representation_experiments.ipynb",
    5: "week05_uhpc_import_and_target_check.ipynb",
    6: "week06_semantic_missingness_strategies.ipynb",
    7: "week07_linear_family_results.ipynb",
    8: "week08_publication_generalization.ipynb",
    9: "week09_uncertainty_calibration.ipynb",
    10: "week10_feature_attribution.ipynb",
}


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def execute_notebook(filename: str) -> None:
    run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=-1",
            str(PROJECT_ROOT / "notebooks" / filename),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S1 Linear workflows and notebooks in dependency order."
    )
    parser.add_argument("--from-week", type=int, choices=WORKFLOWS, default=3)
    parser.add_argument("--to-week", type=int, choices=WORKFLOWS, default=10)
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Run experiment code without executing the result notebooks.",
    )
    parser.add_argument(
        "--notebooks-only",
        action="store_true",
        help="Execute notebooks without rerunning experiment code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_week > args.to_week:
        raise ValueError("--from-week must be less than or equal to --to-week.")

    for week in range(args.from_week, args.to_week + 1):
        print(f"\n{'=' * 16} Week {week} {'=' * 16}", flush=True)

        if not args.notebooks_only:
            for command in WORKFLOWS[week]:
                run([sys.executable, *command])

        if not args.skip_notebooks:
            execute_notebook(NOTEBOOKS[week])

    print("\nS1 Linear workflow completed.", flush=True)


if __name__ == "__main__":
    main()
