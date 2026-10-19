"""
Synthetic A/B test data generator for the Growth Experimentation Agent.

Creates a row-level CSV (user_id, variant, converted) with a configurable
split, conversion rates, sample sizes, and random seed. Safe to import (no
side effects); all work happens under if __name__ == "__main__" or via an
explicit generate() call.

Design notes:
- Parameters are validated inside generate() (not only in the CLI) so direct
  API calls cannot write invalid data; non-numeric types raise ValueError
  (not TypeError).
- Uses a local random.Random(seed) instance; does NOT mutate the global RNG
  state.
- Writes to a temporary file in the target directory and atomically replaces
  the destination via os.replace, so a failure never leaves a partial file.
- user_id values are neutral sequential IDs (u000000, u000001, ...) assigned
  AFTER shuffling, so they do not reveal the arm assignment.
- Realized conversion rates are REPORTED (returned in the summary) but not
  validated against the target probabilities, since random sampling will not
  hit the target exactly.

Usage:
    python generate_mock_data.py
    python generate_mock_data.py --n-control 8000 --n-treatment 8000 \\
        --p-control 0.03 --p-treatment 0.035
    python generate_mock_data.py --output data/custom.csv --seed 7 --overwrite
"""

import argparse
import csv
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, Union

DEFAULT_N_CONTROL = 5000
DEFAULT_N_TREATMENT = 5000
DEFAULT_P_CONTROL = 0.048
DEFAULT_P_TREATMENT = 0.061
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "mock_ab_test.csv"


def _validate_params(
    n_control: int,
    n_treatment: int,
    p_control: float,
    p_treatment: float,
):
    # Coerce to the expected numeric types first so non-numeric inputs raise
    # ValueError (not a raw TypeError from a comparison). Returns the coerced
    # values so callers use validated types.
    try:
        n_control_i = int(n_control)
        n_treatment_i = int(n_treatment)
        p_control_f = float(p_control)
        p_treatment_f = float(p_treatment)
    except (TypeError, ValueError):
        raise ValueError(
            "n_control and n_treatment must be integers; "
            "p_control and p_treatment must be numeric."
        )
    # Reject non-integral numbers (e.g. 10.5) instead of truncating silently.
    for name, original, coerced in (
        ("n_control", n_control, n_control_i),
        ("n_treatment", n_treatment, n_treatment_i),
    ):
        if isinstance(original, float) and original != coerced:
            raise ValueError(f"{name} must be an integer, got {original}.")
        if isinstance(original, str) and str(coerced) != original.strip().lstrip("+"):
            raise ValueError(f"{name} must be an integer, got {original!r}.")
    if n_control_i <= 0:
        raise ValueError(f"n_control must be positive, got {n_control_i}.")
    if n_treatment_i <= 0:
        raise ValueError(f"n_treatment must be positive, got {n_treatment_i}.")
    for name, p in (("p_control", p_control_f), ("p_treatment", p_treatment_f)):
        if math.isnan(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {p}.")
    return n_control_i, n_treatment_i, p_control_f, p_treatment_f


def generate(
    output_path: Union[str, Path] = DEFAULT_OUTPUT,
    n_control: int = DEFAULT_N_CONTROL,
    n_treatment: int = DEFAULT_N_TREATMENT,
    p_control: float = DEFAULT_P_CONTROL,
    p_treatment: float = DEFAULT_P_TREATMENT,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> Dict[str, Union[int, float, str]]:
    """Generate a synthetic row-level A/B test CSV and return summary stats.

    Validates all parameters before any file I/O. Raises FileExistsError if the
    target exists and overwrite is False. Uses atomic write (temp file +
    os.replace) so a crash cannot leave a partial CSV at the destination.

    The returned summary includes realized conversion rates for reporting; they
    are not asserted against the target probabilities (random sampling).
    """
    n_control, n_treatment, p_control, p_treatment = _validate_params(
        n_control, n_treatment, p_control, p_treatment
    )
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True "
            f"(CLI: --overwrite) to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    rows = []
    for _ in range(n_control):
        rows.append(("control", 1 if rng.random() < p_control else 0))
    for _ in range(n_treatment):
        rows.append(("treatment", 1 if rng.random() < p_treatment else 0))

    # Shuffle so row order does not leak the arm assignment pattern, then
    # assign neutral sequential IDs (no arm-prefix blinding leak).
    rng.shuffle(rows)
    rows = [
        (f"u{i:06d}", variant, converted)
        for i, (variant, converted) in enumerate(rows)
    ]

    # Atomic write: temp file in the same directory, then os.replace.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(output_path.parent), prefix=".mock_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "variant", "converted"])
            writer.writerows(rows)
        os.replace(tmp_name, str(output_path))
    except BaseException:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        raise

    # Post-write summary (report, not validate - random rates vary).
    total = len(rows)
    n_c = sum(1 for r in rows if r[1] == "control")
    n_t = sum(1 for r in rows if r[1] == "treatment")
    conv_c = sum(r[2] for r in rows if r[1] == "control")
    conv_t = sum(r[2] for r in rows if r[1] == "treatment")

    # Explicit checks (not assert - must survive python -O).
    if total != n_control + n_treatment:
        raise RuntimeError(f"Row count mismatch: {total}.")
    if n_c != n_control:
        raise RuntimeError(f"Control size mismatch: {n_c} != {n_control}.")
    if n_t != n_treatment:
        raise RuntimeError(f"Treatment size mismatch: {n_t} != {n_treatment}.")

    return {
        "path": str(output_path),
        "total": total,
        "n_control": n_c,
        "n_treatment": n_t,
        "conv_control": conv_c,
        "conv_treatment": conv_t,
        "rate_control": conv_c / n_c if n_c else 0.0,
        "rate_treatment": conv_t / n_t if n_t else 0.0,
        "seed": seed,
    }


def _print_summary(summary: Dict[str, Union[int, float, str]]) -> None:
    print(f"Wrote {summary['total']} rows to {summary['path']}")
    print(f"  seed: {summary['seed']}")
    print(
        f"  control:   n={summary['n_control']}, "
        f"conversions={summary['conv_control']}, "
        f"realized rate={summary['rate_control']:.4%}"
    )
    print(
        f"  treatment: n={summary['n_treatment']}, "
        f"conversions={summary['conv_treatment']}, "
        f"realized rate={summary['rate_treatment']:.4%}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic row-level A/B test CSV."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--n-control", type=int, default=DEFAULT_N_CONTROL,
                        help=f"Control group size (default: {DEFAULT_N_CONTROL}).")
    parser.add_argument("--n-treatment", type=int, default=DEFAULT_N_TREATMENT,
                        help=f"Treatment group size (default: {DEFAULT_N_TREATMENT}).")
    parser.add_argument("--p-control", type=float, default=DEFAULT_P_CONTROL,
                        help=f"Control conversion probability (default: {DEFAULT_P_CONTROL}).")
    parser.add_argument("--p-treatment", type=float, default=DEFAULT_P_TREATMENT,
                        help=f"Treatment conversion probability (default: {DEFAULT_P_TREATMENT}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output file if it already exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        summary = generate(
            output_path=args.output,
            n_control=args.n_control,
            n_treatment=args.n_treatment,
            p_control=args.p_control,
            p_treatment=args.p_treatment,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError, OSError) as e:
        raise SystemExit(f"Error: {e}")
    _print_summary(summary)
