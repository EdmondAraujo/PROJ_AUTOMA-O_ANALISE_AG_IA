"""
Growth Experimentation Agent - Autonomous A/B test analysis + Claude readout.

Reads a row-level A/B test CSV (user_id, variant, converted), runs SRM and
conversion-rate significance checks locally with scipy, and feeds the exact
numbers to Claude for an executive readout.

Usage:
    python main.py --csv data/mock_ab_test.csv
    python main.py --csv data/mock_ab_test.csv --alpha 0.01 --min-effect 0.01
    python main.py --csv data/mock_ab_test.csv --skip-api

Statistical methods:
- SRM: chi-square when all expected cells >= 5; otherwise exact binomial
  (chi-square approximation unreliable for small expected counts).
- Conversion: pooled two-tailed z-test; Fisher's exact when any expected
  cell < 5. CI via Newcombe (1998) method 10 (square-and-add of Wilson
  score deviations; coverage close to nominal).
- MDE/power use the unpooled (Wald) SE computed from observed rates.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3


@dataclass
class SRMResult:
    observed_control: int
    observed_treatment: int
    expected_ratio: float
    alpha: float
    chi2_stat: float
    p_value: float
    exact_p_value: Optional[float]
    decision_p_value: float
    decision_method: str
    passed: bool
    small_sample_warning: Optional[str]
    min_expected: float


@dataclass
class ConversionResult:
    conv_control: int
    n_control: int
    conv_treatment: int
    n_treatment: int
    rate_control: float
    rate_treatment: float
    absolute_lift: float
    relative_lift: float
    z_stat: float
    p_value: float
    min_expected_cell: float
    small_sample_warning: Optional[str]
    fisher_p_value: Optional[float]
    decision_method: str
    decision_p_value: float
    ci_lower: float
    ci_upper: float
    mde_80_power: float
    observed_power: float


def _validate_alpha(alpha: float) -> None:
    if math.isnan(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")


def _wilson_bounds(conv: int, n: int, z: float) -> Tuple[float, float]:
    """Wilson score interval bounds for a single proportion."""
    p = conv / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    spread = z * float(np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)))
    lower = (center - spread) / denom
    upper = (center + spread) / denom
    return lower, upper


def newcombe_diff_ci(
    conv1: int, n1: int, conv2: int, n2: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Newcombe (1998) method 10 CI for the difference p2 - p1.

    Square-and-add combination of Wilson score intervals (MOVER); coverage
    close to nominal. Call with (control, treatment) to get the CI for
    treatment minus control. Validated against known values:
    newcombe_diff_ci(50, 100, 60, 100) -> approx (-0.037302, 0.232123).
    """
    _validate_alpha(alpha)
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Group sizes must be positive.")
    if not (0 <= conv1 <= n1) or not (0 <= conv2 <= n2):
        raise ValueError("Conversion counts must be in [0, n].")
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p1 = conv1 / n1
    p2 = conv2 / n2
    l1, u1 = _wilson_bounds(conv1, n1, z)
    l2, u2 = _wilson_bounds(conv2, n2, z)
    d = p2 - p1
    lower = d - float(np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
    upper = d + float(np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))
    lower = float(np.clip(lower, -1.0, 1.0))
    upper = float(np.clip(upper, -1.0, 1.0))
    return lower, upper


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.05,
) -> SRMResult:
    """Sample Ratio Mismatch check.

    Chi-square test of the observed split vs the expected ratio. When any
    expected cell count is < 5 the chi-square approximation is unreliable,
    so an exact binomial test is computed and used as the DECISION
    (`decision_method='exact-binomial'`, `passed` reflects it); the
    chi-square statistic/p-value are still reported for reference.
    """
    _validate_alpha(alpha)
    if not (0.0 < expected_ratio < 1.0):
        raise ValueError(f"expected_ratio must be in (0, 1), got {expected_ratio}.")
    if n_control < 0 or n_treatment < 0:
        raise ValueError(
            f"negative group sizes: control={n_control}, treatment={n_treatment}."
        )
    total = n_control + n_treatment
    if total == 0:
        raise ValueError("Cannot run SRM check on zero total samples.")

    exp_c = total * expected_ratio
    exp_t = total * (1.0 - expected_ratio)
    chi2 = ((n_control - exp_c) ** 2) / exp_c + ((n_treatment - exp_t) ** 2) / exp_t
    p_value = float(stats.chi2.sf(chi2, df=1))
    min_expected = min(exp_c, exp_t)

    exact_p: Optional[float] = None
    warning: Optional[str] = None
    if min_expected < 5:
        exact_p = float(stats.binomtest(n_control, total, expected_ratio).pvalue)
        warning = (
            "Small expected cell counts (<5): chi-square approximation may be "
            "unreliable; exact binomial test used for the decision."
        )
        decision_method = "exact-binomial"
        decision_p = exact_p
    else:
        decision_method = "chi-square"
        decision_p = p_value

    return SRMResult(
        observed_control=n_control,
        observed_treatment=n_treatment,
        expected_ratio=expected_ratio,
        alpha=alpha,
        chi2_stat=float(chi2),
        p_value=p_value,
        exact_p_value=exact_p,
        decision_p_value=decision_p,
        decision_method=decision_method,
        passed=decision_p >= alpha,
        small_sample_warning=warning,
        min_expected=min_expected,
    )


def conversion_ztest(
    conv_control: int,
    n_control: int,
    conv_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> ConversionResult:
    """Conversion-rate significance test (pooled z-test + Newcombe CI).

    Decision method: 'z-test' when all expected cells >= 5, otherwise
    'fisher-exact' (Fisher's exact p-value decides; z-test p still shown).
    MDE at 80% power and post-hoc power use the unpooled SE from observed
    rates; both are NaN for degenerate rates (0%/100%).
    """
    _validate_alpha(alpha)
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Group sizes must be positive.")
    if not (0 <= conv_control <= n_control) or not (0 <= conv_treatment <= n_treatment):
        raise ValueError("Conversion counts must be in [0, n].")

    rate_c = conv_control / n_control
    rate_t = conv_treatment / n_treatment
    absolute_lift = rate_t - rate_c
    relative_lift = (rate_t - rate_c) / rate_c if rate_c > 0 else float("nan")

    p_pool = (conv_control + conv_treatment) / (n_control + n_treatment)
    se_pool = math.sqrt(
        p_pool * (1.0 - p_pool) * (1.0 / n_control + 1.0 / n_treatment)
    )
    if se_pool > 0:
        z_stat = absolute_lift / se_pool
        p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    else:
        z_stat = 0.0
        p_value = 1.0

    expected = (
        n_control * p_pool,
        n_control * (1.0 - p_pool),
        n_treatment * p_pool,
        n_treatment * (1.0 - p_pool),
    )
    min_expected_cell = min(expected)

    fisher_p: Optional[float] = None
    warning: Optional[str] = None
    if min_expected_cell < 5:
        _, fisher_p = stats.fisher_exact(
            [
                [conv_control, n_control - conv_control],
                [conv_treatment, n_treatment - conv_treatment],
            ]
        )
        fisher_p = float(fisher_p)
        warning = (
            "Small expected cell counts (<5): normal approximation may be "
            "unreliable; Fisher's exact test used for the decision."
        )
        decision_method = "fisher-exact"
        decision_p = fisher_p
    else:
        decision_method = "z-test"
        decision_p = p_value

    ci_lower, ci_upper = newcombe_diff_ci(
        conv_control, n_control, conv_treatment, n_treatment, alpha
    )

    se_unpooled = math.sqrt(
        rate_c * (1.0 - rate_c) / n_control + rate_t * (1.0 - rate_t) / n_treatment
    )
    if se_unpooled > 0:
        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_power = float(stats.norm.ppf(0.80))
        mde_80 = (z_alpha + z_power) * se_unpooled
        observed_power = float(
            stats.norm.cdf(abs(absolute_lift) / se_unpooled - z_alpha)
        )
    else:
        mde_80 = float("nan")
        observed_power = float("nan")

    return ConversionResult(
        conv_control=conv_control,
        n_control=n_control,
        conv_treatment=conv_treatment,
        n_treatment=n_treatment,
        rate_control=rate_c,
        rate_treatment=rate_t,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
        z_stat=float(z_stat),
        p_value=p_value,
        min_expected_cell=min_expected_cell,
        small_sample_warning=warning,
        fisher_p_value=fisher_p,
        decision_method=decision_method,
        decision_p_value=decision_p,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mde_80_power=mde_80,
        observed_power=observed_power,
    )


def load_and_validate_csv(path: str) -> pd.DataFrame:
    """Load and strictly validate a row-level A/B test CSV.

    Required columns: user_id (unique non-empty string, read as text so
    leading zeros survive), variant ('control'/'treatment', case-insensitive),
    converted (integer 0 or 1; 0.5 is an error, not a truncation).
    Raises FileNotFoundError for missing paths and ValueError for everything
    else (empty file, missing columns, nulls, duplicates, bad values).
    Extra columns are ignored.
    """
    required = ["user_id", "variant", "converted"]
    try:
        df = pd.read_csv(path, dtype={"user_id": str})
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty.")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}.")

    df = df[required]

    if df[required].isnull().any().any():
        raise ValueError(
            "Null values found in required columns (user_id, variant, converted)."
        )

    uid = df["user_id"].astype(str)
    if uid.str.strip().eq("").any():
        raise ValueError("user_id values must not be empty or whitespace.")
    if uid.duplicated().any():
        raise ValueError("duplicate user_id values found.")
    df["user_id"] = uid

    variant = df["variant"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(variant.unique()) - {"control", "treatment"})
    if unexpected:
        raise ValueError(
            f"Unexpected variant value(s): {unexpected}. "
            "Expected 'control' or 'treatment'."
        )
    df["variant"] = variant

    converted = pd.to_numeric(df["converted"], errors="coerce")
    if converted.isnull().any():
        raise ValueError("converted contains non-numeric values.")
    if not converted.isin([0, 1]).all():
        raise ValueError("converted must be only 0 or 1.")
    df["converted"] = converted.astype("int64")

    return df


def _fmt_pval(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "N/A"
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.6f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x * 100:.4f}%"


def _fmt_float(x: Optional[float], nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x:.{nd}f}"


def build_user_message(
    srm: SRMResult,
    conv: ConversionResult,
    *,
    alpha: float = 0.05,
    srm_alpha: float = 0.05,
    conv_control: int,
    conv_treatment: int,
    min_effect: Optional[float] = None,
) -> str:
    """Format the exact pre-computed statistics for Claude.

    `conv_control`/`conv_treatment` are the raw conversion counts (used for
    the exact '(x / n)' display); denominators come from the ConversionResult
    itself so direct calls stay consistent.
    """
    conf_pct = round((1.0 - alpha) * 100)
    lines = [
        "Here are the pre-computed statistics from our A/B test engine:",
        "",
        f"## SRM (Sample Ratio Mismatch) Check (alpha={srm_alpha})",
        f"- Expected allocation ratio (control): {srm.expected_ratio * 100:.2f}%",
        f"- Observed control n: {srm.observed_control}",
        f"- Observed treatment n: {srm.observed_treatment}",
        f"- Chi-square statistic: {srm.chi2_stat:.4f}",
        f"- Chi-square p-value: {_fmt_pval(srm.p_value)}",
    ]
    if srm.exact_p_value is not None:
        lines.append(f"- Exact binomial test p-value: {_fmt_pval(srm.exact_p_value)}")
    if srm.small_sample_warning is not None:
        lines.append(f"- WARNING: {srm.small_sample_warning}")
    lines += [
        f"- SRM DECISION method: {srm.decision_method}",
        f"- SRM DECISION p-value: {_fmt_pval(srm.decision_p_value)}",
        f"- SRM PASSED (no mismatch): {srm.passed}",
        f"- Minimum expected cell count: {srm.min_expected:.2f}",
        "",
        f"## Conversion Rate Test (alpha={alpha})",
        f"- Control conversion rate: {_fmt_pct(conv.rate_control)} "
        f"({conv_control} / {conv.n_control})",
        f"- Treatment conversion rate: {_fmt_pct(conv.rate_treatment)} "
        f"({conv_treatment} / {conv.n_treatment})",
        f"- Absolute lift: {_fmt_pct(conv.absolute_lift)}",
        f"- Relative lift: {_fmt_pct(conv.relative_lift)}",
        f"- Z-statistic: {conv.z_stat:.4f}",
        f"- Z-test p-value (two-tailed): {_fmt_pval(conv.p_value)}",
        f"- CONVERSION DECISION method: {conv.decision_method}",
        f"- CONVERSION DECISION p-value: {_fmt_pval(conv.decision_p_value)}",
    ]
    sig = "YES" if conv.decision_p_value < alpha else "NO"
    lines.append(
        f"- Statistically significant at alpha={alpha} "
        f"(via {conv.decision_method}): {sig}"
    )
    lines += [
        f"- {conf_pct}% CI for absolute lift (Newcombe method 10, score): "
        f"[{_fmt_pct(conv.ci_lower)}, {_fmt_pct(conv.ci_upper)}]",
        "- MDE at 80% power (retrospective, computed from observed rates): "
        f"{_fmt_pct(conv.mde_80_power)}",
        "- Observed power (post-hoc, BIASED - context only, not decisive "
        f"for ship/no-ship): {_fmt_float(conv.observed_power)}",
    ]
    if min_effect is not None:
        mde = conv.mde_80_power
        if math.isnan(mde):
            analysis = (
                "MDE is N/A (degenerate rates) - cannot compare to threshold"
            )
        elif mde > min_effect:
            analysis = (
                f"MDE ({_fmt_pct(mde)}) is LARGER than the threshold "
                f"({_fmt_pct(min_effect)}) - underpowered for a meaningful win"
            )
        else:
            analysis = (
                f"MDE ({_fmt_pct(mde)}) is SMALLER than the threshold "
                f"({_fmt_pct(min_effect)}) - adequately powered for a "
                "meaningful win"
            )
        lines.append(f"- Min-effect threshold analysis: {analysis}")
    lines += [
        "",
        "## Sample Sizes",
        f"- Total: {conv.n_control + conv.n_treatment}",
        "",
    ]
    if min_effect is not None:
        lines.append(
            "- Minimum effect threshold (absolute lift) of interest: "
            f"{_fmt_pct(min_effect)}"
        )
    lines.append("Please produce the executive readout following your system instructions.")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a senior growth experimentation analyst. The user sends you PRE-COMPUTED\
 statistics from an A/B test (an SRM integrity check and a conversion-rate\
 analysis). Write a concise executive readout for growth leadership.

Rules:
- Use the EXACT numbers provided. Do NOT recompute, re-derive, or invent\
 statistics; never do arithmetic that is not already in the message.
- Structure: (1) Verdict (SHIP / NO-SHIP / INCONCLUSIVE), (2) SRM integrity,\
 (3) Key results with the confidence interval, (4) Risks & caveats,\
 (5) Recommended next step.
- If SRM failed, the experiment is invalid: recommend NO-SHIP and a rerun\
 regardless of the conversion result.
- The confidence interval is the primary evidence. Post-hoc power is context\
 only and never decisive for ship/no-ship.
- When the decision method is not the default (exact-binomial for SRM,\
 fisher-exact for conversion), say so explicitly and treat the decision\
 p-value as authoritative.
- If a min-effect threshold was provided, judge whether a win is practically\
 meaningful, not just statistically significant.
- Plain business language; under ~300 words.
"""


def _call_claude(
    user_msg: str,
    model: str,
    max_tokens: int,
    temperature: Optional[float],
    timeout: float,
    max_retries: int,
) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: ANTHROPIC_API_KEY is not set. Export it or use --skip-api."
        )
    client = anthropic.Anthropic(
        api_key=api_key, timeout=timeout, max_retries=max_retries
    )
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    parts = [
        block.text
        for block in resp.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts) + "\n"


def run_pipeline(
    csv_path: str,
    expected_ratio: float = 0.5,
    alpha: float = 0.05,
    srm_alpha: float = 0.05,
    skip_api: bool = False,
    model: str = DEFAULT_MODEL,
    min_effect: Optional[float] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: Optional[float] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """Load CSV -> SRM + conversion stats -> (Claude readout | local prompt)."""
    df = load_and_validate_csv(csv_path)

    n_control = int((df["variant"] == "control").sum())
    n_treatment = int((df["variant"] == "treatment").sum())
    if n_control == 0 or n_treatment == 0:
        raise ValueError(
            "CSV must contain both 'control' and 'treatment' rows "
            f"(control n={n_control}, treatment n={n_treatment})."
        )
    conv_control = int(df.loc[df["variant"] == "control", "converted"].sum())
    conv_treatment = int(df.loc[df["variant"] == "treatment", "converted"].sum())

    srm = srm_check(n_control, n_treatment, expected_ratio=expected_ratio, alpha=srm_alpha)
    conv = conversion_ztest(
        conv_control, n_control, conv_treatment, n_treatment, alpha=alpha
    )
    user_msg = build_user_message(
        srm,
        conv,
        alpha=alpha,
        srm_alpha=srm_alpha,
        conv_control=conv_control,
        conv_treatment=conv_treatment,
        min_effect=min_effect,
    )

    banner = "=" * 70 + "\nGROWTH EXPERIMENTATION - EXECUTIVE READOUT\n" + "=" * 70 + "\n"
    if skip_api:
        return (
            banner
            + "[LOCAL MODE - API call skipped]\n"
            + "[NOTE: this prints the exact prompt sent to Claude, not a local "
            "executive readout. A readout requires a live API call.]\n\n"
            + user_msg
            + "\n"
            + "=" * 70
            + "\n"
        )
    reply = _call_claude(user_msg, model, max_tokens, temperature, timeout, max_retries)
    return banner + reply


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Growth Experimentation Agent: analyze an A/B test CSV "
            "(SRM + conversion stats) and get a Claude executive readout."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the row-level A/B test CSV (user_id,variant,converted).",
    )
    parser.add_argument(
        "--expected-ratio", type=float, default=0.5,
        help="Expected control allocation ratio (default: 0.5).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for the conversion test (default: 0.05).",
    )
    parser.add_argument(
        "--srm-alpha", type=float, default=0.05,
        help="Significance level for the SRM check (default: 0.05).",
    )
    parser.add_argument(
        "--min-effect", type=float, default=None,
        help="Minimum practically meaningful absolute lift, in (0, 1).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max completion tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature in [0, 1] (default: SDK default).",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"API timeout in seconds (default: {DEFAULT_TIMEOUT:g}).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=f"API retry budget (default: {DEFAULT_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--skip-api", action="store_true",
        help="Skip the Claude call and print the exact prompt locally.",
    )
    args = parser.parse_args(argv)

    if not (0.0 < args.expected_ratio < 1.0):
        parser.error("--expected-ratio must be in (0, 1).")
    if not (0.0 < args.alpha < 1.0):
        parser.error("--alpha must be in (0, 1).")
    if not (0.0 < args.srm_alpha < 1.0):
        parser.error("--srm-alpha must be in (0, 1).")
    if args.min_effect is not None and not (0.0 < args.min_effect < 1.0):
        parser.error("--min-effect must be in (0, 1).")
    if args.temperature is not None and not (0.0 <= args.temperature <= 1.0):
        parser.error("--temperature must be in [0, 1].")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0.")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0.")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0.")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        out = run_pipeline(
            args.csv,
            expected_ratio=args.expected_ratio,
            alpha=args.alpha,
            srm_alpha=args.srm_alpha,
            skip_api=args.skip_api,
            model=args.model,
            min_effect=args.min_effect,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Error: {e}")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
