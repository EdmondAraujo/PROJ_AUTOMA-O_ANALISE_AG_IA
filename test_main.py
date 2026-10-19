"""
Automated tests for the Growth Experimentation Agent.

Run with:
    python -m pytest test_main.py -v

Covers: SRM chi-square + exact binomial decision, conversion z-test, Newcombe
method 10 score CI, MDE, power, Fisher exact override, degenerate cases
(p=0, p=1, empty groups, invalid converted), strict CSV validation, and prompt
labeling (CI confidence level matches alpha, decision methods reflected).
"""

import math

import pytest

from main import (
    build_user_message,
    conversion_ztest,
    load_and_validate_csv,
    newcombe_diff_ci,
    run_pipeline,
    srm_check,
    _wilson_bounds,
)


@pytest.fixture
def known_csv(tmp_path):
    """CSV fixture with known counts: control 10/100, treatment 20/100."""
    p = tmp_path / "known.csv"
    lines = ["user_id,variant,converted"]
    for i in range(100):
        lines.append(f"c{i:03d},control,{1 if i < 10 else 0}")
    for i in range(100):
        lines.append(f"t{i:03d},treatment,{1 if i < 20 else 0}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def small_csv(tmp_path):
    """CSV fixture triggering Fisher override: control 0/5, treatment 3/5."""
    p = tmp_path / "small.csv"
    lines = ["user_id,variant,converted"]
    for i in range(5):
        lines.append(f"c{i},control,0")
    for i in range(3):
        lines.append(f"t{i},treatment,1")
    for i in range(3, 5):
        lines.append(f"t{i},treatment,0")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# SRM (chi-square + exact binomial decision for small samples)
# ---------------------------------------------------------------------------

class TestSRM:
    def test_balanced_passes(self):
        r = srm_check(5000, 5000, expected_ratio=0.5, alpha=0.05)
        assert r.passed is True
        assert r.chi2_stat == pytest.approx(0.0)
        assert r.p_value == pytest.approx(1.0)
        assert r.exact_p_value is None
        assert r.decision_method == "chi-square"
        assert r.decision_p_value == pytest.approx(r.p_value)

    def test_unbalanced_fails_and_matches_hand_calc(self):
        # chi2 = (500)^2/5000 * 2 = 100 exactly
        r = srm_check(5500, 4500, expected_ratio=0.5, alpha=0.05)
        assert r.passed is False
        assert r.chi2_stat == pytest.approx(100.0)
        assert r.p_value < 1e-20
        assert r.decision_method == "chi-square"

    def test_exact_binomial_overrides_chi_square_for_small_samples(self):
        # srm_check(1, 7): chi-square p ~ 0.0339 (would FAIL), exact binomial
        # p ~ 0.0703 (should PASS). Decision MUST use the exact test.
        r = srm_check(1, 7, expected_ratio=0.5, alpha=0.05)
        assert r.p_value == pytest.approx(0.0339, rel=0.01)
        assert r.exact_p_value == pytest.approx(0.0703, rel=0.01)
        assert r.decision_method == "exact-binomial"
        assert r.decision_p_value == pytest.approx(0.0703, rel=0.01)
        assert r.passed is True  # exact says pass; chi-square alone would fail
        assert r.small_sample_warning is not None

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="zero total samples"):
            srm_check(0, 0)

    def test_negative_sizes_raise(self):
        with pytest.raises(ValueError, match="negative"):
            srm_check(-1, 100)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            srm_check(100, 100, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            srm_check(100, 100, alpha=0.0)

    def test_invalid_expected_ratio_raises(self):
        with pytest.raises(ValueError, match="expected_ratio"):
            srm_check(100, 100, expected_ratio=0.0)
        with pytest.raises(ValueError, match="expected_ratio"):
            srm_check(100, 100, expected_ratio=1.5)

    def test_small_expected_cell_warning_and_exact_p(self):
        r = srm_check(4, 4, expected_ratio=0.5, alpha=0.05)
        assert r.small_sample_warning is not None
        assert "unreliable" in r.small_sample_warning
        assert r.exact_p_value is not None
        assert 0.0 <= r.exact_p_value <= 1.0

    def test_no_warning_large_sample(self):
        r = srm_check(5000, 5000)
        assert r.small_sample_warning is None
        assert r.exact_p_value is None


# ---------------------------------------------------------------------------
# Newcombe method 10 (Wilson score, square-and-add)
# ---------------------------------------------------------------------------

class TestNewcombeCI:
    def test_wilson_p0_has_nonzero_upper(self):
        lower, upper = _wilson_bounds(0, 5000, 1.96)
        assert lower == pytest.approx(0.0, abs=1e-9)
        assert upper > 0.0

    def test_wilson_p1_has_sub_one_lower(self):
        lower, upper = _wilson_bounds(5000, 5000, 1.96)
        assert upper == pytest.approx(1.0, abs=1e-9)
        assert lower < 1.0

    def test_known_value_50_100_vs_60_100(self):
        # Newcombe method 10 (square-and-add with Wilson), NOT the conservative
        # [L2-U1, U2-L1] bound. Verified against hand calculation:
        # Wilson p1=[0.403832,0.596168], p2=[0.502003,0.690599]
        # lower = 0.1 - sqrt((0.6-0.502003)^2 + (0.596168-0.5)^2) = -0.037302
        # upper = 0.1 + sqrt((0.690599-0.6)^2 + (0.5-0.403832)^2) = 0.232123
        ci_low, ci_high = newcombe_diff_ci(50, 100, 60, 100, alpha=0.05)
        assert ci_low == pytest.approx(-0.037302, abs=1e-5)
        assert ci_high == pytest.approx(0.232123, abs=1e-5)
        # Must be narrower than the conservative [L2-U1, U2-L1] bound
        # (which is approx [-0.094166, 0.286767]).
        assert (ci_high - ci_low) < (0.286767 - (-0.094166))

    def test_invalid_group_sizes_raise(self):
        with pytest.raises(ValueError, match="positive"):
            newcombe_diff_ci(1, 0, 2, 100)
        with pytest.raises(ValueError, match="must be in"):
            newcombe_diff_ci(101, 100, 2, 100)
        with pytest.raises(ValueError, match="must be in"):
            newcombe_diff_ci(1, 100, -1, 100)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            newcombe_diff_ci(10, 100, 20, 100, alpha=0.0)

    def test_degenerate_p1_1_vs_p2_0_is_finite_nonzero_width(self):
        ci_low, ci_high = newcombe_diff_ci(5000, 5000, 0, 5000, alpha=0.05)
        assert math.isfinite(ci_low)
        assert math.isfinite(ci_high)
        assert ci_low < ci_high
        assert ci_high <= 0.0
        # Newcombe method 10 for this case: approx [-1.0, -0.998914]
        assert ci_low == pytest.approx(-1.0, abs=1e-6)
        assert ci_high == pytest.approx(-0.998914, abs=1e-5)

    def test_degenerate_p1_0_vs_p2_1_symmetric(self):
        ci_low, ci_high = newcombe_diff_ci(0, 5000, 5000, 5000, alpha=0.05)
        assert math.isfinite(ci_low)
        assert math.isfinite(ci_high)
        assert ci_low > 0.0

    def test_contains_true_diff_for_known_case(self):
        ci_low, ci_high = newcombe_diff_ci(216, 5000, 283, 5000, alpha=0.05)
        assert ci_low < 0.0134 < ci_high


# ---------------------------------------------------------------------------
# Conversion z-test (with Fisher override for small samples)
# ---------------------------------------------------------------------------

class TestConversionZTest:
    def test_known_case_significant_z_test(self):
        r = conversion_ztest(216, 5000, 283, 5000, alpha=0.05)
        assert r.rate_control == pytest.approx(0.0432)
        assert r.rate_treatment == pytest.approx(0.0566)
        assert r.absolute_lift == pytest.approx(0.0134)
        assert r.relative_lift == pytest.approx(0.310185, rel=1e-4)
        assert r.z_stat == pytest.approx(3.0771, rel=1e-3)
        assert r.p_value == pytest.approx(0.00209, rel=1e-3)
        assert r.min_expected_cell > 5
        assert r.small_sample_warning is None
        assert r.fisher_p_value is None
        assert r.decision_method == "z-test"
        assert r.decision_p_value == pytest.approx(r.p_value)

    def test_fisher_overrides_z_test_for_small_samples(self):
        # 0/5 vs 3/5: min expected cell = 1.5 < 5.
        # z-test p = 0.038434 ("significant") but Fisher p = 0.166667
        # (not significant). Decision MUST use Fisher.
        r = conversion_ztest(0, 5, 3, 5, alpha=0.05)
        assert r.min_expected_cell < 5
        assert r.small_sample_warning is not None
        assert r.fisher_p_value == pytest.approx(0.166667, rel=1e-4)
        assert r.p_value == pytest.approx(0.038434, rel=1e-4)
        assert r.decision_method == "fisher-exact"
        assert r.decision_p_value == pytest.approx(0.166667, rel=1e-4)
        assert r.decision_p_value > 0.05  # NOT significant via Fisher

    def test_ci_level_derived_from_alpha(self):
        r = conversion_ztest(216, 5000, 283, 5000, alpha=0.01)
        r95 = conversion_ztest(216, 5000, 283, 5000, alpha=0.05)
        assert (r.ci_upper - r.ci_lower) > (r95.ci_upper - r95.ci_lower)

    def test_not_significant_case(self):
        r = conversion_ztest(50, 1000, 55, 1000, alpha=0.05)
        assert r.decision_p_value > 0.05
        assert r.decision_method == "z-test"
        assert r.absolute_lift == pytest.approx(0.005)

    def test_empty_group_raises(self):
        with pytest.raises(ValueError, match="positive"):
            conversion_ztest(0, 0, 10, 100)

    def test_invalid_conversion_count_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            conversion_ztest(101, 100, 10, 100)
        with pytest.raises(ValueError, match="must be in"):
            conversion_ztest(10, 100, -1, 100)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            conversion_ztest(10, 100, 20, 100, alpha=2.0)

    def test_degenerate_all_zero(self):
        r = conversion_ztest(0, 100, 0, 100)
        assert r.p_value == 1.0
        assert r.z_stat == 0.0
        assert math.isnan(r.mde_80_power)
        assert math.isnan(r.observed_power)
        assert math.isfinite(r.ci_lower)
        assert math.isfinite(r.ci_upper)

    def test_degenerate_all_one(self):
        r = conversion_ztest(100, 100, 100, 100)
        assert r.p_value == 1.0
        assert math.isnan(r.mde_80_power)
        assert math.isnan(r.observed_power)
        assert math.isfinite(r.ci_lower)
        assert math.isfinite(r.ci_upper)

    def test_degenerate_p1_1_p2_0_no_zero_width_ci(self):
        r = conversion_ztest(5000, 5000, 0, 5000, alpha=0.05)
        assert math.isnan(r.mde_80_power)
        assert math.isnan(r.observed_power)
        assert math.isfinite(r.ci_lower) and math.isfinite(r.ci_upper)
        assert (r.ci_upper - r.ci_lower) > 0.0
        assert r.ci_lower == pytest.approx(-1.0, abs=1e-6)
        assert r.ci_upper == pytest.approx(-0.998914, abs=1e-5)

    def test_degenerate_p1_0_p2_1(self):
        r = conversion_ztest(0, 5000, 5000, 5000, alpha=0.05)
        assert r.decision_p_value < 0.05
        assert math.isnan(r.mde_80_power)
        assert math.isnan(r.observed_power)
        assert (r.ci_upper - r.ci_lower) > 0.0

    def test_relative_lift_nan_when_p1_zero(self):
        r = conversion_ztest(0, 100, 10, 100)
        assert math.isnan(r.relative_lift)


# ---------------------------------------------------------------------------
# CSV loader validation
# ---------------------------------------------------------------------------

class TestCSVLoader:
    def _write(self, tmp_path, content, name="ab.csv"):
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return str(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_validate_csv(str(tmp_path / "nope.csv"))

    def test_empty_file_raises(self, tmp_path):
        p = self._write(tmp_path, "")
        with pytest.raises(ValueError, match="empty"):
            load_and_validate_csv(p)

    def test_header_only_file(self, tmp_path):
        p = self._write(tmp_path, "user_id,variant,converted\n")
        df = load_and_validate_csv(p)
        assert len(df) == 0

    def test_missing_columns_raises(self, tmp_path):
        p = self._write(tmp_path, "user_id,converted\nu1,1\n")
        with pytest.raises(ValueError, match="missing required column"):
            load_and_validate_csv(p)

    def test_missing_user_id_column_raises(self, tmp_path):
        p = self._write(tmp_path, "variant,converted\ncontrol,0\n")
        with pytest.raises(ValueError, match="missing required column"):
            load_and_validate_csv(p)

    def test_non_binary_converted_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1,control,2\nu2,treatment,1\n",
        )
        with pytest.raises(ValueError, match="only 0 or 1"):
            load_and_validate_csv(p)

    def test_fractional_converted_not_truncated(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1,control,0.5\nu2,treatment,1\n",
        )
        with pytest.raises(ValueError, match="only 0 or 1"):
            load_and_validate_csv(p)

    def test_negative_converted_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1,control,-1\nu2,treatment,1\n",
        )
        with pytest.raises(ValueError, match="only 0 or 1"):
            load_and_validate_csv(p)

    def test_non_numeric_converted_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1,control,abc\nu2,treatment,1\n",
        )
        with pytest.raises(ValueError, match="non-numeric"):
            load_and_validate_csv(p)

    def test_duplicate_user_id_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1,control,0\nu1,treatment,1\n",
        )
        with pytest.raises(ValueError, match="duplicate user_id"):
            load_and_validate_csv(p)

    def test_unexpected_variant_raises_with_report(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1,control,0\nu2,holdout,1\n",
        )
        with pytest.raises(ValueError, match="Unexpected variant"):
            load_and_validate_csv(p)

    def test_null_variant_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1,,0\nu2,treatment,1\n",
        )
        with pytest.raises(ValueError, match="Null values"):
            load_and_validate_csv(p)

    def test_null_user_id_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\n,control,0\nu2,treatment,1\n",
        )
        with pytest.raises(ValueError, match="Null values"):
            load_and_validate_csv(p)

    def test_null_converted_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1,control,\nu2,treatment,1\n",
        )
        with pytest.raises(ValueError, match="Null values"):
            load_and_validate_csv(p)

    def test_empty_user_id_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\n   ,control,0\nu2,treatment,1\n",
        )
        with pytest.raises(ValueError, match="empty or whitespace"):
            load_and_validate_csv(p)

    def test_leading_zeros_preserved_as_distinct_ids(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\n001,control,0\n1,treatment,1\n",
        )
        df = load_and_validate_csv(p)
        assert set(df["user_id"]) == {"001", "1"}

    def test_extra_columns_ignored(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted,segment,ts\n"
            "u1,control,0,desktop,2024-01-01\n"
            "u2,treatment,1,mobile,2024-01-01\n",
        )
        df = load_and_validate_csv(p)
        assert set(df.columns) == {"user_id", "variant", "converted"}
        assert len(df) == 2

    def test_valid_csv_passes_and_normalizes(self, tmp_path):
        p = self._write(
            tmp_path,
            "user_id,variant,converted\nu1, Control ,0\nu2,TREATMENT,1\n",
        )
        df = load_and_validate_csv(p)
        assert set(df["variant"].unique()) == {"control", "treatment"}
        assert df["converted"].dtype.kind == "i"
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Prompt labeling (CI confidence level, decision methods, exact counts, NaN)
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def _results(self, alpha=0.05):
        srm = srm_check(5000, 5000, expected_ratio=0.5, alpha=0.05)
        conv = conversion_ztest(216, 5000, 283, 5000, alpha=alpha)
        return srm, conv

    def test_ci_label_matches_alpha_95(self):
        srm, conv = self._results(alpha=0.05)
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=216, conv_treatment=283)
        assert "95% CI" in msg

    def test_ci_label_matches_alpha_99(self):
        srm, conv = self._results(alpha=0.01)
        msg = build_user_message(srm, conv, alpha=0.01, srm_alpha=0.05,
                                 conv_control=216, conv_treatment=283)
        assert "99% CI" in msg
        assert "95% CI" not in msg

    def test_ci_label_matches_alpha_90(self):
        srm, conv = self._results(alpha=0.10)
        msg = build_user_message(srm, conv, alpha=0.10, srm_alpha=0.05,
                                 conv_control=216, conv_treatment=283)
        assert "90% CI" in msg

    def test_exact_counts_not_reconstructed(self):
        srm, conv = self._results()
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=216, conv_treatment=283)
        assert "(216 / 5000)" in msg
        assert "(283 / 5000)" in msg

    def test_power_labeled_post_hoc_not_decisive(self):
        srm, conv = self._results()
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=216, conv_treatment=283)
        assert "post-hoc" in msg
        assert "not decisive" in msg

    def test_srm_alpha_shown_separately(self):
        srm, conv = self._results()
        msg = build_user_message(srm, conv, alpha=0.01, srm_alpha=0.05,
                                 conv_control=216, conv_treatment=283)
        assert "SRM (Sample Ratio Mismatch) Check (alpha=0.05)" in msg
        assert "alpha=0.01" in msg

    def test_srm_decision_method_reflected_in_prompt(self):
        # srm_check(1, 7): exact binomial overrides chi-square for the decision
        srm = srm_check(1, 7, expected_ratio=0.5, alpha=0.05)
        conv = conversion_ztest(1, 8, 2, 8, alpha=0.05)
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=1, conv_treatment=2)
        assert "SRM DECISION method: exact-binomial" in msg
        assert "SRM PASSED (no mismatch): True" in msg

    def test_fisher_decision_reflected_in_prompt(self):
        srm = srm_check(5, 5, expected_ratio=0.5, alpha=0.05)
        conv = conversion_ztest(0, 5, 3, 5, alpha=0.05)
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=0, conv_treatment=3)
        assert "CONVERSION DECISION method: fisher-exact" in msg
        assert "via fisher-exact" in msg
        assert "0.166667" in msg

    def test_nan_rendered_as_na(self):
        srm = srm_check(100, 100, expected_ratio=0.5, alpha=0.05)
        conv = conversion_ztest(0, 100, 0, 100, alpha=0.05)
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=0, conv_treatment=0)
        assert "N/A" in msg
        assert "nan%" not in msg
        # No bare 'nan' token outside of legitimate words like 'significance'
        import re
        # strip words that legitimately contain 'nan' as a substring
        cleaned = re.sub(r"significance|significant", "", msg, flags=re.IGNORECASE)
        assert "nan" not in cleaned.lower()

    def test_srm_warning_and_exact_p_shown(self):
        srm = srm_check(4, 4, expected_ratio=0.5, alpha=0.05)
        conv = conversion_ztest(1, 8, 2, 8, alpha=0.05)
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=1, conv_treatment=2)
        assert "WARNING" in msg
        assert "Exact binomial test p-value" in msg

    def test_min_effect_threshold_included(self):
        srm, conv = self._results()
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=216, conv_treatment=283,
                                 min_effect=0.01)
        assert "Min-effect threshold analysis" in msg
        assert "Minimum effect threshold" in msg

    def test_min_effect_absent_when_not_provided(self):
        srm, conv = self._results()
        msg = build_user_message(srm, conv, alpha=0.05, srm_alpha=0.05,
                                 conv_control=216, conv_treatment=283)
        assert "Min-effect threshold analysis" not in msg


# ---------------------------------------------------------------------------
# End-to-end pipeline (offline / --skip-api) using fixtures
# ---------------------------------------------------------------------------

class TestPipelineOffline:
    def test_skip_api_with_fixture(self, known_csv):
        out = run_pipeline(
            known_csv,
            expected_ratio=0.5, alpha=0.05, srm_alpha=0.05,
            skip_api=True, model="claude-sonnet-4-5",
        )
        assert "LOCAL MODE" in out
        assert "SRM DECISION method: chi-square" in out
        assert "CONVERSION DECISION method: z-test" in out
        assert "(10 / 100)" in out
        assert "(20 / 100)" in out

    def test_skip_api_fisher_override_fixture(self, small_csv):
        out = run_pipeline(
            small_csv,
            expected_ratio=0.5, alpha=0.05, srm_alpha=0.05,
            skip_api=True, model="claude-sonnet-4-5",
        )
        assert "CONVERSION DECISION method: fisher-exact" in out
        assert "via fisher-exact" in out

    def test_srm_alpha_independent_of_conversion_alpha(self, known_csv):
        out = run_pipeline(
            known_csv,
            expected_ratio=0.5, alpha=0.01, srm_alpha=0.05,
            skip_api=True, model="claude-sonnet-4-5",
        )
        assert "SRM (Sample Ratio Mismatch) Check (alpha=0.05)" in out
        assert "alpha=0.01" in out

    def test_missing_arm_raises(self, tmp_path):
        p = tmp_path / "one_arm.csv"
        p.write_text(
            "user_id,variant,converted\nu1,control,0\nu2,control,1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="treatment"):
            run_pipeline(str(p), 0.5, 0.05, 0.05, True, "claude-sonnet-4-5")

    def test_bundled_dataset_end_to_end(self):
        # One end-to-end check against the bundled dataset (regenerate with
        # `python generate_mock_data.py --overwrite` if missing).
        out = run_pipeline(
            "data/mock_ab_test.csv",
            expected_ratio=0.5, alpha=0.05, srm_alpha=0.05,
            skip_api=True, model="claude-sonnet-4-5",
        )
        assert "LOCAL MODE" in out
        assert "SRM DECISION method: chi-square" in out
        assert "CONVERSION DECISION method: z-test" in out
