"""
Automated tests for the synthetic A/B test data generator.

Run with:
    python -m pytest test_generate.py -v

Covers: parameter validation (including non-numeric types raising ValueError),
atomic write behavior, neutral user IDs, reproducibility with seed, and
CLI error handling.
"""

import csv

import pytest

from generate_mock_data import generate


@pytest.fixture
def out(tmp_path):
    return tmp_path / "ab.csv"


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_basic_generation(out):
    summary = generate(
        out, n_control=100, n_treatment=100,
        p_control=0.5, p_treatment=0.5, seed=1, overwrite=True,
    )
    assert summary["total"] == 200
    assert summary["n_control"] == 100
    assert summary["n_treatment"] == 100
    rows = _read_rows(out)
    assert rows[0] == ["user_id", "variant", "converted"]
    assert len(rows) == 201


def test_file_exists_without_overwrite_raises(out):
    generate(out, n_control=10, n_treatment=10, seed=1, overwrite=True)
    with pytest.raises(FileExistsError):
        generate(out, n_control=10, n_treatment=10, seed=1, overwrite=False)


def test_non_numeric_n_raises_value_error(out):
    with pytest.raises(ValueError):
        generate(out, n_control="abc", n_treatment=10, seed=1, overwrite=True)
    with pytest.raises(ValueError):
        generate(out, n_control=10.5, n_treatment=10, seed=1, overwrite=True)
    with pytest.raises(ValueError):
        generate(out, n_control=10, n_treatment=None, seed=1, overwrite=True)
    assert not out.exists()  # validation before any file I/O


def test_non_numeric_p_raises_value_error(out):
    with pytest.raises(ValueError):
        generate(out, p_control="high", seed=1, overwrite=True)
    with pytest.raises(ValueError):
        generate(out, p_treatment=[0.5], seed=1, overwrite=True)
    assert not out.exists()


def test_negative_n_raises(out):
    with pytest.raises(ValueError, match="positive"):
        generate(out, n_control=-1, n_treatment=10, seed=1, overwrite=True)
    assert not out.exists()


def test_zero_n_raises(out):
    with pytest.raises(ValueError, match="positive"):
        generate(out, n_control=10, n_treatment=0, seed=1, overwrite=True)
    assert not out.exists()


def test_p_out_of_range_raises(out):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        generate(out, p_control=1.5, seed=1, overwrite=True)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        generate(out, p_treatment=-0.1, seed=1, overwrite=True)
    assert not out.exists()


def test_conversion_counts_respect_targets(out):
    summary = generate(
        out, n_control=1000, n_treatment=1000,
        p_control=0.10, p_treatment=0.20, seed=42, overwrite=True,
    )
    # Realized rates should be in the neighborhood of the targets.
    assert 0.05 <= summary["rate_control"] <= 0.15
    assert 0.15 <= summary["rate_treatment"] <= 0.25


def test_neutral_ids_do_not_leak_arm(out):
    generate(
        out, n_control=50, n_treatment=50, seed=7, overwrite=True,
    )
    rows = _read_rows(out)
    header, data = rows[0], rows[1:]
    id_idx = header.index("user_id")
    variant_idx = header.index("variant")
    for row in data:
        uid = row[id_idx]
        variant = row[variant_idx]
        # IDs are neutral uNNNNNN; the digits must NOT encode the arm.
        assert uid.startswith("u")
        digits = uid[1:]
        assert digits.isdigit()
        assert variant not in uid.lower()


def test_ids_unique_and_sequential(out):
    generate(out, n_control=25, n_treatment=25, seed=3, overwrite=True)
    rows = _read_rows(out)
    ids = [r[0] for r in rows[1:]]
    assert len(set(ids)) == 50
    assert ids == [f"u{i:06d}" for i in range(50)]


def test_reproducible_with_same_seed(out, tmp_path):
    other = tmp_path / "ab2.csv"
    generate(out, n_control=100, n_treatment=100, seed=99, overwrite=True)
    generate(other, n_control=100, n_treatment=100, seed=99, overwrite=True)
    assert out.read_text(encoding="utf-8") == other.read_text(encoding="utf-8")


def test_different_seed_differs(out, tmp_path):
    other = tmp_path / "ab3.csv"
    generate(out, n_control=100, n_treatment=100, seed=1, overwrite=True)
    generate(other, n_control=100, n_treatment=100, seed=2, overwrite=True)
    assert out.read_text(encoding="utf-8") != other.read_text(encoding="utf-8")


def test_parent_directory_created(tmp_path):
    target = tmp_path / "nested" / "dir" / "ab.csv"
    summary = generate(target, n_control=10, n_treatment=10, seed=5,
                       overwrite=True)
    assert target.exists()
    assert summary["path"] == str(target)


def test_overwrite_replaces_file(out):
    generate(out, n_control=10, n_treatment=10, seed=1, overwrite=True)
    first = out.read_bytes()
    generate(out, n_control=10, n_treatment=10, seed=2, overwrite=True)
    second = out.read_bytes()
    assert first != second


def test_summary_rates_are_floats_in_unit_interval(out):
    summary = generate(out, n_control=100, n_treatment=100, seed=11,
                       overwrite=True)
    assert isinstance(summary["rate_control"], float)
    assert isinstance(summary["rate_treatment"], float)
    assert 0.0 <= summary["rate_control"] <= 1.0
    assert 0.0 <= summary["rate_treatment"] <= 1.0


def test_cli_error_on_existing_file_without_overwrite(out, capsys):
    generate(out, n_control=10, n_treatment=10, seed=1, overwrite=True)
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "generate_mock_data.py",
         "--output", str(out), "--n-control", "10", "--n-treatment", "10"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Error" in result.stderr
    assert "already exists" in result.stderr


def test_cli_generation_smoke(tmp_path):
    import subprocess
    import sys
    target = tmp_path / "cli.csv"
    result = subprocess.run(
        [sys.executable, "generate_mock_data.py",
         "--output", str(target),
         "--n-control", "20", "--n-treatment", "20", "--seed", "5"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Wrote 40 rows" in result.stdout
    assert target.exists()
