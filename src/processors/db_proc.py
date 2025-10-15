from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import os
import re


# -----------------------------
# System configuration (project convention)
# -----------------------------


@dataclass
class SystemConfig:
    """Pipeline configuration.

    Note: maintain instance via singleton helper `get_system_config()`.
    """

    raw_csv_path: str = "data/raw/raw_db.csv"

_SYSTEM_CONFIG_SINGLETON: Optional[SystemConfig] = None


def get_system_config() -> SystemConfig:
    global _SYSTEM_CONFIG_SINGLETON
    if _SYSTEM_CONFIG_SINGLETON is None:
        _SYSTEM_CONFIG_SINGLETON = SystemConfig()
    return _SYSTEM_CONFIG_SINGLETON


# -----------------------------
# Main processor
# -----------------------------


class DataProcessor:
    """Preprocessing pipeline for `data/raw/raw_db.csv`.

    Main responsibilities:
    - Read raw CSV
    - Robust numeric parsing of the requested fields
    - Relative/binary calculations (high precision)
    - Coherence checks between binary and relative values
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        *,
        zero_threshold: float = 1e-12,
        coherence_tolerance: float = 1e-8,
    ) -> None:
        self.config = config or get_system_config()
        self.zero_threshold = float(zero_threshold)
        self.coherence_tolerance = float(coherence_tolerance)

        # Required columns for parsing and checks
        self.required_columns = [
            "baseline_perf",
            "new_perf",
            "perf_binary",
            "relative_perf",
            "baseline_cost",
            "new_cost",
            "cost_binary",
            "relative_cost",
            "baseline_method_std",
            "new_method_std",
            "metadata_perf_type",
            "paper_year",
            "paper_domain",
            "paper_granular_domain",
        ]

    # -----------------------------
    # Loading and numeric parsing
    # -----------------------------

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.raw_csv_path)
        return df

    def _normalize_numeric_str(self, value: object) -> Optional[str]:
        """Clean common numeric strings: spaces, commas, underscores, unicode signs.

        Returns a cleaned string or None for empty/clearly non-numeric values.
        """
        if pd.isna(value):
            return None
        if isinstance(value, (int, float, np.floating, np.integer)):
            return str(value)
        if not isinstance(value, str):
            return None

        s = value.strip()
        if s == "":
            return None

        # Common normalizations
        s = s.replace("\u2212", "-")  # unicode minus → '-'
        s = s.replace(",", "")  # remove thousands separator
        s = s.replace("_", "")
        s = s.replace("%", "")  # if percentage appears, treat as raw number
        s = s.replace("$", "")  # if cost comes with symbol

        # Handle strings like 'nan', 'inf', etc.
        sl = s.lower()
        if sl in {"nan", "none", "null", "", "na"}:
            return None
        if sl in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            return None

        return s

    def _parse_numeric_column(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """Convert a column to numeric (float) values, robust to various notations.

        - Accept scientific notation
        - Remove superfluous characters
        - Coerce failures to NaN
        """
        if column_name not in df.columns:
            raise ValueError(f"Coluna ausente no dataset: {column_name}")

        cleaned = df[column_name].apply(self._normalize_numeric_str)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        return numeric.astype(float)

    def _ensure_required_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                "Missing required columns in CSV: " + ", ".join(missing)
            )


    def _normalize_category_text(self, text: object) -> object:
        """Normalize category tokens.

        Rules:
        - Replace any standalone occurrences (case-insensitive) of ML or SSM with TS
        - Replace any standalone occurrences (case-insensitive) of DL with AI
        - Keep SC unchanged
        Works for strings that may include multiple tokens separated by punctuation/space.
        """
        if pd.isna(text):
            return text
        s = str(text)
        # Map DL -> AI first, ML/SSM -> TS
        s = re.sub(r"\bDL\b", "AI", s, flags=re.IGNORECASE)
        s = re.sub(r"\bML\b", "TS", s, flags=re.IGNORECASE)
        s = re.sub(r"\bSSM\b", "TS", s, flags=re.IGNORECASE)
        return s

    def _normalize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to baseline/new categories and comparison_type."""
        cols = [c for c in ["baseline_category", "new_category", "comparison_type"] if c in df.columns]
        for c in cols:
            df[c] = df[c].apply(self._normalize_category_text)
        return df

    def _normalize_scope_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize modeling scope fields to project-standard labels.

        - full_surrogate -> replacement
        - partial_surrogate -> hybrid
        Applies to columns: modeling_scope, modeling_benchmark_type (if present).
        """
        def _map_scope(value: object) -> object:
            if pd.isna(value):
                return value
            s = str(value)
            s = re.sub(r"\bfull_surrogate\b", "replacement", s, flags=re.IGNORECASE)
            s = re.sub(r"\bpartial_surrogate\b", "hybrid", s, flags=re.IGNORECASE)
            return s

        for c in ["modeling_scope", "modeling_benchmark_type"]:
            if c in df.columns:
                df[c] = df[c].apply(_map_scope)
        return df

    def _sign_category(self, value: float) -> str:
        """Categorize sign of a numeric value with tolerance-aware zero.

        Returns one of: 'positive' | 'negative' | 'zero' | 'nan'.
        """
        if pd.isna(value):
            return "nan"
        if abs(value) <= self.coherence_tolerance:
            return "zero"
        return "positive" if value > 0 else "negative"

    # -----------------------------
    # Requested calculations
    # -----------------------------

    def _calculate_relative(self, baseline: float, new: float, metric_type: str) -> Optional[float]:
        """Calculate relative performance/cost value with high precision."""
        if pd.isna(baseline) or pd.isna(new):
            return np.nan

        # Handle true zero baseline (but not small scientific notation values)
        if abs(baseline) < self.zero_threshold:
            if abs(new) < self.zero_threshold:
                return 0.0  # Both effectively zero
            else:
                return np.nan  # Can't calculate meaningful ratio

        # Use numpy's high precision functions for better accuracy
        try:
            # Calculate ratio with maximum precision
            ratio = np.float64(new) / np.float64(baseline)

            if ratio <= 0:
                return np.nan

            # Use numpy's log function which is optimized for precision
            log_ratio = np.log10(ratio)

            # Calculate relative value based on metric type
            if metric_type in ['PL', 'CL']:  # Larger is better/more expensive
                return float(log_ratio)
            elif metric_type in ['PS', 'CS']:  # Smaller is better/more expensive
                return float(-log_ratio)  # Flip sign: smaller new value = better = negative relative
            else:
                return np.nan

        except (ValueError, ArithmeticError, OverflowError):
            return np.nan

    def _calculate_binary(self, baseline: float, new: float, metric_type: str) -> Optional[int]:
        """
        Calculate binary comparison value

        Key fix: Don't artificially inflate small baseline values with epsilon.
        """
        if pd.isna(baseline) or pd.isna(new):
            return np.nan

        # Handle true zeros (but not small scientific notation values)
        if abs(baseline) < self.zero_threshold and abs(new) < self.zero_threshold:
            return 0

        # Check if values are effectively equal using relative comparison
        if abs(baseline) > self.zero_threshold:
            relative_diff = abs(new - baseline) / abs(baseline)
            if relative_diff < 1e-10:  # Very small relative difference
                return 0
        elif abs(new - baseline) < self.zero_threshold:
            return 0

        # Determine better/worse based on metric type - NO epsilon adjustment
        if metric_type == 'PL':  # Performance: Larger is better
            return 1 if new > baseline else -1
        elif metric_type == 'PS':  # Performance: Smaller is better
            return 1 if new < baseline else -1
        elif metric_type == 'CL':  # Cost: Larger is more expensive (worse)
            return 1 if new > baseline else -1
        elif metric_type == 'CS':  # Cost: Smaller is more expensive (worse)
            return 1 if new < baseline else -1  # If new < baseline, new is more expensive (worse)
        else:
            return np.nan

    def _check_coherence(self, binary_val: float, relative_val: float) -> Tuple[bool, str]:
        """Check if binary and relative values are coherent."""
        if pd.isna(binary_val) or pd.isna(relative_val):
            return True, ""

        binary_val = int(binary_val)

        # Check coherence based on binary value
        if binary_val == 1:  # Should have positive relative value
            if relative_val > self.coherence_tolerance:
                return True, ""
            elif relative_val > 0:
                # Positive but below threshold - likely should be binary=0
                pct_change = (np.exp(relative_val) - 1) * 100
                return False, f"binary=1 but relative={relative_val:.4f} ({pct_change:.2f}% change, likely should be binary=0)"
            else:
                return False, f"binary=1 but relative={relative_val:.4f} (negative, contradicts improvement)"

        elif binary_val == -1:  # Should have negative relative value
            if relative_val < -self.coherence_tolerance:
                return True, ""
            elif relative_val < 0:
                # Negative but above threshold - likely should be binary=0
                pct_change = (1 - np.exp(relative_val)) * 100
                return False, f"binary=-1 but relative={relative_val:.4f} ({pct_change:.2f}% degradation, likely should be binary=0)"
            else:
                return False, f"binary=-1 but relative={relative_val:.4f} (positive, contradicts degradation)"

        elif binary_val == 0:  # Should have near-zero relative value
            if abs(relative_val) <= self.coherence_tolerance:
                return True, ""
            else:
                pct_change = (np.exp(abs(relative_val)) - 1) * 100
                direction = "improvement" if relative_val > 0 else "degradation"
                return False, f"binary=0 but relative={relative_val:.4f} ({pct_change:.2f}% {direction}, exceeds threshold)"

        else:
            return False, f"invalid binary value: {binary_val}"

    # -----------------------------
    # Pipeline orchestration
    # -----------------------------

    def process(self) -> pd.DataFrame:
        """Run the pipeline: read, validate, parse, and check coherence.

        Returns a clean DataFrame with numeric columns and coherence flags/notes.
        """
        df = self._load_raw()
        self._ensure_required_columns(df)

        # Robust numeric parsing
        df["baseline_perf"] = self._parse_numeric_column(df, "baseline_perf")
        df["new_perf"] = self._parse_numeric_column(df, "new_perf")
        df["perf_binary"] = self._parse_numeric_column(df, "perf_binary")
        df["relative_perf"] = self._parse_numeric_column(df, "relative_perf")

        df["baseline_cost"] = self._parse_numeric_column(df, "baseline_cost")
        df["new_cost"] = self._parse_numeric_column(df, "new_cost")
        df["cost_binary"] = self._parse_numeric_column(df, "cost_binary")
        df["relative_cost"] = self._parse_numeric_column(df, "relative_cost")

        # Parse paper_year as integer
        df["paper_year"] = self._parse_numeric_column(df, "paper_year").astype('Int64')

        # Normalize categories and comparison types
        df = self._normalize_categories(df)

        # Normalize modeling scope fields
        df = self._normalize_scope_fields(df)

        # Coherence: performance and cost
        perf_checks = df.apply(
            lambda r: self._check_coherence(r.get("perf_binary"), r.get("relative_perf")), axis=1
        )
        df["perf_coherence_ok"] = [ok for ok, _ in perf_checks]
        df["perf_coherence_note"] = [note for _, note in perf_checks]

        cost_checks = df.apply(
            lambda r: self._check_coherence(r.get("cost_binary"), r.get("relative_cost")), axis=1
        )
        df["cost_coherence_ok"] = [ok for ok, _ in cost_checks]
        df["cost_coherence_note"] = [note for _, note in cost_checks]

        # Helper columns for easier filtering/analysis
        binary_to_expected = {1: "positive", 0: "zero", -1: "negative"}

        # Performance helper columns
        df["perf_expected"] = df["perf_binary"].apply(
            lambda v: binary_to_expected.get(int(v), "invalid") if not pd.isna(v) else "nan"
        )
        df["perf_relative_sign"] = df["relative_perf"].apply(self._sign_category)
        df["perf_rel_abs"] = df["relative_perf"].abs()
        df["perf_rel_exceeds_tol"] = df["perf_rel_abs"] > self.coherence_tolerance

        def _perf_category(row: pd.Series) -> str:
            b, rel = row.get("perf_binary"), row.get("relative_perf")
            ok = row.get("perf_coherence_ok")
            if pd.isna(b) or pd.isna(rel):
                return "unknown"
            if ok:
                return "ok"
            b = int(b)
            sign = self._sign_category(rel)
            if b == 0:
                return "zero_but_large" if abs(rel) > self.coherence_tolerance else "ok"
            if b == 1:
                if sign == "negative":
                    return "contradiction_negative"
                if sign == "positive" and abs(rel) <= self.coherence_tolerance:
                    return "below_threshold_should_be_zero"
                return "incoherent"
            if b == -1:
                if sign == "positive":
                    return "contradiction_positive"
                if sign == "negative" and abs(rel) <= self.coherence_tolerance:
                    return "below_threshold_should_be_zero"
                return "incoherent"
            return "invalid_binary"

        df["perf_coherence_category"] = df.apply(_perf_category, axis=1)

        # Cost helper columns
        df["cost_expected"] = df["cost_binary"].apply(
            lambda v: binary_to_expected.get(int(v), "invalid") if not pd.isna(v) else "nan"
        )
        df["cost_relative_sign"] = df["relative_cost"].apply(self._sign_category)
        df["cost_rel_abs"] = df["relative_cost"].abs()
        df["cost_rel_exceeds_tol"] = df["cost_rel_abs"] > self.coherence_tolerance

        def _cost_category(row: pd.Series) -> str:
            b, rel = row.get("cost_binary"), row.get("relative_cost")
            ok = row.get("cost_coherence_ok")
            if pd.isna(b) or pd.isna(rel):
                return "unknown"
            if ok:
                return "ok"
            b = int(b)
            sign = self._sign_category(rel)
            if b == 0:
                return "zero_but_large" if abs(rel) > self.coherence_tolerance else "ok"
            if b == 1:
                if sign == "negative":
                    return "contradiction_negative"
                if sign == "positive" and abs(rel) <= self.coherence_tolerance:
                    return "below_threshold_should_be_zero"
                return "incoherent"
            if b == -1:
                if sign == "positive":
                    return "contradiction_positive"
                if sign == "negative" and abs(rel) <= self.coherence_tolerance:
                    return "below_threshold_should_be_zero"
                return "incoherent"
            return "invalid_binary"

        df["cost_coherence_category"] = df.apply(_cost_category, axis=1)

        # Rename method columns to remove "_std" suffix for final output
        df = df.rename(columns={
            "baseline_method_std": "baseline_method",
            "new_method_std": "new_method"
        })

        return df


__all__ = ["DataProcessor", "SystemConfig", "get_system_config"]


def _run_and_save() -> None:
    """Run processing and save output to data/processed/processed_db.csv with summary."""
    dp = DataProcessor()
    df = dp.process()

    # Ensure output directory exists
    output_path = os.path.join("data", "processed", "processed_db.csv")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    num_rows = len(df)
    num_perf_issues = int((~df["perf_coherence_ok"]).sum())
    num_cost_issues = int((~df["cost_coherence_ok"]).sum())
    print(
        "Processing complete.\n"
        f"  Saved: {output_path}\n"
        f"  Rows: {num_rows}\n"
        f"  Performance incoherent: {num_perf_issues}\n"
        f"  Cost incoherent: {num_cost_issues}"
    )


if __name__ == "__main__":
    _run_and_save()

