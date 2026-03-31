"""Tests for StatCanAdapter (no live network calls)."""

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aieng.forecasting.data.adapters.statcan import (
    StatCanAdapter,
    _normalize_table_id,
    _read_zip,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_raw_statcan_df() -> pd.DataFrame:
    """Minimal mock of a stats-can table DataFrame (post-zip-read, pre-filter).

    Mimics the structure of a CPI table with two geographies and two product
    groups. REF_DATE is already parsed to datetime64, matching what _read_zip
    produces.
    """
    return pd.DataFrame(
        {
            "REF_DATE": pd.to_datetime(
                ["2022-01", "2022-02", "2022-01", "2022-02", "2022-01", "2022-02"]
            ),
            "GEO": ["Canada", "Canada", "Canada", "Canada", "Ontario", "Ontario"],
            "Products and product groups": [
                "All-items",
                "All-items",
                "Food",
                "Food",
                "All-items",
                "All-items",
            ],
            "VALUE": [151.2, 152.4, 165.3, 166.1, 148.0, 149.5],
            "STATUS": ["", "", "", "", "", ""],
        }
    )


def _make_zip_bytes(normalized_id: str, df: pd.DataFrame) -> bytes:
    """Return in-memory bytes of a StatCan-style zip containing a CSV."""
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr(f"{normalized_id}.csv", csv_buf.getvalue())
    return zip_buf.getvalue()


@pytest.fixture()
def adapter(tmp_path: Path) -> StatCanAdapter:
    """Return a StatCanAdapter configured for All-items Canada."""
    return StatCanAdapter(
        table_id="18-10-0004-13",
        member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
        cache_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# _normalize_table_id
# ---------------------------------------------------------------------------

class TestNormalizeTableId:
    def test_strips_hyphens_and_truncates(self) -> None:
        assert _normalize_table_id("18-10-0004-13") == "18100004"

    def test_handles_plain_digits(self) -> None:
        assert _normalize_table_id("18100004") == "18100004"

    def test_handles_short_id(self) -> None:
        assert _normalize_table_id("1234") == "1234"


# ---------------------------------------------------------------------------
# _read_zip
# ---------------------------------------------------------------------------

class TestReadZip:
    """Tests for the private _read_zip helper (reads CSV from zip bytes)."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        raw = pd.DataFrame(
            {"REF_DATE": ["2022-01", "2022-02"], "VALUE": [100.0, 101.0]}
        )
        zip_path = tmp_path / "18100004-eng.zip"
        zip_path.write_bytes(_make_zip_bytes("18100004", raw))

        result = _read_zip(zip_path, "18100004")
        assert isinstance(result, pd.DataFrame)
        assert "REF_DATE" in result.columns
        assert "VALUE" in result.columns

    def test_parses_ref_date_as_datetime(self, tmp_path: Path) -> None:
        raw = pd.DataFrame(
            {"REF_DATE": ["2022-01", "2022-02"], "VALUE": [100.0, 101.0]}
        )
        zip_path = tmp_path / "18100004-eng.zip"
        zip_path.write_bytes(_make_zip_bytes("18100004", raw))

        result = _read_zip(zip_path, "18100004")
        assert pd.api.types.is_datetime64_any_dtype(result["REF_DATE"])


# ---------------------------------------------------------------------------
# StatCanAdapter properties
# ---------------------------------------------------------------------------

class TestStatCanAdapterProperties:
    def test_table_id(self, adapter: StatCanAdapter) -> None:
        assert adapter.table_id == "18-10-0004-13"

    def test_member_filter_is_copy(self, adapter: StatCanAdapter) -> None:
        f1 = adapter.member_filter
        f1["GEO"] = "Ontario"
        assert adapter.member_filter["GEO"] == "Canada"


# ---------------------------------------------------------------------------
# StatCanAdapter.fetch() — all network/disk I/O is mocked
# ---------------------------------------------------------------------------

_MODULE = "aieng.forecasting.data.adapters.statcan"


class TestStatCanAdapterFetch:
    """Tests for StatCanAdapter.fetch() with mocked zip reading.

    The zip file existence check and the download step are mocked so no
    network calls or disk I/O occur. _read_zip is patched to return a
    controlled DataFrame.
    """

    def _patch_fetch(
        self, raw_df: pd.DataFrame, zip_exists: bool = True
    ) -> tuple[MagicMock, MagicMock]:
        """Return (mock for _read_zip, mock for sc.download_tables).

        Use as context managers in test methods.
        """
        mock_read = MagicMock(return_value=raw_df)
        mock_sc = MagicMock()
        return mock_read, mock_sc

    def test_fetch_returns_correct_shape(self, adapter: StatCanAdapter) -> None:
        """fetch() returns a DataFrame with timestamp and value columns."""
        raw = _make_raw_statcan_df()
        with (
            patch(f"{_MODULE}._read_zip", return_value=raw),
            patch(f"{_MODULE}.Path.exists", return_value=True),
        ):
            result = adapter.fetch()

        assert set(result.columns) == {"timestamp", "value"}
        assert len(result) == 2

    def test_fetch_filters_correctly(self, adapter: StatCanAdapter) -> None:
        """fetch() returns only rows matching member_filter."""
        raw = _make_raw_statcan_df()
        with (
            patch(f"{_MODULE}._read_zip", return_value=raw),
            patch(f"{_MODULE}.Path.exists", return_value=True),
        ):
            result = adapter.fetch()

        assert list(result["value"]) == [151.2, 152.4]

    def test_fetch_sorted_by_timestamp(self, adapter: StatCanAdapter) -> None:
        """fetch() returns rows sorted ascending by timestamp."""
        raw = _make_raw_statcan_df().iloc[::-1].reset_index(drop=True)
        with (
            patch(f"{_MODULE}._read_zip", return_value=raw),
            patch(f"{_MODULE}.Path.exists", return_value=True),
        ):
            result = adapter.fetch()

        assert result["timestamp"].is_monotonic_increasing

    def test_fetch_drops_nan_values(self, adapter: StatCanAdapter) -> None:
        """fetch() drops rows where VALUE is NaN."""
        raw = _make_raw_statcan_df()
        raw.loc[raw["REF_DATE"] == pd.Timestamp("2022-02-01"), "VALUE"] = float("nan")
        with (
            patch(f"{_MODULE}._read_zip", return_value=raw),
            patch(f"{_MODULE}.Path.exists", return_value=True),
        ):
            result = adapter.fetch()

        assert len(result) == 1
        assert result["value"].iloc[0] == 151.2

    def test_fetch_raises_on_missing_filter_column(self, tmp_path: Path) -> None:
        """fetch() raises ValueError when a filter column is absent from the table."""
        raw = pd.DataFrame(
            {"REF_DATE": pd.to_datetime(["2022-01"]), "VALUE": [100.0]}
        )
        bad_adapter = StatCanAdapter(
            table_id="18-10-0004-13",
            member_filter={"GEO": "Canada"},
            cache_dir=tmp_path,
        )
        with (
            patch(f"{_MODULE}._read_zip", return_value=raw),
            patch(f"{_MODULE}.Path.exists", return_value=True),
        ):
            with pytest.raises(ValueError, match="GEO"):
                bad_adapter.fetch()

    def test_fetch_raises_when_no_rows_match(self, tmp_path: Path) -> None:
        """fetch() raises RuntimeError when filter matches zero rows."""
        raw = _make_raw_statcan_df()
        bad_adapter = StatCanAdapter(
            table_id="18-10-0004-13",
            member_filter={"GEO": "Narnia"},
            cache_dir=tmp_path,
        )
        with (
            patch(f"{_MODULE}._read_zip", return_value=raw),
            patch(f"{_MODULE}.Path.exists", return_value=True),
        ):
            with pytest.raises(RuntimeError, match="No rows matched"):
                bad_adapter.fetch()

    def test_fetch_triggers_download_when_zip_missing(
        self, tmp_path: Path
    ) -> None:
        """fetch() calls download_tables when the zip is absent."""
        raw = _make_raw_statcan_df()
        adapter_with_tmp = StatCanAdapter(
            table_id="18-10-0004-13",
            member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
            cache_dir=tmp_path,
        )
        with (
            patch(f"{_MODULE}._read_zip", return_value=raw),
            patch("stats_can.sc.download_tables") as mock_download,
        ):
            adapter_with_tmp.fetch()

        mock_download.assert_called_once()

    def test_fetch_raises_on_download_error(self, tmp_path: Path) -> None:
        """fetch() wraps download errors in RuntimeError."""
        adapter_with_tmp = StatCanAdapter(
            table_id="18-10-0004-13",
            member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
            cache_dir=tmp_path,
        )
        with patch(
            "stats_can.sc.download_tables",
            side_effect=ConnectionError("network down"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download"):
                adapter_with_tmp.fetch()
