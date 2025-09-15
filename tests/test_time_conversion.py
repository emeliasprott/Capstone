import datetime
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from GNN.time_utils import convert_to_utc_seconds


def test_year_only_strings_use_midyear_timestamp():
    timestamps = ["1999", "2005"]
    result = convert_to_utc_seconds(timestamps)

    expected = [
        datetime.datetime(1999, 6, 15, tzinfo=datetime.timezone.utc).timestamp(),
        datetime.datetime(2005, 6, 15, tzinfo=datetime.timezone.utc).timestamp(),
    ]

    assert result == pytest.approx(expected)


def test_epoch_seconds_pass_through_unchanged():
    epoch_values = [1_600_000_000.0, 1_620_000_000.5]
    result = convert_to_utc_seconds(epoch_values)

    assert result == pytest.approx(epoch_values)


def test_nan_inputs_fall_back_to_default_timestamp():
    timestamps = [float("nan"), 1_600_000_000.0]
    result = convert_to_utc_seconds(timestamps)

    fallback = datetime.datetime(2000, 6, 15, tzinfo=datetime.timezone.utc).timestamp()

    assert result[0] == pytest.approx(fallback)
    assert result[1] == pytest.approx(1_600_000_000.0)
