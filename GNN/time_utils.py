"""Utilities for normalizing heterogeneous timestamp inputs."""

from __future__ import annotations

import datetime
import math
from typing import Iterable, List

try:  # Optional dependency used only when available.
    import numpy as np
except ImportError:  # pragma: no cover - numpy is present in production environment.
    np = None  # type: ignore[assignment]

try:  # Optional dependency used only when available.
    import torch
except ImportError:  # pragma: no cover - torch is present in production environment.
    torch = None  # type: ignore[assignment]


_FALLBACK_TIMESTAMP = datetime.datetime(2000, 6, 15, tzinfo=datetime.timezone.utc).timestamp()


def _datetime_to_utc_seconds(dt: datetime.datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.timestamp()


def _numeric_to_utc_seconds(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Non-finite numeric timestamp")

    rounded_year = round(value)
    if 1900 <= rounded_year <= 2100 and abs(value - rounded_year) < 1e-6:
        return _datetime_to_utc_seconds(datetime.datetime(rounded_year, 6, 15))

    abs_value = abs(value)
    if abs_value >= 1e18:
        return value / 1e9
    if abs_value >= 1e15:
        return value / 1e6
    if abs_value >= 1e12:
        return value / 1e3
    return value


def _string_to_utc_seconds(value: str) -> float:
    stripped = value.strip()
    if not stripped:
        raise ValueError("Empty timestamp string")

    iso_candidate = stripped[:-1] + '+00:00' if stripped.endswith('Z') else stripped
    try:
        dt = datetime.datetime.fromisoformat(iso_candidate)
    except ValueError:
        dt = None

    if dt is not None:
        return _datetime_to_utc_seconds(dt)

    numeric_value = float(stripped)
    return _numeric_to_utc_seconds(numeric_value)


def _as_iterable(time_data) -> Iterable:
    if torch is not None and isinstance(time_data, torch.Tensor):
        return time_data.detach().cpu().view(-1).tolist()
    if np is not None and isinstance(time_data, np.ndarray):
        return np.asarray(time_data).reshape(-1).tolist()
    if isinstance(time_data, (list, tuple)):
        return list(time_data)
    try:
        return list(time_data)
    except TypeError:
        return [time_data]


def convert_to_utc_seconds(time_data) -> List[float]:
    """Normalize timestamps to UTC seconds since the Unix epoch.

    The function accepts inputs commonly seen in the dataset:

    * ``datetime`` objects (naive timestamps are assumed to be UTC).
    * Numeric epochs expressed in seconds, milliseconds, microseconds,
      or nanoseconds. Magnitude heuristics are used to infer the unit.
    * Year-only values as numbers or strings between 1900 and 2100;
      these are mapped to June 15 of the given year to provide a stable
      mid-year reference point.
    * ISO-8601 formatted strings, including those with trailing ``"Z"``.

    Any value that fails to parse (including NaNs) is replaced with a
    fallback timestamp corresponding to 2000-06-15 00:00:00 UTC.
    """

    iterable = _as_iterable(time_data)
    times: List[float] = []

    for item in iterable:
        try:
            if isinstance(item, datetime.datetime):
                ts = _datetime_to_utc_seconds(item)
            elif isinstance(item, str):
                ts = _string_to_utc_seconds(item)
            elif isinstance(item, (int, float)):
                ts = _numeric_to_utc_seconds(float(item))
            elif torch is not None and isinstance(item, torch.Tensor):
                ts = _numeric_to_utc_seconds(float(item.item()))
            else:
                raise ValueError("Unsupported timestamp type")
        except Exception:
            ts = _FALLBACK_TIMESTAMP
        times.append(float(ts))

    return times


__all__ = ["convert_to_utc_seconds", "_FALLBACK_TIMESTAMP"]
