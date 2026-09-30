from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum


def _convert(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: _convert(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert(v) for v in value]
    return value


def package_to_dict(package) -> dict:
    return _convert(package)
