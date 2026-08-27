"""Compatibility facade for the infrastructure-owned failure model."""

from piper_mobile_manipulation.infrastructure.failure_model import (
    Failure,
    FailureCode,
    FailureLike,
    FailureTag,
    as_failure,
    legacy_failure_adapter,
)

__all__ = [
    'Failure',
    'FailureCode',
    'FailureLike',
    'FailureTag',
    'as_failure',
    'legacy_failure_adapter',
]
