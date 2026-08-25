"""Canonical serialization and hashing for the schema-v5 contract."""

import copy
import hashlib
import json

from piper_tesseract_foxy.contract_core import ContractError


def canonical_bytes(value):
    """Serialize one finite JSON value to the stable wire representation."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        ).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise ContractError(
            'payload is not canonical finite JSON: %s' % error)


def sha256_value(value):
    """Hash the canonical bytes of one value."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    """Hash a file without changing the established block size."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def attach_digest(payload, field):
    """Return a deep copy carrying its canonical digest."""
    value = copy.deepcopy(payload)
    value.pop(field, None)
    value[field] = sha256_value(value)
    return value


def verify_digest(payload, field):
    """Reject a missing, malformed, or non-canonical payload digest."""
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ContractError('%s is missing or invalid' % field)
    value = copy.deepcopy(payload)
    value.pop(field, None)
    if sha256_value(value) != expected:
        raise ContractError('%s does not match canonical payload' % field)
