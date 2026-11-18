"""Resolve frozen historical filenames through the documented relocation."""
import json
from pathlib import Path
from current_data import ROOT, sha


def verify_frozen_files(files):
    """Check current files while preserving the original protocol bytes/hashes.

    Relocated source files have both their original and packaged checksums in
    relocation.json. Data, fitted weights and evaluation protocols keep their
    original checksums. No access to the original repository is required.
    """
    relocation = json.loads((ROOT/'relocation.json').read_text())
    mapping = relocation['frozen_files']
    for original, expected in files.items():
        if original not in mapping:
            raise ValueError('Unregistered frozen path: '+original)
        record = mapping[original]
        if record['original_sha256'] != expected:
            raise ValueError('Frozen protocol checksum differs: '+original)
        for relative, digest in record['packaged_files'].items():
            path = (ROOT/relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or sha(path) != digest:
                raise ValueError('Packaged file changed: '+relative)


def recorded_excluded_seeds():
    return set(json.loads((ROOT/'data/seed_exclusions.json').read_text())['seeds'])
