"""Check downloaded artifacts using Python's standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(full=False):
    manifest = json.loads((ROOT / "models/active_model.json").read_text())
    expected = dict(manifest["source_sha256"])
    expected.update({"models/" + name: value for name, value in
                     manifest["weights_and_metadata_sha256"].items()})
    if full:
        release = json.loads((ROOT / "release_manifest.json").read_text())
        expected.update({name: record["sha256"] for name, record in release["files"].items()})
    errors = []
    for name, value in expected.items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT.resolve()):
            errors.append(name + ": path leaves the artifact directory")
        elif not path.is_file():
            errors.append(name + ": missing")
        else:
            with path.open("rb") as stream:
                pointer = stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1")
            if pointer:
                errors.append(name + ": Git LFS pointer; run the documented git lfs pull command")
            elif digest(path) != value:
                errors.append(name + ": checksum differs from the distributed version")
    if errors:
        raise RuntimeError("Artifact verification failed:\n" + "\n".join(errors))
    print(f"Verified {len(expected)} {'artifact' if full else 'model'} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also verify training data, results, and documentation")
    args = parser.parse_args()
    try:
        verify(args.full)
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
