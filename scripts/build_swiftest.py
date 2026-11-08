"""Build pip packages from a clean copy of the current SwifTest source.

Install build prerequisites with: python -m pip install "setuptools>=61" wheel
Run from any directory: python PATH_TO_REPO/scripts/build_swiftest.py
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def stage_source(root, destination):
    """Exclude other projects, previous builds, caches and research artifacts."""
    files = [root / "pyproject.toml", root / "docs/swiftest.md"]
    files += sorted((root / "swiftest").glob("*.py"))
    files += sorted((root / "swiftest/assets").glob("*.java"))
    for required in ("swiftest/__init__.py", "swiftest/assets/IntentScreenshot.java"):
        if root / required not in files:
            raise FileNotFoundError(root / required)
    for source in files:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(destination / "docs/swiftest.md", destination / "README.md")


def build(kind, source, output):
    # A fresh interpreter prevents backend state from leaking between builds.
    subprocess.run(
        [sys.executable, "-c",
         "import sys; from setuptools import build_meta; "
         "getattr(build_meta, sys.argv[1])(sys.argv[2])",
         "build_" + kind, str(output)],
        cwd=source, check=True,
    )
    suffix = "*.tar.gz" if kind == "sdist" else "*.whl"
    artifacts = list(output.glob(suffix))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one {kind} artifact, found {len(artifacts)}.")
    return artifacts[0]


def unpack_source(archive, destination):
    """Extract regular files only, with the same checks on Python 3.9 and later."""
    destination = destination.resolve()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if (not target.is_relative_to(destination)
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f"Unsafe source archive entry: {member.name}")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not (roots[0] / "pyproject.toml").is_file():
        raise ValueError("Source archive must contain one project root.")
    return roots[0]


def verify_wheel(wheel, staged):
    with zipfile.ZipFile(wheel) as bundle:
        for name in bundle.namelist():
            if not (name.startswith("swiftest/")
                    or name.split("/", 1)[0].startswith("swiftest-")
                    and name.split("/", 1)[0].endswith(".dist-info")):
                raise ValueError(f"Unexpected wheel content: {name}")
        for source in (staged / "swiftest").rglob("*"):
            if source.is_file():
                name = source.relative_to(staged).as_posix()
                if bundle.read(name) != source.read_bytes():
                    raise ValueError(f"Wheel content differs from source: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist",
                        help="Artifact directory (default: repository dist/).")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    scratch = ROOT / ".tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swiftest-build-", dir=scratch) as temp:
        work = Path(temp)
        staged, artifacts = work / "source", work / "artifacts"
        artifacts.mkdir()
        stage_source(ROOT, staged)
        sdist = build("sdist", staged, artifacts)
        extracted = unpack_source(sdist, work / "unpacked")
        wheel = build("wheel", extracted, artifacts)
        verify_wheel(wheel, staged)
        checksums = "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (wheel, sdist)
        )
        # Publish only after both builds and content checks succeed. Keep other
        # releases in the output directory; replace only this version's files.
        output.mkdir(parents=True, exist_ok=True)
        for artifact in (wheel, sdist):
            shutil.copyfile(artifact, output / artifact.name)
            print(output / artifact.name)
        (output / "SHA256SUMS.txt").write_text(checksums, encoding="ascii")
        print(output / "SHA256SUMS.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
