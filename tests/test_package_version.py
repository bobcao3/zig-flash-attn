import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package_source(tmp_path):
    return shutil.copytree(
        ROOT,
        tmp_path / "source",
        ignore=shutil.ignore_patterns(
            ".git",
            ".jj",
            ".pixi",
            ".zig-cache",
            "zig-out",
            "build",
            "dist",
            "__pycache__",
            "*.egg-info",
        ),
    )


def setup_command(source, *arguments, release_tag=None, check=True):
    env = dict(os.environ)
    env.pop("ZIGFA_RELEASE_VERSION", None)
    if release_tag is not None:
        env["ZIGFA_RELEASE_VERSION"] = release_tag
    return subprocess.run(
        [sys.executable, "setup.py", *arguments],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
        check=check,
        timeout=60,
    )


def version_at(source, **options):
    return setup_command(source, "--version", **options).stdout.strip()


@pytest.mark.parametrize("name", ["src/zigfa/torch.py", "zig/entry.zig", "reports/assets/report.css"])
def test_development_version_tracks_package_sources(package_source, name):
    version = version_at(package_source)
    release = (package_source / "VERSION").read_text().strip()
    assert re.fullmatch(re.escape(release) + r"\.dev0\+s[0-9a-f]{16}", version)
    assert version == version_at(ROOT)
    path = package_source / name
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    assert version_at(package_source) != version
    path.write_bytes(original)
    assert version_at(package_source) == version
    cache = package_source / "src" / "zigfa" / "__pycache__"
    cache.mkdir()
    (cache / "torch.pyc").write_bytes(b"cache")
    assert version_at(package_source) == version
    added = package_source / "src" / "zigfa" / "extra.py"
    added.write_text("")
    added_version = version_at(package_source)
    assert added_version != version
    added.rename(added.with_name("renamed.py"))
    assert version_at(package_source) != added_version


@pytest.mark.parametrize("release", [False, True])
def test_source_distribution_preserves_version(package_source, tmp_path, release):
    base = (package_source / "VERSION").read_text().strip()
    tag = f"v{base}" if release else None
    expected = version_at(package_source, release_tag=tag)
    if release:
        assert expected == base
    destination = tmp_path / "sdist"
    setup_command(package_source, "sdist", "--dist-dir", str(destination), release_tag=tag)
    archive, = destination.glob("*.tar.gz")
    with tarfile.open(archive) as distribution:
        distribution.extractall(tmp_path / "unpacked")
    unpacked, = (tmp_path / "unpacked").iterdir()
    assert not (unpacked / ".git").exists()
    assert version_at(unpacked) == expected


def test_release_tag_must_match_version_file(package_source):
    result = setup_command(package_source, "--version", release_tag="v999.0.0", check=False)
    assert result.returncode != 0
    assert "ZIGFA_RELEASE_VERSION must match VERSION" in result.stderr


def test_source_hash_does_not_depend_on_native_path_order(package_source):
    expected = version_at(package_source)
    script = """
import runpy
import sys
from pathlib import Path

Path.__lt__ = lambda left, right: left.as_posix().casefold() < right.as_posix().casefold()
sys.argv = ['setup.py', '--version']
runpy.run_path('setup.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package_source,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == expected
