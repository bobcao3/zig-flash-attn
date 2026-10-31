import hashlib
import os
import shutil
import subprocess
import sys
from email.parser import Parser
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent


def package_version():
    release_version = (ROOT / "VERSION").read_text().strip()
    release_tag = os.environ.get("ZIGFA_RELEASE_VERSION")
    if release_tag is not None:
        if release_tag.removeprefix("v") != release_version:
            raise RuntimeError("ZIGFA_RELEASE_VERSION must match VERSION")
        return release_version
    metadata = ROOT / "PKG-INFO"
    if metadata.is_file() and Parser().parsestr(metadata.read_text())["Version"] == release_version:
        return release_version
    sources = [ROOT / name for name in (
        "VERSION",
        "LICENSE",
        "README.md",
        "MANIFEST.in",
        "pyproject.toml",
        "setup.py",
        "build.zig",
        "reports/profile_viewer.html",
        "reports/assets/report.css",
    )]
    sources.extend((ROOT / "src" / "zigfa").rglob("*.py"))
    sources.extend((ROOT / "src" / "zigfa").rglob("py.typed"))
    sources.extend((ROOT / "zig").rglob("*.zig"))
    digest = hashlib.sha256()
    for path in sorted(sources, key=lambda path: path.relative_to(ROOT).as_posix()):
        for part in (path.relative_to(ROOT).as_posix().encode(), path.read_bytes()):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return f"{release_version}.dev0+s{digest.hexdigest()[:16]}"


def zig_arguments():
    arguments = ["build", "--summary", "all"]
    if os.environ.get("ZIGFA_VERBOSE") == "1":
        arguments.append("--verbose")
    if os.environ.get("ZIGFA_PROFILE") == "1":
        arguments.append("-Dprofile")
    for option, variable in (
        ("ptxas", "PTXAS"),
        ("sm", "ZIGFA_SM"),
        ("source", "ZIGFA_SOURCE"),
        ("head-dim", "ZIGFA_HEAD_DIM"),
        ("variant", "ZIGFA_VARIANT"),
    ):
        for value in filter(None, os.environ.get(variable, "").split(",")):
            arguments.append(f"-D{option}={value}")
    return arguments


def run_zig(arguments):
    try:
        terminal = open("/dev/tty", "w")
    except OSError:
        terminal = None
    output = terminal or sys.stderr
    try:
        subprocess.run([os.environ.get("ZIG", "zig"), *arguments], cwd=ROOT, check=True, stdout=output, stderr=output)
    finally:
        if terminal:
            terminal.close()


class BuildKernels(build_py):
    def run(self):
        run_zig(zig_arguments())
        super().run()
        destination = Path(self.build_lib) / "zigfa" / "cubin"
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True)
        installed = ROOT / "zig-out"
        if (installed / "kernels.declared").read_text().split()[:1] == ["partial"] and os.environ.get(
            "ZIGFA_ALLOW_PARTIAL"
        ) != "1":
            raise RuntimeError("the kernel build is partial; set ZIGFA_ALLOW_PARTIAL=1 to package it anyway")
        for cubin in (installed / "cubin").glob("*.cubin"):
            shutil.copy(cubin, destination)
        for listing in ("kernels.manifest", "kernels.declared"):
            shutil.copy(installed / listing, destination)
        viewer = Path(self.build_lib) / "zigfa" / "viewer"
        (viewer / "assets").mkdir(parents=True, exist_ok=True)
        for page in ("profile_viewer.html", "assets/report.css"):
            shutil.copy(ROOT / "reports" / page, viewer / page)


setup(version=package_version(), cmdclass={"build_py": BuildKernels})
