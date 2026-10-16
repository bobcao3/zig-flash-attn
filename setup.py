import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent


def zig_arguments():
    arguments = ["build", "--summary", "all"]
    if os.environ.get("ZIGFA_VERBOSE") == "1":
        arguments.append("--verbose")
    for option, variable in (
        ("ptxas", "PTXAS"),
        ("sm", "ZIGFA_SM"),
        ("source", "ZIGFA_SOURCE"),
        ("head-dim", "ZIGFA_HEAD_DIM"),
        ("mode", "ZIGFA_MODE"),
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


setup(cmdclass={"build_py": BuildKernels})
