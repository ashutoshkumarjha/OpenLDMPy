# -*- coding: utf-8 -*-
"""Checks whether OpenLDM's runtime dependencies are importable in the
current Python environment, and can install them into a plugin-local
virtual environment (built from QGIS's own bundled Python) if not.

Deliberately standalone -- does not import anything from vendor/ (LULC or
gui), since those transitively import these very dependencies at module
level (LULC.LULCAlgorithms imports rasterio/geopandas/sklearn/etc. at
import time). This module has to work *before* we know any of that is
present, so it stays plain stdlib (subprocess/importlib/sys/venv) -- no
PyQt, so it can be unit-tested without a QGIS/Qt bootstrap. Threading
(running installs off the GUI thread) and any progress-dialog wiring are
the caller's responsibility, not this module's.

Why a venv rather than `pip install --user` straight into QGIS's own
Python: self-contained (nothing installed outside the plugin folder --
doesn't touch QGIS's own site-packages or the user's global one),
trivial to clean up (delete the plugin), and sidesteps write-permission
issues into a possibly admin-owned QGIS install directory. The venv's
site-packages still has to be added to *this already-running*
interpreter's sys.path by the caller (a venv gives you an installation
layout, not a way to swap interpreters mid-process).

The venv is created with `--without-pip`, and packages are installed via
`pip install --target <venv's site-packages>` run through the *base*
interpreter (not the venv's own copy). Confirmed for real inside an
actual QGIS session: QGIS's bundled Python is compiled with a hardcoded
CI build-machine path as its default sys.prefix, so a fresh subprocess
of it can't locate its own stdlib unless PYTHONPATH is forwarded (see
_subprocess_env) -- and that fix cannot reach the one step that needs it
most, because `venv`'s own pip bootstrap (`ensurepip`) deliberately runs
the new venv's interpreter in isolated mode (`-Im`), which ignores
PYTHONPATH by design. `--without-pip` sidesteps that broken step
entirely; `--target` gets packages into the venv's site-packages using
the base interpreter's own pip, which is already proven to work.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import sysconfig
from importlib import import_module
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# import name -> pip package name (differ for a couple of these:
# "sklearn" imports as sklearn but installs as scikit-learn; "yaml"
# imports as yaml but installs as pyyaml).
#
# statsmodels is deliberately NOT listed here, even though
# LULC/modeling.py uses it: it's imported lazily, inside the one
# function that needs it (_fit_logistic_pvalues, a display-only
# Wald-test p-value diagnostic -- sklearn's LogisticRegression remains
# the model actually used for fitting/prediction everywhere). A missing
# or broken statsmodels (confirmed for real on macOS: its compiled
# statsmodels.robust extension can be blocked by hardened-runtime
# code-signing enforcement inside QGIS.app even after a successful pip
# install) degrades that one cosmetic detail (significance stars omitted,
# a warning logged) rather than being treated as a hard blocker that
# refuses to launch the GUI at all over a package the app doesn't
# actually need for anything else.
REQUIRED = {
    "rasterio": "rasterio",
    "geopandas": "geopandas",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "yaml": "pyyaml",
}


def missing_packages() -> List[Tuple[str, str]]:
    """[(import_name, pip_name), ...] for everything in REQUIRED that
    isn't importable right now."""
    missing = []
    for import_name, pip_name in REQUIRED.items():
        try:
            import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    return missing


def venv_python(venv_dir: str) -> str:
    """Path to the venv's own interpreter -- used only to detect whether
    the venv has already been created (ensure_venv), not to run pip: the
    venv is deliberately created with --without-pip and packages are
    installed via the base interpreter's pip instead (see install_missing
    for why)."""
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    exe = "python.exe" if sys.platform == "win32" else "python3"
    return str(Path(venv_dir) / bin_dir / exe)


def venv_site_packages(venv_dir: str) -> str:
    """Path to add to sys.path so the *running* interpreter can import
    whatever the venv's own pip installed."""
    if sys.platform == "win32":
        return str(Path(venv_dir) / "Lib" / "site-packages")
    scheme_path = sysconfig.get_path(
        "purelib", scheme="venv",
        vars={"base": venv_dir, "platbase": venv_dir},
    )
    return scheme_path


def base_interpreter() -> str:
    """The real, standalone CPython binary -- NOT sys.executable, which
    inside an embedded host (QGIS.app's own binary, when this code runs
    as a loaded plugin rather than a plain script) reports the *host
    application's* binary path, not an interpreter one at all (confirmed
    against a real QGIS session: sys.executable there is literally
    ".../QGIS.app/Contents/MacOS/QGIS"). sys._base_executable (Python
    3.11+, which QGIS's own bundled Python always satisfies) is the one
    that still reports the real interpreter binary in that situation."""
    return getattr(sys, "_base_executable", None) or sys.executable


def _subprocess_env() -> dict:
    """Environment for spawning the base interpreter (venv creation, and
    later the venv's own pip) as a subprocess.

    QGIS's bundled Python is compiled with a hardcoded CI build-machine
    path as its default sys.prefix, so a freshly spawned copy of it can't
    locate its own stdlib and fails before it can even import `encodings`
    -- confirmed for real, both invoking it directly from a terminal and,
    worse, from inside this exact install flow running for real inside
    QGIS (the venv's own python failed this exact way installing
    statsmodels). A subprocess doesn't inherit whatever internal sys.path
    fixups the *current*, already-correctly-booted interpreter applied at
    its own startup -- only real OS environment variables do that.
    Forwarding this already-working process's own resolved sys.path as
    PYTHONPATH is what lets the child bootstrap at all.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return env


def has_internet(host: str = "pypi.org", port: int = 443, timeout: float = 3.0) -> bool:
    """A fast (few-second) connectivity probe, checked before handing off
    to pip. Without this, a genuinely offline machine still pays pip's
    own default retry/backoff schedule (multiple retries per package,
    across every missing package) before it gives up -- several minutes
    of a frozen progress dialog for a failure that was knowable in
    seconds."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


# Runs pip via its own Python entry point (pip._internal.cli.main.main)
# rather than the `-m pip` module-invocation shorthand -- functionally
# the same thing (`-m pip` is itself just this exact call, made by pip's
# own __main__.py), but explicit about it: this is pip's actual install
# code, driven directly, with pip's own dependency resolver making every
# call (nothing homemade -- no reimplementing what packages already
# satisfy a requirement). Executed via the *base* interpreter, since
# calling it in-process inside the running QGIS/plugin interpreter would
# let pip mutate that long-lived process's global state (logging
# handlers, warnings filters) for the whole QGIS session, not just for
# the duration of the install -- pip's own docs are explicit that it
# isn't meant to be used as an in-process library for exactly this
# reason. Running it in a real subprocess keeps every side effect
# contained to that subprocess, which simply exits when done.
_PIP_INSTALL_SCRIPT = (
    "import sys\n"
    "from pip._internal.cli.main import main as _pip_main\n"
    "sys.exit(_pip_main(sys.argv[1:]))\n"
)


def _run(cmd: List[str], on_output: Optional[Callable[[str], None]], env: Optional[dict] = None) -> Tuple[bool, str]:
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
    except OSError as exc:
        return False, str(exc)

    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        if on_output:
            on_output(line)
    proc.wait()
    return proc.returncode == 0, "".join(lines)


def ensure_venv(venv_dir: str, on_output: Optional[Callable[[str], None]] = None) -> Tuple[bool, str]:
    """Creates the venv at venv_dir (using the real base interpreter --
    QGIS's own bundled Python -- as its base) if it doesn't already have
    a working interpreter there. No-ops (returns success) if it does.

    --without-pip: venv's normal pip bootstrap runs the new venv's own
    interpreter as `-Im ensurepip`, and -I (isolated mode) ignores
    PYTHONPATH -- the one place _subprocess_env's fix can't reach.
    Packages are installed afterwards via the base interpreter's own
    already-working pip instead (see install_missing)."""
    if Path(venv_python(venv_dir)).exists():
        return True, ""
    return _run(
        [base_interpreter(), "-m", "venv", "--without-pip", venv_dir],
        on_output, env=_subprocess_env(),
    )


def install_missing(
    pip_names: List[str],
    venv_dir: str,
    on_output: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Ensures the plugin-local venv exists, then installs pip_names into
    its site-packages -- via the base interpreter's own pip targeting the
    venv's site-packages dir (`pip install --target`), not the venv's own
    (deliberately pip-less, see ensure_venv) copy.

    Returns (success, full_combined_output) -- combining the venv-creation
    output (if the venv didn't exist yet) and the install output.

    Fails fast (no pip invocation at all) if there's no internet -- venv
    creation itself needs none (it only copies the base interpreter's own
    stdlib), so that step still runs even offline.
    """
    ok, venv_output = ensure_venv(venv_dir, on_output=on_output)
    if not ok:
        return False, venv_output

    if not has_internet():
        message = (
            "No internet connection detected (couldn't reach pypi.org). "
            "Installing OpenLDM's missing Python packages needs internet "
            "access -- please connect and try again.\n"
        )
        if on_output:
            on_output(message)
        return False, venv_output + message

    # --retries/--timeout bound how long a *partial* failure (package
    # reachable, connection drops mid-download, DNS flaky) can drag the
    # progress dialog out for -- the has_internet() check above only
    # catches being offline outright, not a connection that dies partway
    # through a multi-package install.
    cmd = [
        base_interpreter(), "-c", _PIP_INSTALL_SCRIPT,
        "install",
        "--target", venv_site_packages(venv_dir),
        "--retries", "1", "--timeout", "20",
        *pip_names,
    ]
    ok, install_output = _run(cmd, on_output, env=_subprocess_env())
    return ok, venv_output + install_output


def manual_install_command(pip_names: List[str], venv_dir: str) -> str:
    """The exact commands to show a user when an automatic install fails
    or is declined -- mirrors exactly what install_missing() does
    automatically (same base interpreter, same --without-pip/--target
    split), so a user copy-pasting this reproduces the same, already-
    working path rather than a plausible-looking one that hits the same
    ensurepip/-I wall this design works around."""
    return (
        f"{base_interpreter()} -m venv --without-pip {venv_dir}\n"
        f"{base_interpreter()} -m pip install --target "
        f"{venv_site_packages(venv_dir)} {' '.join(pip_names)}"
    )
