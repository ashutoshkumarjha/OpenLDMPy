# coding=utf-8
"""Tests for dependency_check.py in isolation -- deliberately plain
Python, no QGIS bootstrap (unlike the other tests in this directory),
since the module itself is designed to work standalone."""

import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dependency_check as dc  # noqa: E402


def test_missing_packages_flags_a_nonexistent_module(monkeypatch):
    monkeypatch.setattr(dc, "REQUIRED", {"definitely_not_a_real_module_xyz": "fake-pkg"})
    assert dc.missing_packages() == [("definitely_not_a_real_module_xyz", "fake-pkg")]


def test_missing_packages_clears_an_already_importable_module(monkeypatch):
    monkeypatch.setattr(dc, "REQUIRED", {"os": "not-actually-pip-installed", "sys": "also-stdlib"})
    assert dc.missing_packages() == []


def test_missing_packages_reports_only_the_actually_missing_ones(monkeypatch):
    monkeypatch.setattr(
        dc, "REQUIRED", {"os": "stdlib", "definitely_not_a_real_module_xyz": "fake-pkg"}
    )
    assert dc.missing_packages() == [("definitely_not_a_real_module_xyz", "fake-pkg")]


def test_venv_python_path_shape():
    path = dc.venv_python("/plugin/venv")
    if sys.platform == "win32":
        assert path == "/plugin/venv\\Scripts\\python.exe" or path.endswith("Scripts\\python.exe")
    else:
        assert path == "/plugin/venv/bin/python3"


def test_venv_site_packages_under_venv_dir():
    site_packages = dc.venv_site_packages("/plugin/venv")
    assert site_packages.startswith("/plugin/venv" + ("\\" if sys.platform == "win32" else "/"))
    assert "site-packages" in site_packages


def test_ensure_venv_skips_creation_if_interpreter_already_exists(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv"
    fake_python = Path(dc.venv_python(str(venv_dir)))
    fake_python.parent.mkdir(parents=True)
    fake_python.touch()

    def _fail_if_called(cmd, **kwargs):
        raise AssertionError("should not attempt to create the venv again")

    monkeypatch.setattr(dc.subprocess, "Popen", _fail_if_called)

    ok, output = dc.ensure_venv(str(venv_dir))
    assert ok is True
    assert output == ""


def test_ensure_venv_creates_it_with_this_interpreter_when_missing(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv"
    captured = {}

    class _FakeProcess:
        stdout = ["created venv\n"]
        returncode = 0

        def wait(self):
            pass

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProcess()

    monkeypatch.setattr(dc.subprocess, "Popen", _fake_popen)

    ok, output = dc.ensure_venv(str(venv_dir))
    assert ok is True
    assert captured["cmd"] == [dc.base_interpreter(), "-m", "venv", "--without-pip", str(venv_dir)]


def test_install_missing_creates_venv_then_installs_via_base_interpreters_pip(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv"
    calls = []

    class _FakeProcess:
        def __init__(self, lines):
            self.stdout = lines
            self.returncode = 0

        def wait(self):
            pass

    def _fake_popen(cmd, **kwargs):
        calls.append(cmd)
        if "venv" in cmd:
            return _FakeProcess(["created venv\n"])
        return _FakeProcess(["Successfully installed fake-pkg\n"])

    monkeypatch.setattr(dc.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dc, "has_internet", lambda: True)

    success, output = dc.install_missing(["fake-pkg"], str(venv_dir))

    assert success is True
    assert calls[0] == [dc.base_interpreter(), "-m", "venv", "--without-pip", str(venv_dir)]
    assert calls[1] == [
        dc.base_interpreter(), "-c", dc._PIP_INSTALL_SCRIPT,
        "install",
        "--target", dc.venv_site_packages(str(venv_dir)),
        "--retries", "1", "--timeout", "20", "fake-pkg",
    ]
    assert "created venv" in output
    assert "Successfully installed fake-pkg" in output


def test_pip_install_script_calls_pips_own_internal_main():
    """The install step runs pip's actual entry point function, not a
    hand-rolled reimplementation -- exercised for real by exec'ing the
    script (against the real, already-installed pip in this dev env,
    with its main() patched out) and checking it's invoked with the
    right argv rather than pip's install logic being reimplemented by
    hand."""
    import pip._internal.cli.main as real_pip_main_module

    old_argv = sys.argv
    sys.argv = ["-c", "install", "fake-pkg"]
    try:
        with mock.patch.object(real_pip_main_module, "main", return_value=0) as fake_main:
            with mock.patch.object(sys, "exit") as fake_exit:
                exec(compile(dc._PIP_INSTALL_SCRIPT, "<test>", "exec"), {})
    finally:
        sys.argv = old_argv

    fake_main.assert_called_once_with(["install", "fake-pkg"])
    fake_exit.assert_called_once_with(0)


def test_install_missing_skips_pip_call_if_venv_creation_fails(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv"

    def _raise(cmd, **kwargs):
        raise OSError("no such file or directory")

    monkeypatch.setattr(dc.subprocess, "Popen", _raise)

    success, output = dc.install_missing(["whatever"], str(venv_dir))

    assert success is False
    assert "no such file or directory" in output


def test_install_missing_reports_failure_on_nonzero_returncode(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv"
    fake_python = Path(dc.venv_python(str(venv_dir)))
    fake_python.parent.mkdir(parents=True)
    fake_python.touch()  # venv already exists, so only the pip call happens

    class _FakeProcess:
        stdout = ["ERROR: could not find a version that satisfies the requirement\n"]
        returncode = 1

        def wait(self):
            pass

    monkeypatch.setattr(dc.subprocess, "Popen", lambda cmd, **kwargs: _FakeProcess())
    monkeypatch.setattr(dc, "has_internet", lambda: True)

    success, output = dc.install_missing(["nonexistent-package"], str(venv_dir))

    assert success is False
    assert "could not find a version" in output


def test_install_missing_fails_fast_without_internet_and_never_calls_pip(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv"
    fake_python = Path(dc.venv_python(str(venv_dir)))
    fake_python.parent.mkdir(parents=True)
    fake_python.touch()  # venv already exists -- only the pip call is at stake

    def _fail_if_called(cmd, **kwargs):
        raise AssertionError("pip must not be invoked when offline")

    monkeypatch.setattr(dc.subprocess, "Popen", _fail_if_called)
    monkeypatch.setattr(dc, "has_internet", lambda: False)

    success, output = dc.install_missing(["fake-pkg"], str(venv_dir))

    assert success is False
    assert "internet" in output.lower()


def test_has_internet_returns_false_when_connection_fails(monkeypatch):
    def _raise(*a, **k):
        raise OSError("network unreachable")

    monkeypatch.setattr(dc.socket, "create_connection", _raise)
    assert dc.has_internet() is False


def test_has_internet_returns_true_when_connection_succeeds(monkeypatch):
    class _FakeSocket:
        def close(self):
            pass

    monkeypatch.setattr(dc.socket, "create_connection", lambda *a, **k: _FakeSocket())
    assert dc.has_internet() is True


def test_manual_install_command_uses_this_interpreter_and_venv_dir():
    cmd = dc.manual_install_command(["rasterio", "geopandas"], "/plugin/venv")
    assert f"{dc.base_interpreter()} -m venv --without-pip /plugin/venv" in cmd
    assert (
        f"{dc.base_interpreter()} -m pip install --target "
        f"{dc.venv_site_packages('/plugin/venv')} rasterio geopandas"
    ) in cmd


def test_base_interpreter_prefers_base_executable_over_executable(monkeypatch):
    monkeypatch.setattr(dc.sys, "_base_executable", "/real/python3.12", raising=False)
    monkeypatch.setattr(dc.sys, "executable", "/Applications/QGIS.app/Contents/MacOS/QGIS")
    assert dc.base_interpreter() == "/real/python3.12"


def test_base_interpreter_falls_back_to_executable_when_base_executable_missing(monkeypatch):
    monkeypatch.delattr(dc.sys, "_base_executable", raising=False)
    monkeypatch.setattr(dc.sys, "executable", "/some/python")
    assert dc.base_interpreter() == "/some/python"


def test_subprocess_env_forwards_current_sys_path_as_pythonpath(monkeypatch):
    monkeypatch.setattr(dc.sys, "path", ["/a", "/b", "/c"])
    env = dc._subprocess_env()
    assert env["PYTHONPATH"] == f"/a{os.pathsep}/b{os.pathsep}/c"
    # Rest of the real environment (e.g. PATH) still comes through --
    # only PYTHONPATH is being overridden, not the whole environment.
    assert env.get("PATH") == os.environ.get("PATH")


def test_ensure_venv_and_install_missing_use_subprocess_env(monkeypatch, tmp_path):
    """Regression test for the real bug hit inside an actual QGIS session:
    a bare `sys.executable -m venv ...` (or the venv's own pip) subprocess
    call, launched without PYTHONPATH forwarded, fails to boot at all."""
    venv_dir = tmp_path / "venv"
    envs_seen = []

    class _FakeProcess:
        stdout = ["ok\n"]
        returncode = 0

        def wait(self):
            pass

    def _fake_popen(cmd, **kwargs):
        envs_seen.append(kwargs.get("env"))
        return _FakeProcess()

    monkeypatch.setattr(dc.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dc, "has_internet", lambda: True)

    dc.install_missing(["fake-pkg"], str(venv_dir))

    assert len(envs_seen) == 2  # venv creation, then pip install
    for env in envs_seen:
        assert env is not None
        assert "PYTHONPATH" in env
