"""Tests for OpenLDM.py's unified --mode gui/nogui dispatch."""

import pytest

import OpenLDM


def test_no_args_defaults_to_gui(monkeypatch):
    calls = []
    monkeypatch.setattr(OpenLDM, "run_gui", lambda config_path=None: calls.append(("gui", config_path)) or 0)
    monkeypatch.setattr(OpenLDM, "run_nogui", lambda path: calls.append(("nogui", path)))
    monkeypatch.setattr("sys.argv", ["OpenLDM.py"])

    OpenLDM.main()

    assert calls == [("gui", None)]


def test_explicit_mode_gui(monkeypatch):
    calls = []
    monkeypatch.setattr(OpenLDM, "run_gui", lambda config_path=None: calls.append(("gui", config_path)) or 0)
    monkeypatch.setattr("sys.argv", ["OpenLDM.py", "--mode", "gui"])

    OpenLDM.main()

    assert calls == [("gui", None)]


def test_mode_gui_with_config_passes_it_through(monkeypatch):
    """--config in gui mode is loaded into the window on startup (same as
    File > Open) -- not ignored, unlike the earlier design."""
    calls = []
    monkeypatch.setattr(OpenLDM, "run_gui", lambda config_path=None: calls.append(("gui", config_path)) or 0)
    monkeypatch.setattr("sys.argv", ["OpenLDM.py", "--mode", "gui", "--config", "/tmp/scenario.yaml"])

    OpenLDM.main()

    assert calls == [("gui", "/tmp/scenario.yaml")]


def test_mode_nogui_without_config_errors(monkeypatch):
    monkeypatch.setattr("sys.argv", ["OpenLDM.py", "--mode", "nogui"])
    with pytest.raises(SystemExit):
        OpenLDM.main()


def test_mode_nogui_with_config_calls_run_nogui(monkeypatch):
    calls = []
    monkeypatch.setattr(OpenLDM, "run_gui", lambda config_path=None: calls.append(("gui", config_path)) or 0)
    monkeypatch.setattr(OpenLDM, "run_nogui", lambda path: calls.append(("nogui", path)) or 0)
    monkeypatch.setattr("sys.argv", ["OpenLDM.py", "--mode", "nogui", "--config", "/tmp/scenario.yaml"])

    OpenLDM.main()

    assert calls == [("nogui", "/tmp/scenario.yaml")]


def test_invalid_mode_choice_errors(monkeypatch):
    monkeypatch.setattr("sys.argv", ["OpenLDM.py", "--mode", "bogus"])
    with pytest.raises(SystemExit):
        OpenLDM.main()


def test_run_nogui_malformed_config_prints_error_and_returns_nonzero(tmp_path, capsys):
    bad_path = tmp_path / "corrupted.yaml"
    bad_path.write_text("scenario_version: 1\ndata: [unbalanced\n")

    exit_code = OpenLDM.run_nogui(str(bad_path))

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Error" in captured.err
    # No raw traceback -- just the clean message.
    assert "Traceback" not in captured.err


def test_run_nogui_missing_config_prints_error_and_returns_nonzero(capsys):
    exit_code = OpenLDM.run_nogui("/nonexistent/path/scenario.yaml")

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert "Traceback" not in captured.err


def test_main_propagates_nogui_exit_code(monkeypatch):
    monkeypatch.setattr(OpenLDM, "run_nogui", lambda path: 1)
    monkeypatch.setattr("sys.argv", ["OpenLDM.py", "--mode", "nogui", "--config", "/tmp/scenario.yaml"])

    assert OpenLDM.main() == 1
