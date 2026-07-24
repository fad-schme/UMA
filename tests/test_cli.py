from __future__ import annotations

import json
from pathlib import Path

import yaml

from tests.helpers.runtime import build_test_config
from uma.cli import main
from uma.version import __version__


def _config_path(tmp_path: Path) -> Path:
    config = build_test_config(db_root=tmp_path / "db")
    config["llms"]["uma"]["config"]["api_key"] = "llm-secret"
    config["embedding"]["config"]["token"] = "embedding-secret"
    path = tmp_path / "uma.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_version(capsys) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"uma {__version__}"


def test_version_json(capsys) -> None:
    assert main(["--format", "json", "version"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "schema_version": "1",
        "command": "version",
        "status": "ok",
        "data": {"version": __version__},
    }


def test_config_validate(tmp_path: Path, capsys) -> None:
    path = _config_path(tmp_path)

    assert main(["--config", str(path), "config", "validate"]) == 0

    output = capsys.readouterr().out.strip()
    assert output == f"Valid UMA configuration: {path.resolve()}"
    assert not (tmp_path / "db").exists()


def test_config_uses_environment_path(tmp_path: Path, capsys, monkeypatch) -> None:
    path = _config_path(tmp_path)
    monkeypatch.setenv("UMA_CONFIG", str(path))

    assert main(["--format", "json", "config", "validate"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["data"]["path"] == str(path.resolve())


def test_config_show_redacts_secrets(tmp_path: Path, capsys) -> None:
    path = _config_path(tmp_path)

    assert main(["--config", str(path), "--format", "json", "config", "show"]) == 0

    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["data"]["config"]["llms"]["uma"]["config"]["api_key"] == "<redacted>"
    assert result["data"]["config"]["embedding"]["config"]["token"] == "<redacted>"
    assert "llm-secret" not in output
    assert "embedding-secret" not in output


def test_missing_config_is_a_usage_error(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UMA_CONFIG", raising=False)

    assert main(["--format", "json", "config", "validate"]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "Use --config or UMA_CONFIG" in result["errors"][0]["message"]
