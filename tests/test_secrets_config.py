from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.helpers.runtime import build_test_config
from uma.adapters import (
    EnvVarProvider,
    Secret,
    SecretNotFound,
    SecretsProvider,
    SecretsProviderError,
)
from uma.api.memory import UMAMemory


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_public_adapter_surface_exports_secrets_types() -> None:
    assert SecretsProvider is not None
    assert EnvVarProvider is not None
    assert Secret is not None
    assert SecretsProviderError is not None
    assert SecretNotFound is not None


def test_from_yaml_without_secrets_block_preserves_lite_default(tmp_path: Path) -> None:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg_path = _write_config(tmp_path, cfg)

    memory = UMAMemory.from_yaml(str(cfg_path))
    try:
        assert memory._secrets_provider is None
    finally:
        memory.shutdown()


def test_from_yaml_with_secrets_block_initializes_provider(tmp_path: Path) -> None:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg["secrets"] = {
        "provider": "uma.adapters.secrets.EnvVarProvider",
        "options": {"prefix": "UMA"},
    }
    cfg_path = _write_config(tmp_path, cfg)

    memory = UMAMemory.from_yaml(str(cfg_path))
    try:
        assert isinstance(memory._secrets_provider, EnvVarProvider)
        assert memory._secrets_cfg is not None
        assert memory._secrets_cfg.provider == "uma.adapters.secrets.EnvVarProvider"
    finally:
        memory.shutdown()


def test_from_yaml_rejects_unresolvable_secrets_provider(tmp_path: Path) -> None:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg["secrets"] = {
        "provider": "uma.adapters.secrets.DoesNotExist",
        "options": {},
    }
    cfg_path = _write_config(tmp_path, cfg)

    with pytest.raises(RuntimeError, match=r"secrets\.provider"):
        UMAMemory.from_yaml(str(cfg_path))


def test_from_yaml_rejects_invalid_secrets_options(tmp_path: Path) -> None:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg["secrets"] = {
        "provider": "uma.adapters.secrets.EnvVarProvider",
        "options": {"prefix": ""},
    }
    cfg_path = _write_config(tmp_path, cfg)

    with pytest.raises(RuntimeError, match=r"secrets\.options"):
        UMAMemory.from_yaml(str(cfg_path))
