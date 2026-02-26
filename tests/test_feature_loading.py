from __future__ import annotations

import yaml
import pytest

from uma.core.uma_memory import UMAMemory

from tests.helpers.runtime import build_test_config


@pytest.mark.asyncio
async def test_feature_loader_attaches_procedural_feature(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)

    cfg = build_test_config(db_root=db_root)
    cfg["features"] = {
        "load": [
            {
                "name": "procedural",
                "enabled": True,
                "provider": "uma.features.procedural.feature:ProceduralFeature",
                "config": {"max_k": 3},
            }
        ],
        "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False},
    }

    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()

    assert "procedural" in memory.features
    assert callable(getattr(memory, "procedural_health"))


@pytest.mark.asyncio
async def test_feature_loader_skips_failed_attachment_when_policy_is_log_and_skip(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)

    cfg = build_test_config(db_root=db_root)
    cfg["features"] = {
        "load": [
            {
                "name": "procedural",
                "enabled": True,
                "provider": "uma.features.procedural.feature:DoesNotExist",
                "config": {"max_k": 3},
            }
        ],
        "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False},
    }

    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()

    assert "procedural" not in memory.features

