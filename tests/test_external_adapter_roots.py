from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from tests.helpers.runtime import build_test_config
from uma.common.config import UMAConfig
from uma.api.memory import UMAMemory
from uma.common.config_types import parse_plugin_spec


def _clear_plugin_module(module_name: str) -> None:
    for key in list(sys.modules):
        if key == module_name or key.startswith(f"{module_name}."):
            sys.modules.pop(key, None)


def _write_vector_adapter(
    adapter_root: Path,
    *,
    module_name: str = "qdrant_adapter",
    class_name: str = "QdrantIndex",
    marker: str = "default",
) -> None:
    vector_dir = adapter_root / "vector"
    vector_dir.mkdir(parents=True, exist_ok=True)
    (vector_dir / "__init__.py").write_text("")
    (vector_dir / f"{module_name}.py").write_text(
        "\n".join(
            [
                "from uma.adapters.vector.inmemory import InMemoryVectorIndex",
                "",
                f"class {class_name}(InMemoryVectorIndex):",
                f"    MARKER = {marker!r}",
                "",
            ]
        )
    )


def test_load_yaml_registers_external_adapter_root_from_env(tmp_path, monkeypatch):
    adapter_root = tmp_path / "vendor_adapters"
    config_root = tmp_path / "runtime_config"
    db_root = tmp_path / "db"
    config_root.mkdir(parents=True, exist_ok=True)
    db_root.mkdir(parents=True, exist_ok=True)
    _write_vector_adapter(adapter_root)

    cfg = build_test_config(db_root=db_root)
    cfg["storage"]["vector_backend"] = "vector.qdrant_adapter:QdrantIndex"

    cfg_path = config_root / "uma.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("UMA_ADAPTER_ROOTS", str(adapter_root))
    monkeypatch.setattr(sys, "path", list(sys.path))

    UMAConfig.load_yaml(str(cfg_path))

    plugin = parse_plugin_spec("vector.qdrant_adapter:QdrantIndex")
    assert plugin.__module__ == "vector.qdrant_adapter"
    assert str(adapter_root) in sys.path


def test_load_yaml_keeps_config_adjacent_extensions_supported(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    config_root = project_root / "config"
    db_root = tmp_path / "db"
    config_root.mkdir(parents=True, exist_ok=True)
    db_root.mkdir(parents=True, exist_ok=True)
    _write_vector_adapter(project_root / "extensions", marker="config-adjacent")

    cfg = build_test_config(db_root=db_root)
    cfg["storage"]["vector_backend"] = "vector.qdrant_adapter:QdrantIndex"

    cfg_path = config_root / "uma.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.delenv("UMA_ADAPTER_ROOTS", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _clear_plugin_module("vector")

    UMAConfig.load_yaml(str(cfg_path))

    plugin = parse_plugin_spec("vector.qdrant_adapter:QdrantIndex")
    assert plugin.MARKER == "config-adjacent"
    assert str(project_root / "extensions") in sys.path


def test_from_yaml_resolves_external_vector_adapter_without_repo_layout(tmp_path, monkeypatch):
    adapter_root = tmp_path / "external_root"
    runtime_root = tmp_path / "separate_runtime"
    db_root = tmp_path / "db"
    runtime_root.mkdir(parents=True, exist_ok=True)
    db_root.mkdir(parents=True, exist_ok=True)
    _write_vector_adapter(adapter_root)

    cfg = build_test_config(db_root=db_root)
    cfg["storage"]["vector_backend"] = "vector.qdrant_adapter:QdrantIndex"

    cfg_path = runtime_root / "uma.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("UMA_ADAPTER_ROOTS", str(adapter_root))
    monkeypatch.setattr(sys, "path", list(sys.path))

    memory = UMAMemory.from_yaml(str(cfg_path))
    try:
        episodic_index = memory._stores["episodic"].vector_index
        assert type(episodic_index).__module__ == "vector.qdrant_adapter"
        assert str(runtime_root / "extensions") not in sys.path
    finally:
        memory.shutdown()


def test_env_adapter_roots_override_config_adjacent_and_preserve_declared_order(tmp_path, monkeypatch):
    env_root_first = tmp_path / "env_root_first"
    env_root_second = tmp_path / "env_root_second"
    project_root = tmp_path / "project"
    config_root = project_root / "config"
    db_root = tmp_path / "db"
    config_root.mkdir(parents=True, exist_ok=True)
    db_root.mkdir(parents=True, exist_ok=True)

    _write_vector_adapter(env_root_first, module_name="priority_adapter", class_name="PriorityIndex", marker="env-first")
    _write_vector_adapter(env_root_second, module_name="priority_adapter", class_name="PriorityIndex", marker="env-second")
    _write_vector_adapter(project_root / "extensions", module_name="priority_adapter", class_name="PriorityIndex", marker="config-adjacent")

    cfg = build_test_config(db_root=db_root)
    cfg["storage"]["vector_backend"] = "vector.priority_adapter:PriorityIndex"

    cfg_path = config_root / "uma.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv(
        "UMA_ADAPTER_ROOTS",
        os.pathsep.join((str(env_root_first), str(env_root_second))),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    _clear_plugin_module("vector")

    UMAConfig.load_yaml(str(cfg_path))

    plugin = parse_plugin_spec("vector.priority_adapter:PriorityIndex")
    assert plugin.MARKER == "env-first"
    assert sys.path.index(str(env_root_first)) < sys.path.index(str(env_root_second))
    assert sys.path.index(str(env_root_second)) < sys.path.index(str(project_root / "extensions"))
