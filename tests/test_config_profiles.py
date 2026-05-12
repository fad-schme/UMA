from pathlib import Path

import yaml

from uma.common.config import UMAConfig


def test_config_profile_files_have_expected_shapes() -> None:
    root = Path("config")
    assert (root / "uma.yaml").exists()
    assert (root / "uma_lite.yaml").exists()
    assert (root / "uma_cont.yaml").exists()

    default_raw = yaml.safe_load((root / "uma.yaml").read_text(encoding="utf-8"))
    lite_raw = yaml.safe_load((root / "uma_lite.yaml").read_text(encoding="utf-8"))
    cont_raw = yaml.safe_load((root / "uma_cont.yaml").read_text(encoding="utf-8"))

    default_cfg = UMAConfig.load_yaml(str(root / "uma.yaml"))
    lite_cfg = UMAConfig.load_yaml(str(root / "uma_lite.yaml"))
    cont_cfg = UMAConfig.load_yaml(str(root / "uma_cont.yaml"))

    assert default_cfg.profile == "lite"
    assert lite_cfg.profile == "lite"
    assert cont_cfg.profile == "cont"

    assert default_raw == lite_raw
    assert "agent" not in lite_raw["llms"]
    assert "consolidation" not in lite_raw
    assert "hybrid" not in lite_raw["retrieval"]
    assert "rlm" not in lite_raw["retrieval"]
    assert lite_raw["embedding"]["dimension"] == 768
    assert lite_raw["retrieval"]["context"]["include_graph"] is False

    assert lite_cfg.storage.sql_backend == "sqlite"
    assert lite_cfg.storage.vector_backend == "uma.adapters.vector.lancedb:LanceDBIndex"
    assert lite_cfg.storage.graph_backend == "disabled"
    assert lite_cfg.storage.vector_backend.startswith("uma.adapters.vector.")

    assert cont_cfg.storage.sql_backend == "sqlite"
    assert cont_cfg.storage.vector_backend == "uma.adapters.vector.qdrant:QdrantIndex"
    assert cont_cfg.storage.vector_config["url"] == "http://qdrant:6333"
    assert cont_cfg.storage.vector_config["collection"] == "uma_vectors"
    assert cont_cfg.storage.graph_backend == "disabled"
    assert cont_cfg.storage.vector_backend.startswith("uma.adapters.vector.")
    assert cont_raw["embedding"]["dimension"] == 768
    assert "agent" not in cont_raw["llms"]
    assert "consolidation" not in cont_raw
    assert "hybrid" not in cont_raw["retrieval"]
    assert "rlm" not in cont_raw["retrieval"]
    assert cont_raw["retrieval"]["context"]["include_graph"] is False


def test_missing_profile_defaults_to_lite(tmp_path) -> None:
    cfg_path = tmp_path / "uma_no_profile.yaml"
    data = yaml.safe_load(Path("config/uma_lite.yaml").read_text(encoding="utf-8"))
    data.pop("profile", None)
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    cfg = UMAConfig.load_yaml(str(cfg_path))

    assert cfg.profile == "lite"
