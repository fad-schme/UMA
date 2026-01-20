from uma3.core.memory_config import UMAConfig


def test_config_load_uses_repo_yaml():
    cfg = UMAConfig.load_yaml("config/uma3.yaml")
    assert cfg.storage.db_root
    assert cfg.working_memory.max_tokens > 0
    assert cfg.retrieval.max_episodes > 0
