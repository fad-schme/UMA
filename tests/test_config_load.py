from uma.common.config import UMAConfig
import yaml


def test_config_load_uses_repo_yaml():
    cfg = UMAConfig.load_yaml("config/uma.yaml")
    assert cfg.storage.db_root
    assert cfg.working_memory.max_tokens > 0
    assert cfg.retrieval.max_episodes > 0


def test_repo_config_does_not_contain_committed_secrets():
    config_text = open("config/uma.yaml", "r", encoding="utf-8").read()
    assert "PRIVATE KEY" not in config_text

    data = yaml.safe_load(config_text)
    sensitive_keys = {"password", "api_key", "token", "secret"}
    allowed_empty = {"", None}
    allowed_placeholders = {
        "<set-me>",
        "${UMA_GRAPH_PASSWORD}",
        "${UMA_LLM_API_KEY}",
        "${UMA_VECTOR_API_KEY}",
    }

    def _walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                key_str = str(key)
                child_path = f"{path}.{key_str}" if path else key_str
                if key_str.lower() in sensitive_keys:
                    if isinstance(value, str):
                        normalized = value.strip()
                    else:
                        normalized = value
                    assert normalized in allowed_empty or normalized in allowed_placeholders, (
                        f"config/uma.yaml contains a committed sensitive value at {child_path}"
                    )
                _walk(value, child_path)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                _walk(item, f"{path}[{index}]")

    _walk(data)
