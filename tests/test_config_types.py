from uma3.core.config_types import RetrievalConfig


def test_retrieval_config_from_dict_with_rlm():
    cfg = RetrievalConfig.from_dict(
        {
            "max_episodes": 2,
            "max_facts": 3,
            "max_skills": 4,
            "max_graph_items": 5,
            "rlm": {
                "enabled": True,
                "max_steps": 7,
            },
        }
    )

    assert cfg.max_episodes == 2
    assert cfg.max_facts == 3
    assert cfg.max_skills == 4
    assert cfg.max_graph_items == 5
    assert cfg.rlm is not None
    assert cfg.rlm.enabled is True
    assert cfg.rlm.max_steps == 7
    # Defaults preserved
    assert cfg.rlm.max_actions_per_step == 2
    assert cfg.rlm.max_items_per_type == 30
