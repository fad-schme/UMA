from uma.core.utils.registry import FeatureLoader, FeaturePolicy, FeatureRegistry, FeatureSpec
from uma.features.procedural.feature import ProceduralFeature


class DummyMemory:
    def __init__(self) -> None:
        self.features = {}
        self._methods = {}

    def register_methods(self, feature_name, methods, allow_override=None) -> None:
        for name, func in methods.items():
            if name in self._methods:
                raise ValueError(f"duplicate method: {name}")
            self._methods[name] = func
            setattr(self, name, func)


class StubProceduralCore:
    async def add_skill(self, skill, embedding):
        return None

    async def search(self, query_embedding, k=5, owner_type=None, owner_id=None):
        return []

    async def get_skill(self, skill_id):
        return None


class StubEmbedder:
    async def embed(self, texts):
        return [[0.0] * 3 for _ in texts]


def test_feature_loader_registers_only_attached_features():
    registry = FeatureRegistry()
    registry.register(FeatureSpec(name="procedural", provider=ProceduralFeature))

    loader = FeatureLoader(registry, FeaturePolicy())
    memory = DummyMemory()

    loader.load_from_config(
        memory_client=memory,
        feature_cfgs=[
            {"name": "procedural", "enabled": True, "config": {"max_k": 3}},
        ],
        services={"procedural_core": StubProceduralCore(), "embedder": StubEmbedder()},
    )

    assert "procedural" in memory.features
    assert callable(getattr(memory, "procedural_health"))


def test_feature_loader_skips_failed_attachment():
    registry = FeatureRegistry()
    registry.register(FeatureSpec(name="procedural", provider=ProceduralFeature))

    loader = FeatureLoader(registry, FeaturePolicy())
    memory = DummyMemory()

    loader.load_from_config(
        memory_client=memory,
        feature_cfgs=[
            {"name": "procedural", "enabled": True, "config": {"max_k": 3}},
        ],
        services={"procedural_core": StubProceduralCore(), "embedder": None},
    )

    assert "procedural" not in memory.features
