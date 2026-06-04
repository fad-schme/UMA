from pathlib import Path

import yaml


def test_qdrant_compose_file_exists() -> None:
    compose_path = Path("docker/uma_cont/docker-compose.yml")
    assert compose_path.exists()


def test_qdrant_compose_shape() -> None:
    compose_path = Path("docker/uma_cont/docker-compose.yml")
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert isinstance(data, dict)
    services = data.get("services")
    assert isinstance(services, dict)
    assert "qdrant" in services

    qdrant_service = services["qdrant"]
    assert qdrant_service["image"] == "qdrant/qdrant:latest"
    assert "6333:6333" in qdrant_service.get("ports", [])
    assert "6334:6334" in qdrant_service.get("ports", [])
    assert "qdrant_storage:/qdrant/storage" in qdrant_service.get("volumes", [])

    volumes = data.get("volumes")
    assert isinstance(volumes, dict)
    assert "qdrant_storage" in volumes
