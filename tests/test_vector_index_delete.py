from uma.adapters.vector.inmemory import InMemoryVectorIndex


def test_inmemory_delete_removes_vectors():
    idx = InMemoryVectorIndex(dim=3)
    idx.upsert(
        ids=["a", "b"],
        vectors=[[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]],
        tenant_ids=["default", "default"],
        owner_types=["user", "user"],
        owner_ids=["user:u1", "user:u1"],
        extra_metadata=[{}, {}],
    )

    assert "a" in idx._vectors
    idx.delete(["a"])
    assert "a" not in idx._vectors
    assert "b" in idx._vectors
