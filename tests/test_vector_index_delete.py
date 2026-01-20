from uma3.adapters.vector.inmemory import InMemoryVectorIndex


def test_inmemory_delete_removes_vectors():
    idx = InMemoryVectorIndex(dim=3)
    idx.upsert(ids=["a", "b"], vectors=[[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]])

    assert "a" in idx._vectors
    idx.delete(["a"])
    assert "a" not in idx._vectors
    assert "b" in idx._vectors
