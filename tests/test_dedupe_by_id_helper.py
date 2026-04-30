from uma.common.dedupe import dedupe_by_id


class Obj:
    def __init__(self, id):
        self.id = id


def test_dedupe_by_id_handles_dicts_and_objects():
    items = [{"id": "a"}, Obj("a"), {"id": "b"}, Obj("b"), {"id": "a"}]
    out = dedupe_by_id(items)
    assert [getattr(x, "id", x.get("id")) for x in out] == ["a", "b"]

