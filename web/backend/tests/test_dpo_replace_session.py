import random

from modules.util.dpo_replace_session import SessionStore, weighted_shuffle


def test_weighted_shuffle_is_complete_random_and_mildly_old_biased():
    items = [{"id": i, "created": i} for i in range(30)]
    total_rank = [0] * 30
    for seed in range(500):
        shuffled = weighted_shuffle(items, random.Random(seed))
        assert sorted(item["id"] for item in shuffled) == list(range(30))
        for rank, item in enumerate(shuffled):
            total_rank[item["id"]] += rank
    assert sum(total_rank[:10]) < sum(total_rank[-10:])
    assert weighted_shuffle(items, random.Random(1)) != weighted_shuffle(items, random.Random(2))


def test_transaction_recovers_interrupted_commit(tmp_path):
    store = SessionStore(str(tmp_path))
    image = tmp_path / "pair.png"
    image.write_bytes(b"before")
    (tmp_path / "manifest.json").write_text('{"pairs": []}')
    ledger = {"version": 1, "items": {}}
    store.save(ledger)
    store.begin([str(image)])
    image.write_bytes(b"after")
    (tmp_path / "manifest.json").write_text("broken")
    store.save({**ledger, "changed": True})
    restored = SessionStore(str(tmp_path)).load()
    assert restored == ledger
    assert image.read_bytes() == b"before"
    assert (tmp_path / "manifest.json").read_text() == '{"pairs": []}'


def test_committed_transaction_is_not_rolled_back(tmp_path):
    store = SessionStore(str(tmp_path))
    image = tmp_path / "pair.png"
    image.write_bytes(b"before")
    store.save({"version": 1, "items": {}})
    journal = store.begin([str(image)])
    image.write_bytes(b"after")
    store.commit(journal)
    store.load()
    assert image.read_bytes() == b"after"
