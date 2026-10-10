from modules.util.enum.QueueEntryStatus import QueueEntryStatus
from modules.util.queue.QueueManager import QueueManager


def _manager(tmp_path):
    m = QueueManager(str(tmp_path / "queue.json"))
    done = m.add_entry("done")
    failed = m.add_entry("failed")
    skipped = m.add_entry("skipped")
    pending = m.add_entry("pending")
    running = m.add_entry("running")
    m.set_status(done.id, QueueEntryStatus.COMPLETED)
    m.set_status(failed.id, QueueEntryStatus.FAILED)
    failed.failure_history.append({"error": "boom"})
    m.set_status(skipped.id, QueueEntryStatus.SKIPPED)
    m.set_status(running.id, QueueEntryStatus.RUNNING)
    return m, done, failed, skipped, pending, running


def test_reset_unfinished_keeps_completed_entries_done(tmp_path):
    m, done, failed, skipped, pending, running = _manager(tmp_path)
    m.reset_unfinished_to_pending()
    assert done.status == QueueEntryStatus.COMPLETED
    assert failed.status == QueueEntryStatus.PENDING
    assert failed.failure_history == []
    assert skipped.status == QueueEntryStatus.PENDING
    assert pending.status == QueueEntryStatus.PENDING
    assert running.status == QueueEntryStatus.RUNNING
    # The stale RUNNING entry resumes first, then the pending ones in order.
    assert m.get_next_runnable() is running


def test_reset_entry_marks_finished_entry_untrained(tmp_path):
    m, done, failed, skipped, pending, running = _manager(tmp_path)
    assert m.reset_entry(done.id)
    assert done.status == QueueEntryStatus.PENDING
    assert m.reset_entry(failed.id)
    assert failed.failure_history == []
    assert not m.reset_entry(running.id), "a running entry cannot be reset"
    assert running.status == QueueEntryStatus.RUNNING
    assert not m.reset_entry("missing")
    # Persisted: a fresh manager sees the reset.
    assert QueueManager(m.queue_file_path).get_entry(done.id).status == QueueEntryStatus.PENDING


def test_reset_all_resets_every_finished_entry(tmp_path):
    m, done, failed, skipped, pending, running = _manager(tmp_path)
    m.reset_all_to_pending()
    assert {e.status for e in (done, failed, skipped, pending)} == {QueueEntryStatus.PENDING}
    assert running.status == QueueEntryStatus.RUNNING
