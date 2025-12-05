import threading
import time

from sentinel.speech.priority_queue import PhraseQueue


def test_lower_priority_number_is_served_first():
    queue = PhraseQueue()
    queue.put("greeting", priority=2)
    queue.put("alert", priority=0)
    queue.put("unknown", priority=1)
    assert [queue.get(0.1) for _ in range(3)] == ["alert", "unknown", "greeting"]


def test_same_priority_is_first_in_first_out():
    queue = PhraseQueue()
    for text in ("a", "b", "c"):
        queue.put(text, priority=1)
    assert [queue.get(0.1) for _ in range(3)] == ["a", "b", "c"]


def test_full_queue_rejects_equal_or_lower_priority():
    queue = PhraseQueue(capacity=2)
    assert queue.put("a", 1)
    assert queue.put("b", 1)
    assert not queue.put("c", 1)
    assert not queue.put("d", 2)
    assert len(queue) == 2


def test_full_queue_evicts_worst_item_for_urgent_phrase():
    queue = PhraseQueue(capacity=2)
    queue.put("old greeting", 2)
    queue.put("unknown", 1)
    assert queue.put("alert", 0)
    assert [queue.get(0.1), queue.get(0.1)] == ["alert", "unknown"]


def test_get_times_out_when_empty():
    started = time.perf_counter()
    assert PhraseQueue().get(timeout=0.05) is None
    assert time.perf_counter() - started < 0.5


def test_get_wakes_when_item_arrives():
    queue = PhraseQueue()
    result = []
    consumer = threading.Thread(target=lambda: result.append(queue.get(timeout=2.0)))
    consumer.start()
    time.sleep(0.05)
    queue.put("hello")
    consumer.join(2.0)
    assert result == ["hello"]
