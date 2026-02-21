"""Tests for trace store backends."""
from rag_core.models import PipelineTrace

from agentic_graph_rag.trace_store import InMemoryTraceStore, create_trace_store


def _make_trace(trace_id: str) -> PipelineTrace:
    return PipelineTrace(trace_id=trace_id, timestamp="T", query="q")


def test_in_memory_put_and_get():
    store = InMemoryTraceStore()
    trace = _make_trace("tr_1")
    store.put(trace)
    assert store.get("tr_1") is not None
    assert store.get("tr_1").trace_id == "tr_1"


def test_in_memory_get_missing():
    store = InMemoryTraceStore()
    assert store.get("nonexistent") is None


def test_in_memory_eviction():
    store = InMemoryTraceStore(max_size=3)
    for i in range(5):
        store.put(_make_trace(f"tr_{i}"))

    # Oldest should be evicted
    assert store.get("tr_0") is None
    assert store.get("tr_1") is None
    # Newest should remain
    assert store.get("tr_2") is not None
    assert store.get("tr_3") is not None
    assert store.get("tr_4") is not None


def test_create_trace_store_defaults_to_in_memory():
    store = create_trace_store()
    assert isinstance(store, InMemoryTraceStore)
