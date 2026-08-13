import pytest

from groundswarm.memory.schema import MemoryEntry, Scope
from groundswarm.memory.gate import validate
from groundswarm.memory.store import MemoryStore, MemoryWriteRejected


def make_entry(**overrides):
    defaults = dict(
        content="the sky is blue",
        scope=Scope.PROJECT,
        scope_id="proj-1",
        provenance="test-suite",
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


def test_valid_entry_accepted():
    result = validate(make_entry())
    assert result.accepted
    assert result.reason == "ok"


def test_empty_content_rejected():
    result = validate(make_entry(content="   "))
    assert not result.accepted
    assert "content" in result.reason


def test_confidence_out_of_range_rejected():
    result = validate(make_entry(confidence=1.5))
    assert not result.accepted
    assert "confidence" in result.reason


def test_negative_ttl_rejected():
    result = validate(make_entry(ttl_seconds=-10))
    assert not result.accepted
    assert "ttl_seconds" in result.reason


def test_org_scope_without_review_rejected():
    result = validate(make_entry(scope=Scope.ORG, scope_id="org-1", reviewed=False))
    assert not result.accepted
    assert "review" in result.reason


def test_org_scope_with_review_accepted():
    result = validate(make_entry(scope=Scope.ORG, scope_id="org-1", reviewed=True))
    assert result.accepted


def test_store_write_raises_on_invalid_entry():
    store = MemoryStore()
    with pytest.raises(MemoryWriteRejected):
        store.write(make_entry(content=""))


def test_store_write_persists_valid_entry():
    store = MemoryStore()
    entry = make_entry()
    store.write(entry)
    active = store.active_entries(scope=Scope.PROJECT, scope_id="proj-1")
    assert len(active) == 1
    assert active[0].content == "the sky is blue"


def test_supersedes_deactivates_old_entry():
    store = MemoryStore()
    old = make_entry(content="old fact")
    store.write(old)
    new = make_entry(content="corrected fact", supersedes=old.id)
    store.write(new)
    active = store.active_entries(scope=Scope.PROJECT, scope_id="proj-1")
    assert len(active) == 1
    assert active[0].content == "corrected fact"


def test_ttl_expiry_excludes_entry():
    store = MemoryStore()
    entry = make_entry(ttl_seconds=10, created_at=1000.0)
    store.write(entry)
    active_before = store.active_entries(scope=Scope.PROJECT, scope_id="proj-1", now=1005.0)
    active_after = store.active_entries(scope=Scope.PROJECT, scope_id="proj-1", now=1011.0)
    assert len(active_before) == 1
    assert len(active_after) == 0


def test_load_context_always_includes_org_entries():
    store = MemoryStore()
    store.write(make_entry(scope=Scope.ORG, scope_id="org-1", reviewed=True, content="org fact"))
    ctx = store.load_context()
    assert len(ctx["org"]) == 1
    assert ctx["org"][0].content == "org fact"


def test_load_context_budgets_project_entries_by_token_estimate():
    store = MemoryStore()
    long_content = "x" * 4000  # ~1000 estimated tokens
    store.write(make_entry(content=long_content, scope_id="proj-2"))
    store.write(make_entry(content="short fact", scope_id="proj-2"))
    ctx = store.load_context(project_scope_id="proj-2", max_project_tokens=1000)
    assert len(ctx["project"]) == 1
    assert ctx["project"][0].content == long_content
