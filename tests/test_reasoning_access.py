"""Tests for access control rules (access.mg)."""
from __future__ import annotations

from pathlib import Path

from pymangle.ast_nodes import Constant, TermType
from pymangle.engine import eval_program
from pymangle.external import ExternalPredicateRegistry
from pymangle.parser import parse

_RULES_DIR = Path(__file__).parent.parent / "agentic_graph_rag" / "reasoning" / "rules"


def _eval_access(extra_facts: str = "") -> object:
    """Parse access.mg with extra facts and evaluate."""
    source = (_RULES_DIR / "access.mg").read_text()
    full = extra_facts + "\n" + source
    program = parse(full)
    return eval_program(program)


class TestAccessRules:
    def test_permit_viewer_read_public(self):
        """Viewer can read public resources."""
        store = _eval_access('user_role("alice", /viewer).')
        allowed = list(store.get_by_predicate("allowed"))
        # alice as viewer should be allowed to read public
        alice_perms = [
            a for a in allowed
            if a.args[0].value == "alice"
            and a.args[1].value == "read"
            and a.args[2].value == "public"
        ]
        assert len(alice_perms) == 1

    def test_deny_overrides_permit(self):
        """Deny rule prevents write to pii even for admin."""
        store = _eval_access('user_role("admin_user", /admin).')
        allowed = list(store.get_by_predicate("allowed"))
        # admin should NOT be allowed to write pii (deny overrides)
        write_pii = [
            a for a in allowed
            if a.args[1].value == "write"
            and a.args[2].value == "pii"
        ]
        assert len(write_pii) == 0

    def test_role_inheritance(self):
        """Admin inherits analyst → viewer roles. Can read public and sensitive."""
        store = _eval_access('user_role("boss", /admin).')
        allowed = list(store.get_by_predicate("allowed"))
        boss_perms = {
            (a.args[1].value, a.args[2].value)
            for a in allowed if a.args[0].value == "boss"
        }
        # Admin inherits: viewer→read/public, analyst→read/sensitive, admin→read/pii
        assert ("read", "public") in boss_perms
        assert ("read", "sensitive") in boss_perms
        assert ("read", "pii") in boss_perms

    def test_viewer_cannot_read_sensitive(self):
        """Viewer has no permission for sensitive resources."""
        store = _eval_access('user_role("viewer_user", /viewer).')
        allowed = list(store.get_by_predicate("allowed"))
        sensitive = [
            a for a in allowed
            if a.args[0].value == "viewer_user"
            and a.args[2].value == "sensitive"
        ]
        assert len(sensitive) == 0

    def test_denied_query_audit(self):
        """Denied actions are recorded for audit."""
        store = _eval_access('user_role("viewer_user", /viewer).')
        denied = list(store.get_by_predicate("denied_query"))
        # Viewer should have denied entries for actions they can't do
        viewer_denied = [
            d for d in denied if d.args[0].value == "viewer_user"
        ]
        assert len(viewer_denied) > 0
