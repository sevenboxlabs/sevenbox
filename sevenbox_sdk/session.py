"""
SevenboxSession -- an ambient context that Sevenbox-decorated functions find automatically
via a contextvar, so application code never has to thread a store/executor through
every call by hand. This is the other half of "make creating Warrants almost
invisible": no explicit store parameter anywhere in application code.

    with SevenboxSession() as session:
        ticket = session.axiom({"ticket": 91})
        ...

    with SevenboxSession(replay_from=previous_session.store) as session:
        ...  # same code, no real effects are invoked

Reuses warrant_runtime.effects.ReplayEffectExecutor for replay lookups rather than
reimplementing the parent-ids matching logic -- same semantics, same documented
limitation (matching by parent_ids only), as Phase 2.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional

from warrant_core import (
    EpistemicClass,
    Origin,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    compute_frontier,
    create_warrant,
    find_transitive_obligations,
)
from warrant_runtime import ReplayEffectExecutor

from .traced import Traced

_current_session: ContextVar[Optional["SevenboxSession"]] = ContextVar("sevenbox_session", default=None)


def require_session() -> "SevenboxSession":
    session = _current_session.get()
    if session is None:
        raise RuntimeError(
            "No active SevenboxSession. Wrap your code in `with SevenboxSession() as session:` "
            "before calling any @derive/@oracle/@tool/@human function."
        )
    return session


class SevenboxSession:
    def __init__(
        self,
        *,
        store: Optional[WarrantStore] = None,
        replay_from: Optional[WarrantStore] = None,
        origin: Optional[Origin] = None,
    ) -> None:
        self.store = store if store is not None else WarrantStore()
        self.mode = "replay" if replay_from is not None else "live"
        self._replay_executor = (
            ReplayEffectExecutor(reference_store=replay_from) if replay_from is not None else None
        )
        # Session-wide default Origin (org/team/agent/run) -- applied to
        # every Warrant produced through this session, including via
        # session.axiom() and every @derive/@oracle/@tool/@human call, unless
        # a decorated function overrides it per-call (see decorators.py).
        # None here means "no attribution," exactly today's behavior.
        self.origin = origin
        self._token = None

    def axiom(
        self,
        content: Any,
        *,
        epistemic: EpistemicClass = EpistemicClass.VERIFIED,
        reversibility: ReversibilityClass = ReversibilityClass.PURE,
        origin: Optional[Origin] = None,
    ) -> Traced:
        w = create_warrant(
            content=content, source_kind=SourceKind.AXIOM,
            epistemic=epistemic, reversibility=reversibility,
            origin=origin if origin is not None else self.origin,
        )
        self.store.add(w)
        return Traced(content, w.id)

    def replay_lookup(self, rule_name: str, parent_ids: list):
        assert self._replay_executor is not None, "replay_lookup() called outside replay mode"
        return self._replay_executor.invoke(rule_name, parent_ids, request=None)

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)

    def __enter__(self) -> "SevenboxSession":
        self._token = _current_session.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_session.reset(self._token)
        return False
