"""
The decorator surface. This is the entire public vocabulary an application
developer needs:

    @derive        -- pure function of its Traced inputs
    @oracle        -- calls an LLM (or anything else non-deterministic you're
                       willing to call "the model")
    @tool          -- calls an external system (HTTP, DB, ...)
    @human         -- takes human input
    @irreversible  -- modifier: this call's effect can't be undone, only compensated
    @compensable   -- modifier: this call's effect can be undone by an explicit compensation

Every one of these returns a _SevenboxFunction, a callable that:
  - requires an active SevenboxSession (raises a clear error otherwise)
  - reads parent_ids from whichever of its arguments are Traced
  - in live mode, calls the wrapped function directly
  - in replay mode (for oracle/tool/human), never calls the wrapped function --
    looks the cached response up via session.replay_lookup() instead
  - wraps the result as a new Warrant and returns it as a new Traced
  - returns plain None (not a Traced) if the wrapped function returns None,
    which application code can use for ordinary Python control flow
    (`if action: ...`) instead of anything graph-shaped

Every decorator also accepts `redaction=RedactionPolicy(...)` (warrant_core),
applied to the EffectRecord this call produces. Optional, defaults to None --
existing code that never passes it is byte-for-byte unaffected. Fault recovery
is deliberately NOT built into this layer -- see warrant_runtime.failures for
why that's scoped to the Runtime specifically.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from warrant_core import (
    EffectRecord,
    EpistemicClass,
    Origin,
    RedactionPolicy,
    ReversibilityClass,
    SourceKind,
    apply_redaction,
    create_warrant,
)
from warrant_core import PURE_SOURCE_KINDS

from .session import require_session
from .traced import Traced, collect_parent_ids, unwrap_args, unwrap_kwargs


@dataclasses.dataclass(frozen=True)
class _SevenboxFunction:
    fn: Callable
    source_kind: SourceKind
    epistemic: EpistemicClass
    reversibility: ReversibilityClass
    name: str
    redaction: Optional[RedactionPolicy] = None
    origin: Optional[Origin] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Traced]:
        session = require_session()
        parent_ids = collect_parent_ids(args, kwargs)
        raw_args = unwrap_args(args)
        raw_kwargs = unwrap_kwargs(kwargs)
        # Per-call origin overrides the session default, which overrides None
        # -- same precedence as session.axiom()'s own origin parameter.
        effective_origin = self.origin if self.origin is not None else session.origin

        if self.source_kind in PURE_SOURCE_KINDS:
            result = self.fn(*raw_args, **raw_kwargs)
            if result is None:
                return None
            w = create_warrant(
                content=result, source_kind=self.source_kind,
                epistemic=self.epistemic, reversibility=self.reversibility,
                parent_ids=parent_ids, origin=effective_origin,
            )
        else:
            request = {"args": raw_args, "kwargs": raw_kwargs}
            if session.mode == "live":
                result = self.fn(*raw_args, **raw_kwargs)
                if result is None:
                    return None
                raw_response = result
            else:
                cached = session.replay_lookup(self.name, parent_ids)
                result = cached.content
                raw_response = cached.raw
            w = create_warrant(
                content=result, source_kind=self.source_kind,
                epistemic=self.epistemic, reversibility=self.reversibility,
                parent_ids=parent_ids, origin=effective_origin,
                effect_record=EffectRecord(
                    request=apply_redaction(request, self.redaction),
                    response=apply_redaction(raw_response, self.redaction),
                    source_name=self.name,
                    redacted=self.redaction is not None,
                ),
            )

        session.store.add(w)
        return Traced(result, w.id)

    def with_reversibility(self, reversibility: ReversibilityClass) -> "_SevenboxFunction":
        return dataclasses.replace(self, reversibility=reversibility)


def _make_decorator(source_kind: SourceKind, default_epistemic: EpistemicClass):
    def decorator(
        fn=None, *, epistemic: EpistemicClass = default_epistemic,
        reversibility: ReversibilityClass = ReversibilityClass.PURE,
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
    ):
        def wrap(f: Callable) -> _SevenboxFunction:
            return _SevenboxFunction(
                f, source_kind, epistemic, reversibility, f.__qualname__, redaction, origin
            )
        return wrap(fn) if fn is not None else wrap
    return decorator


derive = _make_decorator(SourceKind.DERIVED, EpistemicClass.VERIFIED)
oracle = _make_decorator(SourceKind.ORACLE, EpistemicClass.INFERRED)
tool = _make_decorator(SourceKind.TOOL, EpistemicClass.VERIFIED)
human = _make_decorator(SourceKind.HUMAN, EpistemicClass.ASSUMED)


def irreversible(sevenbox_fn: _SevenboxFunction) -> _SevenboxFunction:
    if not isinstance(sevenbox_fn, _SevenboxFunction):
        raise TypeError(
            "@irreversible must wrap a @derive/@oracle/@tool/@human-decorated "
            "function, e.g.:\n  @irreversible\n  @derive\n  def refund(...): ..."
        )
    return sevenbox_fn.with_reversibility(ReversibilityClass.IRREVERSIBLE)


def compensable(sevenbox_fn: _SevenboxFunction) -> _SevenboxFunction:
    if not isinstance(sevenbox_fn, _SevenboxFunction):
        raise TypeError(
            "@compensable must wrap a @derive/@oracle/@tool/@human-decorated function."
        )
    return sevenbox_fn.with_reversibility(ReversibilityClass.COMPENSABLE)
