"""
Traced -- a value with a Warrant id attached.

This is the whole trick that makes "creating Warrants almost invisible" possible:
every Sevenbox-decorated function returns a Traced instead of a raw value, and every
Sevenbox-decorated function accepts Traced arguments and automatically reads their
`warrant_id` as parent_ids for the new Warrant it's about to create. The
application developer never writes `parent_ids=[...]` anywhere -- it falls out of
which Traced values they happened to pass into which function call.

Traced supports __getitem__ so that projecting into a structured result (e.g.
`intent["order_id"]`) still carries the same warrant_id forward -- it's a view,
not new evidence, so it doesn't need a new Warrant of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Traced:
    value: Any
    warrant_id: str

    def __getitem__(self, key: Any) -> "Traced":
        return Traced(self.value[key], self.warrant_id)

    def unwrap(self) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"Traced({self.value!r}, warrant_id={self.warrant_id[:8]}...)"


def unwrap_one(value: Any) -> Any:
    return value.value if isinstance(value, Traced) else value


def unwrap_args(args: tuple) -> tuple:
    return tuple(unwrap_one(a) for a in args)


def unwrap_kwargs(kwargs: dict) -> dict:
    return {k: unwrap_one(v) for k, v in kwargs.items()}


def collect_parent_ids(args: tuple, kwargs: dict) -> list:
    ids = [a.warrant_id for a in args if isinstance(a, Traced)]
    ids += [v.warrant_id for v in kwargs.values() if isinstance(v, Traced)]
    if not ids:
        raise ValueError(
            "No Traced (Warrant-backed) argument was passed to this call -- every "
            "non-axiom Sevenbox function needs at least one Traced input for causal "
            "grounding. Did you forget to start from session.axiom(...)?"
        )
    return ids
