"""Why a backlog action was refused, without saying how to tell a browser."""

from __future__ import annotations

#: The kinds of refusal. The endpoint maps each to a status code; nothing in
#: this package knows what those are.
KINDS = (
    "invalid",  # the request does not make sense as asked
    "forbidden",  # this person may not do this to this item
    "not_found",  # something the action names does not exist for them
    "conflict",  # the item or slice is not in a state that allows it
    "unknown_user",  # the person to assign to is not somebody
    "unavailable",  # something the server needs is missing
)


class ActionRefused(Exception):
    def __init__(self, kind: str, detail: str):
        if kind not in KINDS:
            raise ValueError(f"unknown refusal kind: {kind}")
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


__all__ = ["KINDS", "ActionRefused"]
