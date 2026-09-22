"""User-split validation shared by PERMA Phase-2 experiments."""

from __future__ import annotations

from data_adapter import ALL_USER_IDS


def resolve_user_split(
    test_user: int | None,
    requested_eval_users: list[int] | None,
    requested_train_users: list[int] | None,
) -> tuple[set[int], set[int]]:
    eval_user_values = (
        [test_user] if test_user is not None else requested_eval_users
    )
    if not eval_user_values:
        raise ValueError("at least one Phase-2 evaluation user is required")
    eval_users = set(eval_user_values)
    if len(eval_users) != len(eval_user_values):
        raise ValueError("evaluation users contain duplicate user IDs")
    unknown_eval = sorted(eval_users - set(ALL_USER_IDS))
    if unknown_eval:
        raise ValueError(f"unknown PERMA evaluation users: {unknown_eval}")

    if requested_train_users is None:
        train_users = set(ALL_USER_IDS) - eval_users
        if not train_users:
            raise ValueError("at least one Phase-2 training user is required")
        return train_users, eval_users
    train_users = set(requested_train_users)
    if not train_users:
        raise ValueError("at least one Phase-2 training user is required")
    unknown = sorted(train_users - set(ALL_USER_IDS))
    if unknown:
        raise ValueError(f"unknown PERMA training users: {unknown}")
    overlap = sorted(train_users & eval_users)
    if overlap:
        raise ValueError(f"training and evaluation users overlap: {overlap}")
    if len(train_users) != len(requested_train_users):
        raise ValueError("--train_users contains duplicate user IDs")
    return train_users, eval_users
