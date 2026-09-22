"""Deterministic training-data positions for exact checkpoint resume."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoaderPosition:
    """Position immediately after the last completed micro-batch."""

    epoch: int
    batches_consumed_in_epoch: int
    batches_per_epoch: int

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("data epoch must be non-negative")
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        if not 0 <= self.batches_consumed_in_epoch < self.batches_per_epoch:
            raise ValueError(
                "batches_consumed_in_epoch must be in "
                f"[0, {self.batches_per_epoch}), got "
                f"{self.batches_consumed_in_epoch}"
            )

    @classmethod
    def from_micro_batches(
        cls,
        micro_batches: int,
        *,
        batches_per_epoch: int,
    ) -> "DataLoaderPosition":
        micro_batches = int(micro_batches)
        batches_per_epoch = int(batches_per_epoch)
        if micro_batches < 0:
            raise ValueError("micro_batches must be non-negative")
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        epoch, offset = divmod(micro_batches, batches_per_epoch)
        return cls(epoch, offset, batches_per_epoch)

    @classmethod
    def from_state_dict(
        cls,
        state: dict,
        *,
        expected_batches_per_epoch: int,
    ) -> "DataLoaderPosition":
        stored_batches_per_epoch = int(state["batches_per_epoch"])
        if stored_batches_per_epoch != int(expected_batches_per_epoch):
            raise ValueError(
                "checkpoint batches_per_epoch does not match this run: "
                f"stored={stored_batches_per_epoch}, "
                f"current={expected_batches_per_epoch}"
            )
        return cls(
            epoch=int(state["data_epoch"]),
            batches_consumed_in_epoch=int(state["batches_consumed_in_epoch"]),
            batches_per_epoch=stored_batches_per_epoch,
        )

    def state_dict(self) -> dict[str, int]:
        return {
            "data_epoch": self.epoch,
            "batches_consumed_in_epoch": self.batches_consumed_in_epoch,
            "batches_per_epoch": self.batches_per_epoch,
        }

    @property
    def total_micro_batches(self) -> int:
        return self.epoch * self.batches_per_epoch + self.batches_consumed_in_epoch
