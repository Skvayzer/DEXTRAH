"""Stateful utilities for ADEPT's success-driven ADR curriculum."""

from __future__ import annotations

from typing import Any

import torch


class RollingSuccessRate:
    """Fixed-size ring buffer of completed-episode success outcomes."""

    def __init__(self, capacity: int, device: str | torch.device):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.values = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        self.cursor = 0
        self.count = 0

    @property
    def ready(self) -> bool:
        return self.count == self.capacity

    @property
    def rate(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.values[: self.count].float().mean().item())

    def update(self, outcomes: torch.Tensor) -> None:
        outcomes = outcomes.detach().to(device=self.values.device, dtype=torch.bool).flatten()
        if outcomes.numel() == 0:
            return
        if outcomes.numel() >= self.capacity:
            self.values.copy_(outcomes[-self.capacity :])
            self.cursor = 0
            self.count = self.capacity
            return

        count = outcomes.numel()
        first = min(count, self.capacity - self.cursor)
        self.values[self.cursor : self.cursor + first].copy_(outcomes[:first])
        remainder = count - first
        if remainder:
            self.values[:remainder].copy_(outcomes[first:])
        self.cursor = (self.cursor + count) % self.capacity
        self.count = min(self.capacity, self.count + count)

    def clear(self) -> None:
        self.values.zero_()
        self.cursor = 0
        self.count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "values": self.values.detach().cpu(),
            "cursor": self.cursor,
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"success-window capacity changed from {state['capacity']} to {self.capacity}"
            )
        values = torch.as_tensor(state["values"], dtype=torch.bool, device=self.values.device)
        if values.shape != self.values.shape:
            raise ValueError("invalid success-window tensor shape")
        cursor = int(state["cursor"])
        count = int(state["count"])
        if not 0 <= cursor < self.capacity or not 0 <= count <= self.capacity:
            raise ValueError("invalid success-window cursor or count")
        self.values.copy_(values)
        self.cursor = cursor
        self.count = count
