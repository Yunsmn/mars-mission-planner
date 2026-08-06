"""The proposer — a LOCAL model (Gemma 4 via Ollama) that ONLY proposes action sequences.

It never executes anything. Malformed or invalid proposals are dropped, never crash the loop.
The surrogate + gate are what make trusting a local model acceptable.

TODO(bob): implement (task B9).
"""
from __future__ import annotations

from common.types import ActionSeq, MissionState, Perception


class Proposer:
    def __init__(self, name: str, host: str, temperature: float) -> None:
        self.name = name
        self.host = host
        self.temperature = temperature

    def propose(self, state: MissionState, perception: Perception, k: int) -> list[ActionSeq]:
        """Query Gemma 4 for k candidate action sequences; parse + validate to schema."""
        raise NotImplementedError(
            "TODO(bob): call Ollama, parse structured output, drop invalid candidates"
        )
