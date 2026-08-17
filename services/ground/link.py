"""The deep-space link — the ONLY place the Earth<->Mars light-time delay lives.

Every uplink (operator -> MARVIN) and downlink (MARVIN -> console) is transmitted through here and
pays the delay. HOUSTON<->operator never touches this (both are on Earth). Keeping the delay in one
module is what makes the invariant auditable: if it's not routed through a DeepSpaceLink, it didn't
cross space.

The delay is a *compressed* light-time — real one-way Mars is 4-24 minutes; we run ~12 s so a demo
is watchable. The UI must always label it as compressed, never claim real-time Mars.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Transmission:
    """One packet in flight across the link."""
    payload: Any
    direction: str                 # "uplink" (Earth->Mars) or "downlink" (Mars->Earth)
    sent_at: float
    deliver_at: float
    kind: str = "message"          # message | briefing | telemetry | panel | deviation | hazcam | log

    @property
    def in_flight_s(self) -> float:
        return max(0.0, self.deliver_at - time.time())


@dataclass
class DeepSpaceLink:
    """A one-way-delay channel with a FIFO 'channel busy' model (there is one rover, one antenna).

    `delay_s` is applied to BOTH directions. `transmit` awaits the light-time then hands the packet
    to `on_arrival`. `queue_depth` lets the UI show 'N transmissions ahead of you'.
    """
    delay_s: float = 12.0
    downlink_handlers: list[Callable[[Transmission], Awaitable[None]]] = field(default_factory=list)
    _busy_until: float = 0.0
    _uplink_depth: int = 0

    def on_downlink(self, handler: Callable[[Transmission], Awaitable[None]]) -> None:
        """Register a coroutine that receives each downlink packet as it arrives on Earth."""
        self.downlink_handlers.append(handler)

    @property
    def queue_depth(self) -> int:
        return self._uplink_depth

    async def uplink(self, payload: Any, kind: str = "message") -> Transmission:
        """Earth -> Mars. FIFO: a new transmission waits behind whatever is still crossing."""
        now = time.time()
        start = max(now, self._busy_until)          # channel busy until the one ahead clears
        self._busy_until = start + self.delay_s
        self._uplink_depth += 1
        tx = Transmission(payload, "uplink", now, start + self.delay_s, kind)
        try:
            await asyncio.sleep(max(0.0, tx.deliver_at - now))
        finally:
            self._uplink_depth -= 1
        return tx

    async def downlink(self, payload: Any, kind: str = "message") -> Transmission:
        """Mars -> Earth. Delivered to every registered handler once the light-time has elapsed."""
        now = time.time()
        tx = Transmission(payload, "downlink", now, now + self.delay_s, kind)
        await asyncio.sleep(self.delay_s)
        for handler in self.downlink_handlers:
            await handler(tx)
        return tx
