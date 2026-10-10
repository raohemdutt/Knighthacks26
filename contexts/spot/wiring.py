"""Builds the inspection context and wires it to the bus (called from contexts/gateway/bootstrap.py).

    spot = build_spot(data_dir, bus, runtime, approvals)
    spot.read, spot.live, spot.player, spot.assistant, spot.orders

- `spot.` events rebuild the live picture and the work-order list from the log (replay=True).
- The work-order executor is live only and only ever runs after an approval.
- The kit folder is ATLAS_SPOT_KIT, else fixtures/spot/kit. Without a kit everything still builds and answers
  "not configured".
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from contexts.spot.application.assistant import Assistant
from contexts.spot.application.replay import Player, SpotLive, build_stream
from contexts.spot.application.service import Spot
from contexts.spot.application.workorders import KIND, WorkOrders
from contexts.spot.infrastructure.kit import kit_path, load_kit

log = logging.getLogger("atlas.spot")


@dataclass
class SpotContext:
    read: Spot
    live: SpotLive
    player: Player
    assistant: Assistant
    orders: WorkOrders
    runtime: object | None = None


def build_spot(data_dir: str | Path, bus, runtime=None, approvals=None, kit_folder: str | Path | None = None) -> SpotContext:
    try:
        kit = load_kit(kit_folder)
    except Exception:  # a broken kit must never stop the box from booting
        log.exception("SPOT kit at %s could not be loaded", kit_path(kit_folder))
        kit = None
    if kit is None:
        log.info("no SPOT kit at %s: the inspection screen will say how to add one", kit_path(kit_folder))
    read = Spot(kit)
    stream = build_stream(kit) if kit else []
    live = SpotLive(kit, stream)
    player = Player(kit, stream, bus, live)

    def publish(event) -> None:  # the executor runs inside the loop, in a background handler
        asyncio.get_running_loop().create_task(bus.publish(event))

    orders = WorkOrders(read, data_dir, publish)
    bus.subscribe("spot.work_order.created", orders.on_created, replay=True)
    bus.subscribe("spot.", live.on_event, replay=True)  # state only: the picture rebuilds from the log
    if approvals is not None:  # live only, and only after a person approves the card
        approvals.executors[KIND] = orders.executor(approvals.executors.get(KIND))
    return SpotContext(read=read, live=live, player=player, assistant=Assistant(read, runtime), orders=orders, runtime=runtime)
