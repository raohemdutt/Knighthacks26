"""Telemetry: what Atlas costs to run, per node and per answer.

Samplers publish one event type per source (`contracts/telemetry.py`) onto the bus; the
projection folds them per node and is rebuilt from the event log. Knows nothing about what a
frame or an answer contains.

    domain/          the known nodes; each source's sample converted to watts in one place
    application/     the projection, the sampler runner, the service the app runs
    infrastructure/  recorded telemetry files; what machine this is
    fakes/           a sampler that replays recorded telemetry as live samples
    wiring.py        build_telemetry(source, bus), used by the gateway when ATLAS_TELEMETRY_SOURCE is set
"""
