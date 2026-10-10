"""Telemetry: what Atlas costs to run, per node and per answer.

Samplers publish one event type per source (`contracts/telemetry.py`) onto the bus; the
projection folds them per node and is rebuilt from the event log. Knows nothing about what a
frame or an answer contains.

    domain/          the known nodes; each source's sample converted to watts in one place; reading the
                     Jetson's tegrastats and nvpmodel output
    application/     the projection, the sampler runner, the service the app runs; the Jetson power sampler
                     and the measurement of its CPU cost
    infrastructure/  recorded telemetry files; what machine this is; the Jetson's tools and sysfs files; the
                     pod power samplers read through the hub proxy
    fakes/           a sampler that replays recorded telemetry as live samples; stand-ins for tegrastats
                     and nvpmodel that replay recorded output
    wiring.py        build_telemetry(source, bus), used by the gateway when ATLAS_TELEMETRY_SOURCE is set
                     ("fake", "tegrastats" on the Jetson, or "pods" for the GPU pods' power samplers)
    jetson_power.py  python -m contexts.telemetry.jetson_power probe|cpu: the Jetson sampler's first run on the box
"""
