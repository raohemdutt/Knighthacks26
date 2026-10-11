"""Telemetry: what Atlas costs to run, per node and per answer.

Samplers publish one event type per source (`contracts/telemetry.py`) onto the bus; the
projection folds them per node and is rebuilt from the event log. Knows nothing about what a
frame or an answer contains.

Nodes: jetson, pod-a (the teacher's pod, hub account C) and pod-b (the trainer's pod, account D).
The spec's gpu-a and gpu-b are pod-a and pod-b (telemetry ruling 3); samples still carrying the old names are read as these.

Joules per answer (telemetry.joules_per_answer): the answering node's power integrated over each request window,
gross, with the node's idle watts beside it, split by output tokens where requests overlap, pod samples moved onto
the Jetson's clock by an offset measured through the hub proxy (telemetry.clock.offset); application/joules.py.

    domain/          the known nodes; each source's sample converted to watts in one place; reading the
                     Jetson's tegrastats and nvpmodel output; energy over a window, its split, the clock offset
    application/     the projection, the sampler runner, the service the app runs; the Jetson power sampler
                     and the measurement of its CPU cost; joules per answer
    infrastructure/  recorded telemetry files; what machine this is; the Jetson's tools and sysfs files; the
                     pod power samplers read through the hub proxy; a pod's clock offset from one exchange
    fakes/           a sampler that replays recorded telemetry as live samples; stand-ins for tegrastats
                     and nvpmodel that replay recorded output
    wiring.py        build_telemetry(source, bus), used by the gateway when ATLAS_TELEMETRY_SOURCE is set
                     ("fake", "tegrastats" on the Jetson, "pods" for the GPU pods' power samplers, or both
                     joined by a comma, "tegrastats,pods")
    jetson_power.py  python -m contexts.telemetry.jetson_power probe|cpu: the Jetson sampler's first run on the box
"""
