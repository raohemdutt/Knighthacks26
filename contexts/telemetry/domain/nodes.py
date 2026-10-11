"""The nodes telemetry describes, and the check a sampler makes before it speaks for one.

The nodes are jetson, pod-a and pod-b (telemetry ruling 3). pod-a and pod-b are roles, not machines: pod-a is the
teacher's pod (hub account C on the SJC hub today) and pod-b the trainer's pod (account D). The spec's gpu-a and
gpu-b are these two pods by their old names; samples that still carry the old names (a pod sampler started with
--node gpu-a, an old event log) are read as pod-a and pod-b by `canonical`.
"""

JETSON = "jetson"
POD_A = "pod-a"  # the teacher's pod (account C)
POD_B = "pod-b"  # the trainer's pod (account D)
KNOWN_NODES = (JETSON, POD_A, POD_B)
PODS = (POD_A, POD_B)
OLD_NAMES = {"gpu-a": POD_A, "gpu-b": POD_B}  # the spec's names for the pods, before telemetry ruling 3


def canonical(node: str) -> str:
    """The node's name today: an old name (gpu-a, gpu-b) becomes its pod's; any other name is returned as it is."""
    return OLD_NAMES.get(node, node)


def check_node(node: str) -> str:
    """Return `node` if it is one Atlas knows, else raise: a typo would start a new, silent node in the log."""
    if node not in KNOWN_NODES:
        raise ValueError(f"unknown node {node!r}; expected one of {', '.join(KNOWN_NODES)}")
    return node
