"""The nodes telemetry describes, and the check a sampler makes before it speaks for one."""

JETSON = "jetson"
GPU_A = "gpu-a"  # the teacher pod
GPU_B = "gpu-b"  # the trainer pod
KNOWN_NODES = (JETSON, GPU_A, GPU_B)


def check_node(node: str) -> str:
    """Return `node` if it is one Atlas knows, else raise: a typo would start a new, silent node in the log."""
    if node not in KNOWN_NODES:
        raise ValueError(f"unknown node {node!r}; expected one of {', '.join(KNOWN_NODES)}")
    return node
