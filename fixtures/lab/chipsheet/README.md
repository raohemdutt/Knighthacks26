# Chip sheet fixtures

Tool output in the shape `lab/kernels/chipsheet/` parses, for tests and `chipsheet.py self-check`. Every file here is
synthetic: the layout follows ROCm 7.13's rocminfo, hipcc, amd-smi and LLVM `-Rpass-analysis=kernel-resource-usage`
output and the probe's own JSON lines, but the numbers are invented and are not measurements of any chip. The amd-smi
power reading the self-check uses is the real one recorded on pod C,
`fixtures/telemetry/pod_power/strixhalo_amdsmi_metric_power.txt`. Replace a file here with the pod's own output once
pod D has run the sheet, and keep its name ending in `_recorded` instead of `_synthetic`.
