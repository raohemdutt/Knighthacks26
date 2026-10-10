"""Train the kit's anomaly scorer and save its weights.

    python -m contexts.spot.train                       # reads ATLAS_SPOT_KIT or fixtures/spot/kit, writes fixtures/spot/anomaly_scorer.json
    python -m contexts.spot.train --kit <folder> --out <file.json> [--check]

Deterministic: the same kit always gives the same file. `--check` trains and compares with the file on disk instead
of writing (exit 1 when they differ). Nothing here reads validation_results.json. See contexts/spot/domain/scorer.py.
"""

import argparse
import json
import sys
from pathlib import Path

from contexts.spot.domain.scorer import train
from contexts.spot.infrastructure.kit import kit_path, load_kit

DEFAULT_OUT = "fixtures/spot/anomaly_scorer.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    kit = load_kit(args.kit)
    if kit is None:
        print(f"no SPOT kit at {kit_path(args.kit)}", file=sys.stderr)
        return 2
    model = train(kit)
    text = json.dumps(model, indent=1) + "\n"
    out = Path(args.out)
    if args.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == text
        print("weights match a retrain" if same else f"{out} differs from a retrain")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}: {len(model['features'])} features, threshold {model['threshold']}")
    print(model["verdict"])
    for f, w in sorted(zip(model["features"], model["weights"]), key=lambda t: -abs(t[1])):
        print(f"  {w:+.3f}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
