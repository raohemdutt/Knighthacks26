"""Speak the script (video/script.yaml) and check it fits the slots.

    python video/scripts/voice.py            # synthesise what changed, write voice.json, print the fit report
    python video/scripts/voice.py --report   # the report only, from the existing files

For every line: video/public/voice/<slot>-<n>.mp3 (edge-tts, the voice named in the script; cached by text hash).
Then video/src/voice.json: per slot, each line's text, start second within the slot, duration and file, laid out
in order with `gap_s` between lines, honouring an `at:` pin when it is later than the natural start. The report
says, per slot, how many seconds of speech there are against the budget, and fails when a slot overflows its
59.8 s (the number-key layout's fixed slot) so nobody finds out at render time.
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "script.yaml"
VOICE_DIR = ROOT / "public" / "voice"
OUT = ROOT / "src" / "voice.json"
SLOT_S = 59.8
TITLE_S = 1.5  # the title card: speech starts after it


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


async def speak(text: str, voice: str, rate: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate).save(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(SCRIPT.read_text(encoding="utf-8"))
    voice, rate, gap, budget = cfg["voice"], cfg.get("rate", "+0%"), float(cfg.get("gap_s", 0.6)), float(cfg.get("budget_s", 52))
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    slots, problems = {}, []
    for key, slot in sorted(cfg["slots"].items(), key=lambda kv: int(kv[0])):
        key = int(key)
        t = TITLE_S if key else 0.0
        lines = []
        for n, line in enumerate(slot["lines"]):
            text = line["text"].strip()
            digest = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:10]
            path = VOICE_DIR / f"{key}-{n}-{digest}.mp3"
            if not path.exists():
                if args.report:
                    problems.append(f"slot {key} line {n}: not synthesised yet (run without --report)")
                    continue
                for stale in VOICE_DIR.glob(f"{key}-{n}-*.mp3"):
                    stale.unlink()
                asyncio.run(speak(text, voice, rate, path))
            d = duration(path)
            start = max(t, float(line.get("at", 0)))
            lines.append({"n": n, "text": text, "start": round(start, 2), "duration": round(d, 2),
                          "file": f"voice/{path.name}"})
            t = start + d + gap
        speech = sum(l["duration"] for l in lines)
        end = lines[-1]["start"] + lines[-1]["duration"] if lines else 0
        slots[key] = {"title": slot["title"], "lines": lines, "speech_s": round(speech, 1), "ends_at_s": round(end, 1)}
        flag = "OVER" if end > SLOT_S else ("tight" if speech > budget else "ok")
        print(f"slot {key} {slot['title']:<18} {speech:5.1f} s speech, ends {end:5.1f} s of {SLOT_S}  {flag}")
        if end > SLOT_S:
            problems.append(f"slot {key} overflows its {SLOT_S} s by {end - SLOT_S:.1f} s: cut words or move a line")
        elif speech > budget:
            problems.append(f"slot {key} is tight: {speech:.1f} s of speech against a {budget:.0f} s budget")
    OUT.write_text(json.dumps({"voice": voice, "slots": slots}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT.parent)}")
    for p in problems:
        print("  " + p)
    return 1 if any(p.startswith("slot") and "overflows" in p for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
