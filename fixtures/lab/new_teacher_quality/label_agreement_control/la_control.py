"""R64 control arm of the label-agreement check: the same 30 examples, the same code path (check_agreement), the
serving v0.31.0 teacher, but the instruction NOT reworded (prefix ""), so the only change from the stored labels is
the build (the seed is 1 because the check refuses 0; at temperature 0 the seed does not enter greedy decoding).

    la_control.py DATASET OUT_DIR
"""
import json
import sys
from pathlib import Path

from lab import hub, label, label_agreement as la

la.RewordedTeacher.__init__.__defaults__ = ("",)
teacher = label.Teacher.from_hub(hub.load_accounts(["C"])["C"], port=8000, stream=True)
with teacher:
    report = la.check_agreement(sys.argv[1], teacher, sample="first", size=30, second_seed=1, first_seed=0,
                                run_id=None, workers=1)
report["second_pass"]["rewording"] = "none (R64 control arm: prefix removed)"
paths = la.write_report(report, Path(sys.argv[2]))
print(json.dumps({"run_id": report["run_id"], "verdict": report["verdict"], "overall": report["overall"],
                  "fields": report.get("fields"), "report": {k: str(v) for k, v in paths.items()}}))
