"""L36: the two pooled trainings on one Colab A100, one after the other, launched through the workstation's wrapper.

The session must be READY with the base model in /content/models/Qwen3.5-4B (lab/l16v2/colab_setup.sh), the branch
uploaded as a git bundle to /content/l36.bundle, and v2's dataset tarball (sha256 816847a9...e703, from the L16v2
run) at /content/dataset-tool-calling-v2.tar.gz. From the workstation (Git Bash, MSYS_NO_PATHCONV=1):

    bash $W/tooling/dispatcher/colab.sh exec -s <session> --timeout 90 -f <clone>/lab/l36/colab_job.py

The first exec clones the bundle, unpacks v2, checks its rows against the frozen sha256, pools the two sets with
lab/l36/pool.py and starts the job detached. Every later exec prints the state and the log tails. The job trains
(a) the f2r4 recipe, then (b) the v3 builder's default recipe, each at most 62 steps with validation every 8 steps,
the best checkpoint kept and early stop after 2 evaluations that did not beat the best by 3 percent (the v2
search's full-length rule). If (a) takes over 35 minutes, (b) is skipped and SKIPPED_B says so. Each run folder is
packed as /content/l36/run-<run>.tar.gz with a .sha256 beside it; DONE marks the end.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path("/content/l36")
REPO = Path("/content/l36repo")
BRANCH = "ctx/learning/v3-pooled"
V2_TAR = "/content/dataset-tool-calling-v2.tar.gz"
V2_TAR_SHA = "816847a99fc6d3c223c8e6a1bc5523ed1ee586fb24bb9ad0e6f0cedd90f6e703"
V2_ROWS_SHA = "64e3d4222e9e007e588016a43b8914e1a3f4c9e72b1152727824d8934c135b9d"
POOLED_ROWS_SHA = "1fd5a39f915b2c2877791d2b8e6915ab11bfe23f5b7b6d4f7d0c40b1abb863d3"   # the workstation's pool.py build
COMMON = ("--batch 4 --grad-accum 4 --keep-best --seed 42 --max-steps 62 --eval-steps 8 --early-stop 2 "
          "--early-stop-min-rel 0.03")
RUNS = [("l36a-f2r4-pooled", "--lr 0.000427627 --rank 4 --alpha 8 --dropout 0.1360 --warmup-ratio 0.0093"),
        ("l36b-v3default-pooled", "--lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 --warmup-ratio 0.10")]
LIMIT_A_S = 2100

ONE_RUN = """
run={run}
rm -rf {root}/data/pooled-$run && cp -r {root}/pooled {root}/data/pooled-$run
D={root}/data/pooled-$run
echo "$run start $(date -u +%FT%TZ) $(date +%s)" >> {root}/times.txt
python3 lab/train.py --cuda --no-install --dataset $D/tool-calling-v3v2-pooled --rows $D/rows-train.jsonl \
  --val-rows $D/rows-validation.jsonl --model /content/models/Qwen3.5-4B \
  --base-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out {root}/runs/$run --run-id $run \
  {common} {recipe} > {root}/runs/$run.stdout.json 2> {root}/runs/$run.log
echo $? > {root}/runs/$run.rc
echo "$run end $(date -u +%FT%TZ) $(date +%s)" >> {root}/times.txt
cp {root}/runs/$run.stdout.json {root}/runs/$run.log {root}/runs/$run.rc {root}/runs/$run/ 2>/dev/null
( cd {root}/runs && tar -czf {root}/run-$run.tar.gz $run )
( cd {root} && sha256sum run-$run.tar.gz > run-$run.tar.gz.sha256 && stat -c %s run-$run.tar.gz >> run-$run.tar.gz.sha256 )
touch {root}/runs/$run.DONE
"""

SCRIPT = f"""
set -u
cd {REPO}
export LAB_STATE_DIR={ROOT}/state LAB_TASK=L36 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
{ONE_RUN.format(run=RUNS[0][0], root=ROOT, common=COMMON, recipe=RUNS[0][1])}
s=$(grep "^{RUNS[0][0]} start" {ROOT}/times.txt | awk '{{print $4}}'); e=$(grep "^{RUNS[0][0]} end" {ROOT}/times.txt | awk '{{print $4}}')
if [ $((e - s)) -gt {LIMIT_A_S} ]; then
  echo "run a took $((e - s)) s, over {LIMIT_A_S} s: run b is skipped" > {ROOT}/runs/SKIPPED_B
else
{ONE_RUN.format(run=RUNS[1][0], root=ROOT, common=COMMON, recipe=RUNS[1][1])}
fi
touch {ROOT}/runs/DONE
"""


def sh(command: str) -> str:
    done = subprocess.run(command, shell=True, capture_output=True, text=True)
    return (done.stdout + done.stderr).strip()


if (ROOT / "runs" / "STARTED").exists():
    print(sh(f"cat {ROOT}/times.txt; ls {ROOT}/runs; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"))
    for run, _ in RUNS:
        log = ROOT / "runs" / f"{run}.log"
        if log.exists():
            print("--", run)
            print(sh(f"grep -a -i -E 'eval|valid|error|Traceback' {log} | tail -n 6; tail -n 3 {log} | cut -c1-300"))
            print(sh(f"cat {ROOT}/runs/{run}/validation.jsonl 2>/dev/null | cut -c1-200"))
else:
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    print(sh(f"sha256sum {V2_TAR}"))
    if V2_TAR_SHA not in sh(f"sha256sum {V2_TAR}"):
        raise SystemExit("v2 dataset tarball sha256 does not match")
    print(sh(f"rm -rf {REPO} && git clone -q -b {BRANCH} /content/l36.bundle {REPO} && git -C {REPO} log --format='repo %H %s' -1"))
    print(sh(f"tar -xzf {V2_TAR} -C {ROOT}/data && sha256sum {ROOT}/data/tool-calling-v2/rows-train.jsonl"))
    if V2_ROWS_SHA not in sh(f"sha256sum {ROOT}/data/tool-calling-v2/rows-train.jsonl"):
        raise SystemExit("v2 rows sha256 does not match the frozen one")
    print(sh(f"cd {REPO} && rm -rf {ROOT}/pooled && python3 lab/l36/pool.py --v2 {ROOT}/data/tool-calling-v2 "
             f"--v3 fixtures/learning/tool-calling-v3 --out {ROOT}/pooled > {ROOT}/pool.json; cat {ROOT}/pool.json | grep rows_train"))
    if POOLED_ROWS_SHA not in (ROOT / "pool.json").read_text():
        raise SystemExit("the pooled rows on the VM differ from the workstation's build")
    print(sh("pip uninstall -y torchao 2>&1 | tail -n 1"))
    (ROOT / "runs" / "STARTED").write_text("started\n")
    subprocess.Popen(["bash", "-c", SCRIPT], start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=open(ROOT / "job.out", "w"), stderr=subprocess.STDOUT)
    print("state: started", [run for run, _ in RUNS])
