"""L33: train tool-calling-v3 on a Colab A100 with Ilan1's v2 recipe, launched through the workstation's Colab wrapper.

The session must be READY with the base model in /content/models/Qwen3.5-4B (lab/l16v2/colab_setup.sh) and the
branch uploaded as a git bundle to /content/l33.bundle. From the workstation, in the clone (Git Bash):

    git bundle create ../L33-pack/l33.bundle ctx/learning/tool-calling-v3
    bash $W/tooling/dispatcher/colab.sh upload -s <session> "$(cygpath -m ../L33-pack/l33.bundle)" /content/l33.bundle
    bash $W/tooling/dispatcher/colab.sh exec -s <session> --timeout 90 -f "$(cygpath -m lab/l33/colab_train.py)"

The first exec clones the bundle, takes torchao out (PEFT refuses Colab's 0.10.0, as in lab/l16v2/colab_job.sh) and
starts lab/train.py --cuda detached. Every later exec prints the run's state and the last lines of its log. When
the trainer exits, the run folder is packed as /content/l33/run-l33-v3-recipe.tar.gz with its .sha256, then DONE is
written. Pull it with the wrapper's download.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path("/content/l33")
REPO = Path("/content/l33repo")
RUN = "l33-v3-recipe"
OUT = ROOT / "runs" / RUN
DS = "fixtures/learning/tool-calling-v3"
RECIPE = ("--batch 4 --grad-accum 4 --keep-best --seed 42 --max-steps 62 --eval-steps 8 --lr 1e-4 --rank 16 "
          "--alpha 32 --dropout 0.05 --warmup-ratio 0.10")
SCRIPT = f"""
set -u
cd {REPO}
export LAB_STATE_DIR={ROOT}/state LAB_TASK=L33 PYTHONIOENCODING=utf-8
python3 lab/train.py --cuda --no-install --dataset {DS} --rows {DS}/rows-train.jsonl --val-rows {DS}/rows-validation.jsonl \\
  --model /content/models/Qwen3.5-4B --base-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out {OUT} --run-id {RUN} \\
  {RECIPE} > {ROOT}/runs/{RUN}.stdout.json 2> {ROOT}/runs/{RUN}.log
echo $? > {ROOT}/runs/{RUN}.rc
cp {ROOT}/runs/{RUN}.stdout.json {ROOT}/runs/{RUN}.log {OUT}/ 2>/dev/null
( cd {ROOT}/runs && tar -czf {ROOT}/run-{RUN}.tar.gz {RUN} )
( cd {ROOT} && sha256sum run-{RUN}.tar.gz > run-{RUN}.tar.gz.sha256 )
touch {ROOT}/runs/DONE
"""

if (ROOT / "runs" / "STARTED").exists():
    print("state:", "done rc=" + (ROOT / "runs" / f"{RUN}.rc").read_text().strip() if (ROOT / "runs" / "DONE").exists()
          else "running")
    log = ROOT / "runs" / f"{RUN}.log"
    if log.exists():
        print("".join(log.read_text(errors="replace").splitlines(keepends=True)[-12:]))
else:
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    subprocess.run(["rm", "-rf", str(REPO)], check=True)
    subprocess.run(["git", "clone", "-q", "-b", "ctx/learning/tool-calling-v3", "/content/l33.bundle", str(REPO)], check=True)
    print(subprocess.run(["git", "-C", str(REPO), "log", "--format=repo %H %s", "-1"], capture_output=True, text=True).stdout)
    print(subprocess.run("pip uninstall -y torchao 2>&1 | tail -n 1", shell=True, capture_output=True, text=True).stdout)
    (ROOT / "runs" / "STARTED").write_text("started\n")
    subprocess.Popen(["bash", "-c", SCRIPT], start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
    print("state: started", RUN)
