#!/bin/bash
# One run of lab/train.py --cuda on a Colab worker, detached, for the search that is driven from the workstation
# (lab/l16v2/bo_local.py): a short trial or a full-length run. It is started at most once per name, writes its files
# under /content/l16v2/runs/<name>/, and when the trainer ends (or is interrupted for pruning) packs them as a
# product with its .sha256 and then writes runs/<name>/DONE, so the workstation can pull and verify it.
#   usage: colab_trial.sh <name> <adapter: yes|no> <trainer flags ...>
set -u
NAME="${1:?name}"; WITH_ADAPTER="${2:?yes|no}"; shift 2
OUT=/content/l16v2; MODEL=/content/models/Qwen3.5-4B; REPO=/content/Knighthacks26; DS=$OUT/data/tool-calling-v2
RUN=$OUT/runs/$NAME
mkdir -p "$RUN" $OUT/products
[ -e "$RUN/STARTED" ] && { echo "already started"; exit 0; }
date -u +%FT%TZ > "$RUN/STARTED"
cd $REPO
export LAB_STATE_DIR=$OUT/state LAB_TASK=L16 LAB_ACTOR="Ilan1 agent" LAB_TIER=opus PYTHONIOENCODING=utf-8
python3 lab/train.py --cuda --no-install --dataset $DS --rows $DS/rows-train.jsonl --val-rows $DS/rows-validation.jsonl \
  --model $MODEL --base-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out "$RUN" --run-id "$NAME" \
  --batch 4 --grad-accum 4 --keep-best --seed 42 "$@" > "$RUN/stdout.json" 2> "$RUN/train.log"
RC=$?
echo "$RC" > "$RUN/RC"
MEMBERS="$NAME/stdout.json $NAME/train.log $NAME/RC $NAME/STARTED"
for f in run.json telemetry.jsonl validation.jsonl; do [ -e "$RUN/$f" ] && MEMBERS="$MEMBERS $NAME/$f"; done
if [ "$WITH_ADAPTER" = yes ] && [ -d "$RUN/adapter" ]; then MEMBERS="$MEMBERS $NAME/adapter"
elif [ -e "$RUN/adapter/adapter_config.json" ]; then MEMBERS="$MEMBERS $NAME/adapter/adapter_config.json"; fi
( cd $OUT/runs && tar -czf $OUT/products/run-$NAME.tar.gz $MEMBERS )
( cd $OUT/products && echo "$(sha256sum run-$NAME.tar.gz | cut -d' ' -f1) $(stat -c %s run-$NAME.tar.gz) run-$NAME.tar.gz" > run-$NAME.tar.gz.sha256 )
echo "$RC" > "$RUN/DONE"
