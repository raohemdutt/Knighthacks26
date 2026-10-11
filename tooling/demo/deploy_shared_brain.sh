#!/usr/bin/env bash
# Put the Shared brain card (G4, and G6's two-pod brain and answer cost) on the box without replacing the rest of its
# code, for the morning operator.
#
#     tooling/demo/deploy_shared_brain.sh                         # alexander@100.124.34.36, the workspace .env
#     tooling/demo/deploy_shared_brain.sh alexander@<box> --env /path/to/.env
#
# Run from this branch's checkout in Git Bash, with node, ssh key access to the box, and the workspace .env that holds
# the pods' accounts: AUP_HUB_USER_C, AUP_HUB_TOKEN_C, AUP_HUB_URL_C (the head, required) and, for the two-pod card,
# AUP_HUB_USER_D, AUP_HUB_TOKEN_D, AUP_HUB_URL_D and LAB_POD_ROLES (C:pod-a,D:pod-b).
#
# What it does, and only this:
#   1. builds the web app here (web/out, the static export the gateway serves at :8000);
#   2. on the box: backs up contexts/gateway/app.py, bootstrap.py, contexts/learning/lab_deploy.py and web/out to
#      ~/shared-brain-backup-<time>, unpacks the new web/out, contexts/circle/brain_power.py and brain_pair.py,
#      contexts/learning/answer_energy.py and lab_deploy.py, and adds the wiring blocks with
#      tooling/demo/shared_brain_patch.py (anchored, idempotent; the box's gateway files differ from main, so they
#      are patched, not replaced), then byte-compiles the files and imports the gateway with the box's Python. If the
#      import fails, every file is put back from the backup and the script stops;
#   3. appends the pods' keys to the box's .env when they are missing (values are never printed);
#   4. does NOT restart anything. The new screen is served at once; the new routes need the gateway restarted,
#      which needs sudo and is the operator's step (printed at the end).
set -euo pipefail

target="alexander@100.124.34.36"
envfile=""
while [ $# -gt 0 ]; do
    case "$1" in
        --env) envfile="$2"; shift 2 ;;
        *) target="$1"; shift ;;
    esac
done
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
if [ -z "$envfile" ]; then
    for f in "$repo/.env" "$repo/../.env" "$repo/../../.env"; do [ -f "$f" ] && envfile="$f" && break; done
fi
[ -n "$envfile" ] && [ -f "$envfile" ] || { echo "no .env found: pass --env <file>" >&2; exit 2; }
n="$(grep -cE '^(AUP_HUB_USER_C|AUP_HUB_TOKEN_C|AUP_HUB_URL_C)=' "$envfile" || true)"
[ "$n" = "3" ] || { echo "$envfile has $n of the 3 account C keys" >&2; exit 2; }
keys='^(AUP_HUB_USER_C|AUP_HUB_TOKEN_C|AUP_HUB_URL_C|AUP_HUB_USER_D|AUP_HUB_TOKEN_D|AUP_HUB_URL_D|LAB_POD_ROLES)='
m="$(grep -cE "$keys" "$envfile" || true)"
[ "$m" = "7" ] || echo "note: $envfile has $m of the 7 keys; without pod D's three and LAB_POD_ROLES the card shows pod D offline" >&2

echo "1/3 building the web app"
(cd web && npm ci --no-audit --no-fund && npm run build)

echo "2/3 copying to $target and patching the gateway (no restart)"
pack="$(mktemp -t shared-brain.XXXXXX)"
trap 'rm -f "$pack"' EXIT
tar czf "$pack" web/out contexts/circle/brain_power.py contexts/circle/brain_pair.py \
    contexts/learning/answer_energy.py contexts/learning/lab_deploy.py tooling/demo/shared_brain_patch.py
ssh "$target" "rm -f /tmp/shared-brain.tgz"
scp -q "$pack" "$target:/tmp/shared-brain.tgz"
ssh "$target" 'set -e
cd ~/Knighthacks26
b=~/shared-brain-backup-$(date +%Y%m%d-%H%M%S)
mkdir -p "$b/contexts/gateway" "$b/contexts/learning"
cp contexts/gateway/app.py contexts/gateway/bootstrap.py "$b/contexts/gateway/"
[ -f contexts/learning/lab_deploy.py ] && cp contexts/learning/lab_deploy.py "$b/contexts/learning/"
mv web/out "$b/web-out"
tar xzf /tmp/shared-brain.tgz && rm -f /tmp/shared-brain.tgz
python3 tooling/demo/shared_brain_patch.py .
py=~/atlas/venv-yolo/bin/python
if ! $py -m py_compile contexts/gateway/app.py contexts/gateway/bootstrap.py contexts/circle/brain_power.py \
        contexts/circle/brain_pair.py contexts/learning/answer_energy.py contexts/learning/lab_deploy.py \
   || ! $py -c "import contexts.gateway.app" ; then
    echo "the patched gateway does not import: putting every file back from $b" >&2
    cp "$b"/contexts/gateway/app.py "$b"/contexts/gateway/bootstrap.py contexts/gateway/
    if [ -f "$b/contexts/learning/lab_deploy.py" ]; then cp "$b/contexts/learning/lab_deploy.py" contexts/learning/;
    else rm -f contexts/learning/lab_deploy.py; fi
    rm -f contexts/circle/brain_pair.py contexts/learning/answer_energy.py
    rm -rf web/out && mv "$b/web-out" web/out
    exit 1
fi
echo "backup: $b"'

echo "3/3 the pods' accounts in the box's .env (values not shown)"
grep -E "$keys" "$envfile" | ssh "$target" 'cd ~/Knighthacks26
while IFS= read -r line; do k="${line%%=*}"; grep -q "^$k=" .env || printf "%s\n" "$line" >> .env; done
echo "pod keys in the box .env: $(grep -cE "^(AUP_HUB_USER_C|AUP_HUB_TOKEN_C|AUP_HUB_URL_C|AUP_HUB_USER_D|AUP_HUB_TOKEN_D|AUP_HUB_URL_D|LAB_POD_ROLES)=" .env) of 7"
grep -E "^LAB_POD_ROLES=" .env | grep -qv "^LAB_POD_ROLES=C:pod-a,D:pod-b$" && echo "note: the box .env sets LAB_POD_ROLES to something other than C:pod-a,D:pod-b; left as it is" || true'

cat <<EOF

Done, nothing restarted. To switch the new routes on (needs sudo on the box):
    ssh -t $target sudo systemctl restart atlas-gateway
then check, from any machine on the tailnet:
    curl -s http://${target#*@}:8000/health
    curl -s http://${target#*@}:8000/api/circle/brain/power | head -c 300
    curl -s http://${target#*@}:8000/api/circle/shared-brain | head -c 600
Rollback: copy the files back from the backup folder printed above (contexts/gateway/app.py, bootstrap.py,
contexts/learning/lab_deploy.py), move its web-out back to ~/Knighthacks26/web/out, and restart the gateway the same way.
EOF
