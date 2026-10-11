#!/usr/bin/env bash
# Put the Shared brain card (G4) on the box without replacing the rest of its code, for the morning operator.
#
#     tooling/demo/deploy_shared_brain.sh                         # alexander@100.124.34.36, the workspace .env
#     tooling/demo/deploy_shared_brain.sh alexander@<box> --env /path/to/.env
#
# Run from this branch's checkout (work/G4b) in Git Bash, with node, ssh key access to the box, and the workspace
# .env that holds the teacher pod's account (AUP_HUB_USER_C, AUP_HUB_TOKEN_C, AUP_HUB_URL_C).
#
# What it does, and only this:
#   1. builds the web app here (web/out, the static export the gateway serves at :8000);
#   2. on the box: backs up contexts/gateway/app.py, bootstrap.py and web/out to ~/shared-brain-backup-<time>,
#      unpacks the new web/out and contexts/circle/brain_power.py, and adds the five wiring blocks with
#      tooling/demo/shared_brain_patch.py (anchored, idempotent; the box's gateway files differ from main, so they
#      are patched, not replaced), then byte-compiles the three files with the box's Python;
#   3. appends the three account C keys to the box's .env when they are missing (values are never printed);
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

echo "1/3 building the web app"
(cd web && npm ci --no-audit --no-fund && npm run build)

echo "2/3 copying to $target and patching the gateway (no restart)"
pack="$(mktemp -t shared-brain.XXXXXX)"
trap 'rm -f "$pack"' EXIT
tar czf "$pack" web/out contexts/circle/brain_power.py tooling/demo/shared_brain_patch.py
ssh "$target" "rm -f /tmp/shared-brain.tgz"
scp -q "$pack" "$target:/tmp/shared-brain.tgz"
ssh "$target" 'set -e
cd ~/Knighthacks26
b=~/shared-brain-backup-$(date +%Y%m%d-%H%M%S)
mkdir -p "$b/contexts/gateway"
cp contexts/gateway/app.py contexts/gateway/bootstrap.py "$b/contexts/gateway/"
mv web/out "$b/web-out"
tar xzf /tmp/shared-brain.tgz && rm -f /tmp/shared-brain.tgz
python3 tooling/demo/shared_brain_patch.py .
~/atlas/venv-yolo/bin/python -m py_compile contexts/gateway/app.py contexts/gateway/bootstrap.py contexts/circle/brain_power.py
echo "backup: $b"'

echo "3/3 the teacher pod's account in the box's .env (values not shown)"
grep -E '^(AUP_HUB_USER_C|AUP_HUB_TOKEN_C|AUP_HUB_URL_C)=' "$envfile" | ssh "$target" 'cd ~/Knighthacks26
while IFS= read -r line; do k="${line%%=*}"; grep -q "^$k=" .env || printf "%s\n" "$line" >> .env; done
echo "account C keys in the box .env: $(grep -cE "^(AUP_HUB_USER_C|AUP_HUB_TOKEN_C|AUP_HUB_URL_C)=" .env) of 3"'

cat <<EOF

Done, nothing restarted. To switch the new routes on (needs sudo on the box):
    ssh -t $target sudo systemctl restart atlas-gateway
then check, from any machine on the tailnet:
    curl -s http://${target#*@}:8000/health
    curl -s http://${target#*@}:8000/api/circle/brain/power | head -c 300
Rollback: copy the two files back from the backup folder printed above, move its web-out back to
~/Knighthacks26/web/out, and restart the gateway the same way.
EOF
