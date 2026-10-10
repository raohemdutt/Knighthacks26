#!/usr/bin/env bash
# Copy this working tree to the Jetson and restart the app (the box has no GitHub login, so code reaches it this way).
#
#     ./jetson/push.sh                          # alexander@ubuntu.local, ~/Knighthacks26
#     ./jetson/push.sh alexander@192.168.1.126  # a specific address
#     ./jetson/push.sh --web                    # also build the phone app first (needs node)
#
# Not copied: .git, .env, data/, the detector engine, model files, virtualenvs. The event log stays, so the app
# rebuilds its memory from it when it restarts. Run from anywhere in the repo; needs ssh access by key.
set -euo pipefail

build_web=0
target="alexander@ubuntu.local"
for arg in "$@"; do
    case "$arg" in
        --web) build_web=1 ;;
        *) target="$arg" ;;
    esac
done
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
dest="Knighthacks26"

if [ "$build_web" -eq 1 ]; then
    (cd web && npm install --no-audit --no-fund && npm run build)
fi

pack="$(mktemp -t atlas-push.XXXXXX)"
trap 'rm -f "$pack"' EXIT
tar czf "$pack" --exclude=.git --exclude=.venv --exclude=node_modules --exclude=__pycache__ --exclude='*.pyc' \
    --exclude=.env --exclude=data --exclude='*.engine' --exclude='*.gguf' --exclude=web/.next --exclude=web/node_modules .
scp -q "$pack" "$target:/tmp/atlas-push.tgz"
ssh "$target" "mkdir -p ~/$dest && cd ~/$dest && tar xzf /tmp/atlas-push.tgz && rm /tmp/atlas-push.tgz && sudo systemctl restart atlas-gateway"

echo -n "waiting for the app"
for _ in $(seq 1 40); do
    if ssh "$target" "curl -sf -m 3 localhost:8000/health >/dev/null"; then
        echo; echo "up: $(ssh "$target" 'curl -s localhost:8000/health')"
        exit 0
    fi
    echo -n "."; sleep 3
done
echo; echo "the app did not answer within two minutes: ssh $target journalctl -u atlas-gateway -n 50" >&2
exit 1
