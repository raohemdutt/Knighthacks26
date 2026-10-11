#!/usr/bin/env bash
# Copy this working tree to the Jetson and restart the app (the box has no GitHub login, so code reaches it this way).
#
#     ./jetson/push.sh                          # alexander@ubuntu.local, ~/Knighthacks26
#     ./jetson/push.sh alexander@172.20.10.6  # a specific address
#     ./jetson/push.sh --web                    # also build the phone app first (needs node)
#     ./jetson/push.sh --zones                  # also overwrite the box's config/zones.yaml with this tree's
#
# Not copied: .git, .env, data/, the detector engine, model files, virtualenvs. The event log stays, so the app
# rebuilds its memory from it when it restarts. Run from anywhere in the repo; needs ssh access by key.
set -euo pipefail

build_web=0
push_zones=0
target="alexander@ubuntu.local"
for arg in "$@"; do
    case "$arg" in
        --web) build_web=1 ;;
        --zones) push_zones=1 ;;
        *) target="$arg" ;;
    esac
done
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
dest="Knighthacks26"

if [ "$build_web" -eq 1 ]; then
    (cd web && npm install --no-audit --no-fund && npm run build)
fi

# Zones drawn on site (the Zones page) live only in the box's config/zones.yaml. A push must not quietly undo them
# minutes before a demo, so a differing file on the box is kept unless --zones says otherwise.
zones_exclude=""
if [ "$push_zones" -eq 0 ]; then
    box_sum="$(ssh "$target" "cksum < ~/$dest/config/zones.yaml 2>/dev/null | cut -d' ' -f1" || true)"
    tree_sum="$(cksum < config/zones.yaml | cut -d' ' -f1)"
    if [ -n "$box_sum" ] && [ "$box_sum" != "$tree_sum" ]; then
        echo "note: the box's config/zones.yaml differs from this tree's (set on site?): keeping the box's. Use --zones to overwrite it."
        zones_exclude="--exclude=config/zones.yaml"
    fi
fi

pack="$(mktemp -t atlas-push.XXXXXX)"
trap 'rm -f "$pack"' EXIT
tar czf "$pack" $zones_exclude --exclude=.git --exclude=.venv --exclude=node_modules --exclude=__pycache__ --exclude='*.pyc' \
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
