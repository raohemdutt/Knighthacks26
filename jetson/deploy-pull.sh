#!/usr/bin/env bash
# Deploy the newest `prod` bundle to this Jetson (run by the atlas-deploy timer, or by hand on the box).
#
#     jetson/deploy-pull.sh              # deploy if the jetson-prod release has a bundle this box is not running
#     jetson/deploy-pull.sh --dry-run    # download and list the files a deploy would change; change nothing
#     jetson/deploy-pull.sh --force      # deploy even if this sha is already deployed or failed before
#     touch ~/.atlas-deploy-paused       # freeze deploys (demo time); rm it to resume
#
# A push to the prod branch builds the bundle on GitHub (.github/workflows/jetson-prod.yml: tests, phone app, tarball)
# and publishes it with a manifest.json to the `jetson-prod` release. This script pulls: the box needs no GitHub login
# while the repository is public. Like jetson/push.sh it never touches .env, data/, the engine or the models, keeps the
# box's config/zones.yaml, and restarts atlas-gateway (and atlas-voice when it runs). New pip packages and changed
# systemd units are not installed: that is still `sudo jetson/jetson_setup.sh`.
#
# Before it overwrites anything it saves the box's copy of every file the bundle replaces. If the gateway does not
# answer /health within two minutes, those files are put back, the gateway is restarted, and the sha is marked failed
# so the timer does not retry it every minute (--force does). Unreachable GitHub or no release yet: nothing happens.
set -euo pipefail

main() {
    local dry=0 force=0 arg
    for arg in "$@"; do
        case "$arg" in
            --dry-run) dry=1 ;;
            --force) force=1 ;;
            *) echo "unknown argument: $arg" >&2; exit 2 ;;
        esac
    done
    local slug="${ATLAS_DEPLOY_REPO:-raohemdutt/Knighthacks26}"
    local base="https://github.com/$slug/releases/download/jetson-prod"
    local app="${ATLAS_DEPLOY_DIR:-$HOME/Knighthacks26}"
    local state="${ATLAS_DEPLOY_STATE:-$HOME/atlas/deploy}"
    local paused="$HOME/.atlas-deploy-paused"
    mkdir -p "$state/bundles"

    exec 9>"$state/lock"
    flock -n 9 || { echo "another deploy is running"; exit 0; }
    if [ -e "$paused" ] && [ "$dry" -eq 0 ]; then
        echo "deploys are paused ($paused)"; exit 0
    fi

    local manifest
    if ! manifest="$(curl -fsSL -m 20 "$base/manifest.json" 2>/dev/null)"; then
        note "$state" unreachable "" "could not read the jetson-prod manifest (offline, firewall, or no release yet)"
        exit 0
    fi
    local sha bundle sum
    read -r sha bundle sum < <(python3 -c '
import json, re, sys
m = json.loads(sys.argv[1])
sha, bundle, sum_ = m["sha"], m["bundle"], m["sha256"]
assert re.fullmatch(r"[0-9a-f]{40}", sha) and re.fullmatch(r"atlas-[0-9a-f]{12}\.tgz", bundle) and re.fullmatch(r"[0-9a-f]{64}", sum_)
print(sha, bundle, sum_)' "$manifest") || { note "$state" bad-manifest "" "manifest.json is not in the expected shape"; exit 1; }

    local current failed
    current="$(cat "$state/deployed" 2>/dev/null || true)"
    failed="$(cat "$state/failed" 2>/dev/null || true)"
    if [ "$dry" -eq 0 ] && [ "$force" -eq 0 ]; then
        [ "$sha" = "$current" ] && exit 0
        [ "$sha" = "$failed" ] && exit 0  # failed its health check before: wait for a new push, or --force
    fi

    local file="$state/bundles/$bundle"
    if [ ! -s "$file" ]; then
        echo "downloading $bundle"
        curl -fsSL -m 600 -o "$file.part" "$base/$bundle" || { rm -f "$file.part"; note "$state" unreachable "$sha" "bundle download failed"; exit 0; }
        mv "$file.part" "$file"
    fi
    if [ "$(sha256sum "$file" | cut -d' ' -f1)" != "$sum" ]; then
        rm -f "$file"; note "$state" bad-bundle "$sha" "sha256 does not match the manifest"; exit 1
    fi

    # The box's config/zones.yaml is drawn on site (the Zones page): never replaced from here.
    local keep=()
    [ -e "$app/config/zones.yaml" ] && keep=(--exclude=config/zones.yaml)

    local stage
    stage="$(mktemp -d)"
    trap "rm -rf '$stage'" EXIT
    tar xzf "$file" -C "$stage" "${keep[@]}"
    local changed="$state/changed.txt"
    (cd "$stage" && find . -type f -printf '%P\n' | sort | while IFS= read -r f; do
        cmp -s "$f" "$app/$f" || echo "$f"
    done) > "$changed"
    echo "$(wc -l < "$changed") files differ from $sha (list: $changed)"
    if [ "$dry" -eq 1 ]; then
        cat "$changed"; exit 0
    fi

    # Save the box's copy of each file about to be replaced, so a failed deploy can put it back.
    local backup="$state/before-${sha:0:12}.tgz"
    (cd "$app" && while IFS= read -r f; do [ -e "$f" ] && printf '%s\n' "$f"; done < "$changed") > "$state/replaced.txt"
    tar czf "$backup" -C "$app" -T "$state/replaced.txt"
    mkdir -p "$app"
    (cd "$stage" && tar cf - -T "$changed") | tar xf - -C "$app"

    echo "deployed files of $sha; restarting atlas-gateway"
    sudo -n systemctl restart atlas-gateway
    if healthy; then
        systemctl is-active --quiet atlas-voice && sudo -n systemctl restart atlas-voice
        echo "$sha" > "$state/deployed"
        rm -f "$state/failed"
        { ls -1t "$state"/before-*.tgz 2>/dev/null | tail -n +4 | xargs -r rm -f; } || true
        { ls -1t "$state"/bundles/atlas-*.tgz 2>/dev/null | tail -n +4 | xargs -r rm -f; } || true
        note "$state" deployed "$sha" "$(curl -s -m 3 localhost:8000/health)"
        exit 0
    fi

    echo "the gateway did not answer /health within two minutes: putting the previous files back" >&2
    tar xzf "$backup" -C "$app"
    sudo -n systemctl restart atlas-gateway
    echo "$sha" > "$state/failed"
    if healthy; then
        note "$state" rolled-back "$sha" "health check failed; the previous files are back and the gateway answers"
    else
        note "$state" rolled-back "$sha" "health check failed; previous files restored but the gateway still does not answer"
    fi
    exit 1
}

healthy() {
    local _
    for _ in $(seq 1 40); do
        curl -sf -m 3 localhost:8000/health >/dev/null && return 0
        sleep 3
    done
    return 1
}

note() {  # state result sha detail: one line for the journal, and last.json for anyone checking
    local state="$1" result="$2" sha="$3" detail="$4"
    echo "deploy: $result ${sha:0:12} $detail"
    python3 -c 'import json, sys, datetime
print(json.dumps({"result": sys.argv[1], "sha": sys.argv[2], "detail": sys.argv[3],
                  "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}))' \
        "$result" "$sha" "$detail" > "$state/last.json"
}

main "$@"
