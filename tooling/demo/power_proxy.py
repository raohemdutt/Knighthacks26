"""Local helper for the live power page (tooling/demo/power.html): serves the page and forwards its calls.

    python tooling/demo/power_proxy.py                  # then open http://127.0.0.1:8787/
    python tooling/demo/power_proxy.py --port 8787 --env ../.env --account C

Why it exists: the pod's power sampler (lab/pod_power.py, port 8790 on the pod) is only reachable through the hub
proxy with `Authorization: token <hub token>`, and a page in a browser cannot send that header to another origin.
The helper reads the token from the workspace .env and adds it server side; the page never sees it.

Routes (all on 127.0.0.1 only):
    GET  /                 the page
    GET  /config           the box's address (from the `jetson` key of .env) and the idle figures the page draws
    GET  /pod/status       the pod sampler's /status, through the hub proxy
    GET  /pod/samples?after=N   the pod sampler's /samples, plus local_sent_ms / local_recv_ms so the page can map
                           the pod's clock onto this machine's
    POST /box/ask          {"text": ...} -> the box gateway's POST /api/ask (the same path the phone and voice use)
    POST /teacher/ask      {"text": ...} -> the teacher's /v1/chat/completions on the pod (streamed, thinking off),
                           answered with the window in this machine's epoch ms so the page can integrate pod watts

Stdlib only. Never prints a secret: errors name the status and the route, not the token.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
IDLE = {"pod_w": 10.27, "jetson_w": 15.76}  # config/telemetry_idle.yaml (T10 pod C, T13 Jetson VDD_IN)


def read_env(paths):
    found = {}
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            found.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return found


def env_candidates(explicit):
    if explicit:
        return [Path(explicit)]
    out, d = [], HERE
    for _ in range(6):
        out.append(d / ".env")
        d = d.parent
    return out


class Cfg:
    pass


CFG = Cfg()


def http(method, url, headers=None, body=None, timeout=30):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:2000]
    except Exception as e:  # no connection, timeout
        return 599, json.dumps({"error": type(e).__name__}).encode()


def pod_url(port, path):
    return f"{CFG.hub}/user/{CFG.user}/proxy/{port}{path}"


def pod_headers(extra=None):
    h = {"Authorization": f"token {CFG.token}", "User-Agent": UA}
    h.update(extra or {})
    return h


def teacher_ask(text, max_tokens):
    body = json.dumps({
        "model": "teacher", "stream": True, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": text}],
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(pod_url(8000, "/v1/chat/completions"), data=body, method="POST",
                                 headers=pod_headers({"Content-Type": "application/json",
                                                      "Accept": "text/event-stream"}))
    start = time.time() * 1000
    pieces, tokens = [], 0
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                for ch in chunk.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        pieces.append(piece)
                        tokens += 1
    except urllib.error.HTTPError as e:
        return 502, {"error": f"teacher answered {e.code}"}
    except Exception as e:
        return 502, {"error": f"teacher unreachable: {type(e).__name__}"}
    end = time.time() * 1000
    return 200, {"answer": "".join(pieces), "start_ms": start, "end_ms": end, "chunks": tokens}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send(self, status, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/power.html"):
            return self.send(200, (HERE / "power.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/config":
            return self.send(200, {"box_http": CFG.box_http, "box_ws": CFG.box_ws, "idle": IDLE,
                                   "pod_label": CFG.pod_label})
        if u.path == "/pod/status":
            st, data = http("GET", pod_url(8790, "/status"), pod_headers(), timeout=10)
            return self.send(st, data)
        if u.path == "/pod/samples":
            after = parse_qs(u.query).get("after", ["0"])[0]
            sent = time.time() * 1000
            st, data = http("GET", pod_url(8790, f"/samples?after={int(after)}"), pod_headers(), timeout=10)
            recv = time.time() * 1000
            if st != 200:
                return self.send(st, {"error": f"pod sampler answered {st}"})
            try:
                d = json.loads(data)
            except ValueError:
                return self.send(502, {"error": "pod sampler sent no JSON"})
            # Only what the page draws: ts and socket_w per sample, so a 1-second poll stays small.
            out = [{"seq": s["seq"], "ts": s["event"]["ts"], "w": s["event"].get("socket_w"),
                    "mhz": s["event"].get("gfx_mhz"), "busy": s["event"].get("gfx_activity_pct")}
                   for s in d.get("samples", [])]
            return self.send(200, {"last_seq": d.get("last_seq"), "pod_sent": d.get("pod_sent"),
                                   "pod_received": d.get("pod_received"), "local_sent_ms": sent,
                                   "local_recv_ms": recv, "samples": out})
        return self.send(404, {"error": "no such route"})

    def do_POST(self):
        u = urlparse(self.path)
        b = self.body_json()
        text = (b.get("text") or "").strip()
        if not text:
            return self.send(400, {"error": "text is empty"})
        if u.path == "/box/ask":
            h = {"Content-Type": "application/json"}
            if CFG.box_token:
                h["Authorization"] = f"Bearer {CFG.box_token}"
            st, data = http("POST", f"{CFG.box_http}/api/ask", h, json.dumps({"text": text}).encode(), timeout=150)
            return self.send(st, data)
        if u.path == "/teacher/ask":
            st, payload = teacher_ask(text, int(b.get("max_tokens") or 96))
            return self.send(st, payload)
        return self.send(404, {"error": "no such route"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--env", help="the .env to read (default: the nearest .env above this file)")
    ap.add_argument("--account", default="C", help="hub account letter of the pod whose sampler is drawn (default C)")
    ap.add_argument("--box", help="the box's address (default: the `jetson` key of .env)")
    a = ap.parse_args()
    env = dict(read_env(env_candidates(a.env)))
    env.update({k: v for k, v in os.environ.items() if k.startswith(("AUP_HUB_", "ATLAS_TOKEN"))})
    L = a.account.upper()
    missing = [k for k in (f"AUP_HUB_USER_{L}", f"AUP_HUB_TOKEN_{L}") if not env.get(k)]
    if missing:
        sys.exit(f"missing keys in .env: {', '.join(missing)}")
    CFG.user, CFG.token = env[f"AUP_HUB_USER_{L}"], env[f"AUP_HUB_TOKEN_{L}"]
    CFG.hub = (env.get(f"AUP_HUB_URL_{L}") or env.get("LAB_HUB_URL") or "https://sjc.aupcloud.io").rstrip("/")
    box = a.box or env.get("jetson")
    if not box:
        sys.exit("no box address: pass --box or set the `jetson` key in .env")
    CFG.box_http, CFG.box_ws = f"http://{box}:8000", f"ws://{box}:8000/ws"
    CFG.box_token = env.get("ATLAS_TOKEN")
    CFG.pod_label = f"pod {L}"
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"power page: http://127.0.0.1:{a.port}/   (pod {L} sampler through the hub proxy, box gateway on :8000)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
