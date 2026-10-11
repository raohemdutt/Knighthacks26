"""python -m lab publish-adapter: publish the trained LoRA adapter and its model card to the team Hugging Face repo (L8).

    python -m lab publish-adapter --repo <owner/name> --adapter-dir <dir> --task-spec <spec.json> [--run-id R]
                                  [--private] [--fake]

What it does, in order, stopping at the first refusal and printing one JSON line:
1. reads the evaluation gate's ledger row (the newest row with the held-out metrics, or the one whose
   refs.run_id is --run-id) and refuses when a metric is missing;
2. reads the base model's card again through the hub client and refuses unless its license is apache-2.0;
3. builds the upload folder in a temporary directory: the adapter's files, README.md (the card, filled from
   lab/model_card_template.md and the row) and LICENSE (the Apache 2.0 text);
4. scans every file of that folder for an image, a caption, a label, a frame_ref, a data/ path or a camera id and
   refuses when it finds one;
5. uploads the folder through the client.

The hub client is injected (publish(client=...)); lab/hf_fake.py records what would be uploaded and every test uses
it. --fake runs the same steps against that fake and uploads nothing. Without --fake the real client is built; it
imports huggingface_hub only then, and reads the token from the environment variable HF_TOKEN, which is never
printed or logged. The command for the real run is in lab/L8_REAL_RUN.md.

The gate's row. L4 writes one ledger row per run (lab/refine/learning/RUBRIC.md, LT4 and LT5): refs.run_id and
these metrics, all read by name and none typed here:
  learning.heldout.current, .candidate, .teacher   scores as fractions of 1
  learning.heldout.n_items, .n_groups              the held-out set's size as items and as groups
  learning.heldout.noise                           run-to-run noise of the score, as a fraction of 1
  learning.heldout.p_value                         the paired p-value, candidate against current
The task spec file is a contracts.learning.TaskSpec as JSON: its task and instruction go on the card.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from string import Template
from typing import Any, Protocol

from contracts.learning import TaskSpec
from lab.machine import ledger

CLI_VERB = "publish-adapter"   # registers python -m lab publish-adapter
BASE_MODEL = "Qwen/Qwen3.5-4B"
LICENSE_ID = "apache-2.0"
TOKEN_ENV = "HF_TOKEN"          # the name the credential check lists in the workspace .env; only the name is used here
TEMPLATE = Path(__file__).resolve().with_name("model_card_template.md")
GATE_PREFIX = "learning.heldout."
GATE_METRICS = ("current", "candidate", "teacher", "n_items", "n_groups", "noise", "p_value")
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"

APACHE_LICENSE = r"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
"""


class HubClient(Protocol):
    def read_card(self, repo_id: str) -> str: ...

    def upload_folder(self, repo_id: str, folder: Path, *, private: bool = False) -> str: ...


# --- the privacy check ---------------------------------------------------------------------------------------
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif", ".svg"})
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")
# (kind, pattern) over the file's bytes read as latin-1; each is a thing that would show a person's own data
PRIVACY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, re.compile(rx, re.IGNORECASE | re.ASCII))
    for kind, rx in (
        ("image", r"!\[[^\]]*\]\([^)]*\)|data:image/|<img\b"),                 # an image embedded in text
        ("caption", r"\bcaptions?\b"),
        ("label", r"\blabell?(?:s|ed|ing)?\b"),
        ("frame_ref", r"frame_?ref"),
        ("data_path", r"(?<![A-Za-z0-9_.-])data[/\\]"),                         # data/ at the start of a path, not metadata/
        ("camera_id", r"camera[_\- ]?id\b|\bcam[_\-]?\d+\b|\bcamera[_\- ]\d+\b|/dev/video\d+"),
    )
)
_CHUNK = 8 * 1024 * 1024
_OVERLAP = 256


def _is_image_file(path: Path, head: bytes) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES or head.startswith(IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")


def privacy_findings(folder: Path) -> list[dict[str, str]]:
    """Every file under folder that is an image or whose bytes match a privacy pattern: [{"file", "kind"}], one per
    file and kind, in file order. Never returns the matched text, so a finding cannot leak what it found."""
    found: list[dict[str, str]] = []
    folder = Path(folder)
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        kinds: list[str] = []
        with path.open("rb") as fh:
            head = fh.read(16)
            if _is_image_file(path, head):
                kinds.append("image")
            fh.seek(0)
            tail = b""
            while chunk := fh.read(_CHUNK):
                text = (tail + chunk).decode("latin-1")
                for kind, rx in PRIVACY_PATTERNS:
                    if kind not in kinds and rx.search(text):
                        kinds.append(kind)
                tail = chunk[-_OVERLAP:]
        found.extend({"file": rel, "kind": kind} for kind in kinds)
    return found


# --- the card ------------------------------------------------------------------------------------------------
class Refusal(Exception):
    def __init__(self, outcome: str, error: str, **more: Any) -> None:
        super().__init__(error)
        self.result = {"ok": False, "outcome": outcome, "error": error, **more}


def gate_row(run_id: str | None, state_dir: str | os.PathLike[str] | None) -> dict[str, Any]:
    """The newest ledger row carrying every held-out metric (and refs.run_id == run_id when given)."""
    rows = [r for r in ledger.read(state_dir=state_dir, has_metric=GATE_PREFIX + "candidate")
            if run_id is None or (r.get("refs") or {}).get("run_id") == run_id]
    if not rows:
        raise Refusal("no_ledger_row", "no ledger row with learning.heldout.candidate" + (f" for run {run_id}" if run_id else ""))
    row = max(rows, key=lambda r: r["seq"])
    missing = [GATE_PREFIX + m for m in GATE_METRICS if ledger.finite_number(row["metrics"].get(GATE_PREFIX + m)) is None]
    if missing:
        raise Refusal("row_incomplete", f"ledger row {row['seq']} lacks {', '.join(missing)}", seq=row["seq"])
    return row


def _score(value: float) -> str:
    return f"{value:.3f}"


def render_card(row: dict[str, Any], spec: TaskSpec, repo: str) -> str:
    m = row["metrics"]
    g = lambda name: m[GATE_PREFIX + name]  # noqa: E731
    return Template(TEMPLATE.read_text(encoding="utf-8")).substitute(
        base=BASE_MODEL, repo_name=repo.split("/")[-1], task=spec.task,
        instruction=" ".join(spec.instruction.split()), run_id=(row.get("refs") or {}).get("run_id") or "unrecorded", seq=row["seq"],
        current=_score(g("current")), candidate=_score(g("candidate")), teacher=_score(g("teacher")),
        n_items=int(g("n_items")), n_groups=int(g("n_groups")), noise=f"{g('noise'):.4g}", p_value=f"{g('p_value'):.4g}")


def card_license(card: str) -> str | None:
    """The license key in a model card's YAML metadata, None when the card has none or cannot be parsed."""
    import yaml

    parts = card.split("---")
    if not card.lstrip().startswith("---") or len(parts) < 3:
        return None
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    value = meta.get("license") if isinstance(meta, dict) else None
    return value if isinstance(value, str) else None


def check_base_license(client: HubClient) -> None:
    found = card_license(client.read_card(BASE_MODEL))
    if found != LICENSE_ID:
        raise Refusal("base_license", f"the card of {BASE_MODEL} gives license {found!r}, not {LICENSE_ID!r}; nothing was pushed")


def stage(adapter_dir: Path, folder: Path, card: str) -> None:
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / CONFIG).is_file() or not (adapter_dir / WEIGHTS).is_file():
        raise Refusal("bad_adapter", f"{adapter_dir} needs {CONFIG} and {WEIGHTS}")
    try:
        base = json.loads((adapter_dir / CONFIG).read_text(encoding="utf-8")).get("base_model_name_or_path")
    except (ValueError, AttributeError):
        base = None
    if base != BASE_MODEL:
        raise Refusal("wrong_base", f"{CONFIG} names base {base!r}, not {BASE_MODEL!r}")
    for path in sorted(adapter_dir.iterdir()):
        if path.is_file() and path.name not in ("README.md", "LICENSE"):
            shutil.copy2(path, folder / path.name)
    (folder / "README.md").write_text(card, encoding="utf-8")
    (folder / "LICENSE").write_text(APACHE_LICENSE.strip("\n") + "\n", encoding="utf-8")


def publish(*, repo: str, adapter_dir: str | os.PathLike[str], spec_path: str | os.PathLike[str], client: HubClient,
            run_id: str | None = None, state_dir: str | os.PathLike[str] | None = None, private: bool = False) -> dict[str, Any]:
    """Run the five steps in the module docstring; returns the result dict, never raises on a refusal."""
    try:
        try:
            spec = TaskSpec.model_validate_json(Path(spec_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Refusal("bad_task_spec", f"cannot read {spec_path} as a TaskSpec: {type(exc).__name__}") from exc
        row = gate_row(run_id, state_dir)
        card = render_card(row, spec, repo)
        check_base_license(client)
        with tempfile.TemporaryDirectory(prefix="publish-adapter-") as tmp:
            folder = Path(tmp)
            stage(Path(adapter_dir), folder, card)
            findings = privacy_findings(folder)
            if findings:
                raise Refusal("privacy", f"{len(findings)} privacy finding(s); nothing was pushed", findings=findings)
            files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
            url = client.upload_folder(repo, folder, private=private)
        return {"ok": True, "outcome": "published", "repo": repo, "url": url, "files": files, "ledger_seq": row["seq"],
                "base_license": LICENSE_ID}
    except Refusal as refusal:
        return refusal.result


# --- the real client -----------------------------------------------------------------------------------------
class RealHubClient:
    """The Hub through huggingface_hub, imported here and nowhere else. The token is held, never printed."""

    def __init__(self, token: str) -> None:
        import huggingface_hub

        self._hub = huggingface_hub
        self._token = token

    def read_card(self, repo_id: str) -> str:
        path = self._hub.hf_hub_download(repo_id, "README.md", token=self._token)
        return Path(path).read_text(encoding="utf-8")

    def upload_folder(self, repo_id: str, folder: Path, *, private: bool = False) -> str:
        api = self._hub.HfApi(token=self._token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(folder), commit_message="Publish LoRA adapter and model card")
        return f"https://huggingface.co/{repo_id}"


def main(argv: list[str]) -> dict[str, Any]:
    """Publish the trained LoRA adapter and its model card to the team Hugging Face repo."""
    ap = argparse.ArgumentParser(prog="python -m lab publish-adapter", description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="the Hub repo to publish to, owner/name")
    ap.add_argument("--adapter-dir", required=True, type=Path, help="folder with adapter_config.json and adapter_model.safetensors")
    ap.add_argument("--task-spec", required=True, type=Path, help="the task's TaskSpec as JSON; its task and instruction go on the card")
    ap.add_argument("--run-id", help="the gate run whose ledger row to use (default: the newest gate row)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--fake", action="store_true", help="run every step against the recording fake and upload nothing")
    args = ap.parse_args(argv)
    if args.fake:
        from lab.hf_fake import FakeHubClient

        client: HubClient = FakeHubClient()
    else:
        token = os.environ.get(TOKEN_ENV, "").strip()
        if not token:
            return {"ok": False, "outcome": "missing_token", "error": f"environment variable {TOKEN_ENV} is not set"}
        client = RealHubClient(token)
    result = publish(repo=args.repo, adapter_dir=args.adapter_dir, spec_path=args.task_spec, client=client,
                     run_id=args.run_id, private=args.private)
    if args.fake:
        result["fake"] = True
    return result
