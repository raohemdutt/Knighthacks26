"""The group brain's way to the model: the teacher's vLLM server on the pod in the pod-a role, through the hub proxy.

    link = TeacherLink.from_env()           # the account LAB_POD_ROLES puts in the pod-a role (C on SJC today)
    link.model()                            # "group-brain" when the server serves that name, else "teacher"
    link.chat([ChatMessage(role="user", content="...")])   # a ChatResult: text only, thinking off, priority -10

It adds no client of its own: the call goes through lab/proxy.py (the box's one client for a pod's model server: the
hub token, the browser User-Agent, streaming under the proxy's 125 second cut, the proxy's errors mapped to kinds) and
the stock runtime client over it (lab.proxy.proxied_runtime builds runtime.llama.LlamaServerRuntime).

What every request carries, over what the runtime writes:
- model: "group-brain". lab/pod_a_teacher.sh serves the teacher under two names, "teacher" and "group-brain"; a server
  started before that answers only to "teacher". model() asks GET /v1/models once (no member text in it) and keeps
  "group-brain" when it is there, else "teacher", so the link works before and after the teacher's next restart; a 404
  for the model forgets the choice so the next question asks again.
- priority: PLANNING_PRIORITY (-10). With --scheduling-policy priority vLLM serves the lowest value first and, when the
  KV cache runs out, preempts the highest; lab/label.py sends none, which is 0, so a planning question goes ahead of a
  labeling run and continuous batching keeps the labels decoding beside it. At the pinned vLLM commit (b1388b1) a
  server on the default fcfs policy accepts the field and ignores it, so the same request works on either.
- chat_template_kwargs {"enable_thinking": false}: Qwen3 answers without a thinking block.
- max_tokens: MAX_TOKENS (256), about 90 s at the Strix Halo teacher's 2.9 tokens a second. Streamed underneath, so
  the 125 s cut never applies.
- text only: the transport refuses, before a byte is sent, any request that is not plain text.

Only the pod in the pod-a role is ever asked: an account in another role is refused (PodRoleError).
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from contracts.types import ChatMessage, ChatResult, ModelVersion
from lab import hub as lab_hub
from lab.proxy import NotFound, PodLink, proxied_runtime

GROUP_BRAIN = "group-brain"
TEACHER = "teacher"
TEACHER_PORT = 8000
TEACHER_ROLE = "pod-a"
TEACHER_BASE = "Qwen/Qwen3.8-27B"
PLANNING_PRIORITY = -10
MAX_TOKENS = 256
TIMEOUT_S = 130.0  # the longest silence between two pieces of a streamed answer; the proxy's own read timeout


class NoTeacherModel(Exception):
    """The server on the teacher's port serves neither name."""

    kind = "no_model"


def teacher_letter(roles: Mapping[str, str]) -> str:
    """The one account letter LAB_POD_ROLES puts in the teacher's pod-a role (PodRoleError for none or several)."""
    letters = sorted(letter for letter, role in roles.items() if role == TEACHER_ROLE)
    if len(letters) != 1:
        named = ", ".join(f"{letter}:{role}" for letter, role in sorted(roles.items())) or "none"
        raise lab_hub.PodRoleError(f"the group brain needs exactly one account in the {TEACHER_ROLE} role; "
                                   f"{lab_hub.ENV_POD_ROLES} names {named}")
    return letters[0]


class TeacherLink:
    """The teacher's model server behind one PodLink (see the module docstring). Implements the application's
    BrainLink: node, host, priority, max_tokens, model(), chat()."""

    def __init__(self, link: PodLink, *, port: int = TEACHER_PORT, priority: int | None = PLANNING_PRIORITY,
                 max_tokens: int = MAX_TOKENS, node: str = TEACHER_ROLE, timeout_s: float = TIMEOUT_S) -> None:
        self.link = link
        self.port = port
        self.priority = priority
        self.max_tokens = max_tokens
        self.node = node
        self.host = urllib.parse.urlsplit(link.base_url).hostname or link.base_url
        self.timeout_s = timeout_s
        self._model: str | None = None
        self._runtime: Any = None

    @classmethod
    def from_env(cls, *, letter: str | None = None, environ: Mapping[str, str] | None = None,
                 env_files: Sequence[Path] | None = None, transport: httpx.BaseTransport | None = None,
                 **kwargs: Any) -> "TeacherLink":
        """The link to the teacher's pod: the account in the pod-a role of LAB_POD_ROLES (or `letter`, which must fill
        that role), its keys and hub from the environment and the .env files as lab/hub.py reads them."""
        roles = lab_hub.pod_roles(environ, env_files)
        letter = (letter or teacher_letter(roles)).upper()
        if roles.get(letter) != TEACHER_ROLE:
            raise lab_hub.PodRoleError(f"account {letter} fills {roles.get(letter) or 'no role'}, not the teacher's "
                                       f"{TEACHER_ROLE}; the group brain asks only the teacher's pod")
        account = lab_hub.load_accounts([letter], environ=environ, env_files=env_files)[letter]
        return cls(PodLink.for_account(account, transport=transport, environ=environ), **kwargs)

    def __repr__(self) -> str:
        return f"TeacherLink(node={self.node!r}, host={self.host!r}, port={self.port})"

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.http.close()
        self.link.close()

    def extra(self) -> dict[str, Any]:
        """The fields every chat request carries over what the runtime writes."""
        fields: dict[str, Any] = {"max_tokens": self.max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        if self.priority is not None:
            fields["priority"] = self.priority
        return fields

    def model(self) -> str:
        """The served name to ask: "group-brain" when the server has it, else "teacher" (asked once, then kept)."""
        if self._model is None:
            names = self.link.server(self.port).models()
            if GROUP_BRAIN in names:
                self._model = GROUP_BRAIN
            elif TEACHER in names:
                self._model = TEACHER
            else:
                raise NoTeacherModel(f"the server on port {self.port} serves {', '.join(names) or 'nothing'}, "
                                     f"neither {GROUP_BRAIN!r} nor {TEACHER!r}")
        return self._model

    def chat(self, messages: list[ChatMessage]) -> ChatResult:
        """One answer, whole. Raises lab.proxy's errors (each with its kind) or ValueError for a request that is not
        plain text, which then never left."""
        model = self.model()
        runtime = self._runtime
        if runtime is None or runtime.model.version != model:
            if runtime is not None:
                runtime.http.close()
            runtime = self._runtime = proxied_runtime(
                ModelVersion(version=model, base=TEACHER_BASE), self.link, self.port, model_name=model,
                extra=self.extra(), text_only=True, timeout_s=self.timeout_s)
        try:
            return runtime.chat(messages)
        except NotFound:
            self._model = None  # the server changed under us (restarted with other names): ask again next time
            raise
