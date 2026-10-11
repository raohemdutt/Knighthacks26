"""Every test that uses the fake database (FakeNeon) runs twice: once as psycopg on a TCP connection and once
through the real HTTPS adapter with a fake endpoint behind it. The tests themselves do not change."""

import types

import pytest

from contexts.circle.tests import neon_helpers

TRANSPORTS = ["tcp", "https"]


def _names(code: types.CodeType) -> set[str]:
    """Every name a function mentions, lambdas and inner functions included."""
    found = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            found |= _names(const)
    return found


def uses_fake_neon(fn, seen: set | None = None) -> bool:
    """True when `fn`, or a helper of its own module that it calls, builds a FakeNeon."""
    seen = seen if seen is not None else set()
    if not isinstance(fn, types.FunctionType) or fn in seen:
        return False
    seen.add(fn)
    names = _names(fn.__code__)
    if fn.__globals__.get("FakeNeon") is neon_helpers.FakeNeon and "FakeNeon" in names:
        return True
    helpers = (fn.__globals__.get(n) for n in names)
    return any(uses_fake_neon(h, seen) for h in helpers if getattr(h, "__module__", None) == fn.__module__)


def pytest_generate_tests(metafunc):
    if "neon_transport" in metafunc.fixturenames:
        metafunc.parametrize("neon_transport", TRANSPORTS, indirect=True)
    elif uses_fake_neon(metafunc.function) and not getattr(metafunc.module, "NAMES_ITS_TRANSPORT", False):
        metafunc.fixturenames.append("neon_transport")
        metafunc.parametrize("neon_transport", TRANSPORTS, indirect=True)


@pytest.fixture
def neon_transport(request, monkeypatch):
    monkeypatch.setattr(neon_helpers, "TRANSPORT", request.param)
    return request.param
