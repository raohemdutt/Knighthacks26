import os

from contexts.gateway.__main__ import load_dotenv


def test_dotenv_loads_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nATLAS_T1=one\nexport ATLAS_T2="two"\nATLAS_T3=\nATLAS_T4=keep-mine\n', encoding="utf-8")
    monkeypatch.setenv("ATLAS_T4", "already-set")
    for k in ("ATLAS_T1", "ATLAS_T2", "ATLAS_T3"):
        monkeypatch.delenv(k, raising=False)
    assert load_dotenv(env) == 2
    assert (os.environ["ATLAS_T1"], os.environ["ATLAS_T2"], os.environ["ATLAS_T4"]) == ("one", "two", "already-set")
    assert "ATLAS_T3" not in os.environ
    assert load_dotenv(tmp_path / "missing.env") == 0
