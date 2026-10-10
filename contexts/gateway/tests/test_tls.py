import ssl

from contexts.gateway.tls import self_signed, ssl_options


def test_https_is_off_unless_asked_for(tmp_path, monkeypatch):
    for name in ("ATLAS_HTTPS", "ATLAS_SSL_CERT", "ATLAS_SSL_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert ssl_options(tmp_path) == {} and not (tmp_path / "tls").exists()


def test_the_box_makes_a_certificate_it_can_serve_and_reuses_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_HTTPS", "1")
    monkeypatch.delenv("ATLAS_SSL_CERT", raising=False)
    opts = ssl_options(tmp_path)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(opts["ssl_certfile"], opts["ssl_keyfile"])  # a matching pair uvicorn can load
    first = (tmp_path / "tls" / "cert.pem").read_bytes()
    assert b"BEGIN CERTIFICATE" in first and self_signed(tmp_path / "tls")[0].read_bytes() == first  # made once


def test_a_real_certificate_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_HTTPS", "1")
    monkeypatch.setenv("ATLAS_SSL_CERT", "/etc/atlas/cert.pem")
    monkeypatch.setenv("ATLAS_SSL_KEY", "/etc/atlas/key.pem")
    assert ssl_options(tmp_path) == {"ssl_certfile": "/etc/atlas/cert.pem", "ssl_keyfile": "/etc/atlas/key.pem"}
