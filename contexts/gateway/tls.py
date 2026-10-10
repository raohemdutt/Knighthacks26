"""HTTPS for the gateway, so a phone on the LAN may use its microphone.

Browsers only allow the microphone (and speech recognition) on a secure page: https, or localhost.
With ATLAS_HTTPS=1 the gateway serves https with a certificate it makes for itself on first start
and keeps under <data dir>/tls/. The phone shows a warning once ("not private": Advanced, proceed),
because nobody but this box signed it. ATLAS_SSL_CERT and ATLAS_SSL_KEY name a real certificate instead.
"""

import ipaddress
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path


def local_names() -> tuple[list[str], list[str]]:
    """Host names and addresses this box answers to on the LAN."""
    host = socket.gethostname()
    names = list(dict.fromkeys(["localhost", host, host.split(".")[0] + ".local"]))
    addresses = ["127.0.0.1"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # no packet is sent; this only asks which interface would be used
            addresses.append(s.getsockname()[0])
    except OSError:
        pass
    return names, list(dict.fromkeys(addresses))


def self_signed(directory: Path, days: int = 30) -> tuple[Path, Path]:
    """A certificate and key for this box, made once and reused until the box's address changes."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_file, key_file, names_file = directory / "cert.pem", directory / "key.pem", directory / "names.txt"
    names, addresses = local_names()
    wanted = ",".join(names + addresses)
    if cert_file.is_file() and key_file.is_file() and names_file.is_file() and names_file.read_text() == wanted:
        return cert_file, key_file
    directory.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Atlas box")])
    now = datetime.now(timezone.utc)
    alt = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(a)) for a in addresses]
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    if os.name == "posix":
        key_file.chmod(0o600)
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    names_file.write_text(wanted)
    return cert_file, key_file


def ssl_options(data_dir: Path) -> dict:
    """uvicorn's ssl arguments from the environment; empty when https is off."""
    cert, key = os.environ.get("ATLAS_SSL_CERT"), os.environ.get("ATLAS_SSL_KEY")
    if cert and key:
        return {"ssl_certfile": cert, "ssl_keyfile": key}
    if os.environ.get("ATLAS_HTTPS", "0") == "1":
        cert_file, key_file = self_signed(Path(data_dir) / "tls")
        return {"ssl_certfile": str(cert_file), "ssl_keyfile": str(key_file)}
    return {}
