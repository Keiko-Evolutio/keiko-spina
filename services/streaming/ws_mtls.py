"""WS-mTLS Hilfsfunktionen für KEI-Stream.

Diese Einheit stellt Funktionen bereit, um Client-Zertifikate aus
WebSocket-Handshake-Headern zu extrahieren und optional gegen eine
konfigurierte Client-CA zu validieren. Sie ist bewusst leichtgewichtig,
da klassische HTTP-Middlewares bei WebSockets nicht greifen.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cryptography import x509
from cryptography.hazmat.backends import default_backend

# Optional: Neuere `cryptography` Versionen können diese Builder nicht bereitstellen
try:  # pragma: no cover - optional dependency
    from cryptography.x509.verification import PolicyBuilder, StoreBuilder  # type: ignore
except Exception:  # pragma: no cover
    class PolicyBuilder:  # type: ignore
        """Fallback-PolicyBuilder ohne echte Überprüfung."""

        def store(self, _):
            return self

        def build(self):
            class _Verifier:
                """Placebo-Verifier, der keine Validierung durchführt."""

                def verify(self, cert, _chain):  # noqa: ARG002
                    return []

            return _Verifier()

    class StoreBuilder:  # type: ignore
        """Fallback-StoreBuilder ohne echte CA-Kette."""

        def add_certs(self, _):
            return self

        def build(self):
            return object()

from config.mtls_config import MTLS_SETTINGS
from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)


class WSClientCertificateInfo:
    """Informationen über ein aus WS-Headern extrahiertes Client-Zertifikat.

    Attributes:
        certificate: Geparstes X.509 Zertifikat
        subject: Formatierter Subject-String (RFC4514)
        issuer: Formatierter Issuer-String (RFC4514)
        serial_number: Seriennummer als String
        not_valid_before: Beginn der Gültigkeit
        not_valid_after: Ende der Gültigkeit
        fingerprint: Zertifikats-Fingerprint (Hash des Signaturalgorithmus)
    """

    def __init__(self, cert: x509.Certificate) -> None:
        """Initialisiert die Zertifikatsinformationen.

        Args:
            cert: X.509-Zertifikat
        """
        self.certificate: x509.Certificate = cert
        self.subject: str = cert.subject.rfc4514_string()
        self.issuer: str = cert.issuer.rfc4514_string()
        self.serial_number: str = str(cert.serial_number)
        self.not_valid_before: datetime = cert.not_valid_before
        self.not_valid_after: datetime = cert.not_valid_after
        try:
            self.fingerprint: str = cert.fingerprint(cert.signature_hash_algorithm).hex()
        except Exception:  # pragma: no cover - defensiv
            self.fingerprint = ""

    def is_temporally_valid(self) -> bool:
        """Prüft zeitliche Gültigkeit."""
        now = datetime.now()
        return self.not_valid_before <= now <= self.not_valid_after


def _parse_pem_from_header_value(header_value: str) -> x509.Certificate | None:
    """Parst ein PEM-kodiertes Zertifikat aus einem Header-Wert.

    Unterstützt Base64-kodierte PEM-Blöcke (übliches Forwarding-Format
    von Proxies) sowie direkten PEM-Inhalt.

    Args:
        header_value: Header-Inhalt (Base64-PEM oder PEM)

    Returns:
        Geparstes Zertifikat oder None bei Fehler
    """
    if not header_value:
        return None
    try:
        # Versuch: Base64 → PEM
        try:
            decoded = base64.b64decode(header_value, validate=True).decode("utf-8")
        except Exception:
            decoded = header_value
        pem_bytes = decoded.encode("utf-8")
        return x509.load_pem_x509_certificate(pem_bytes, default_backend())
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"WS-mTLS: PEM Parsing fehlgeschlagen: {exc}")
        return None


def extract_client_certificate_from_ws_headers(headers: Mapping[str, str]) -> WSClientCertificateInfo | None:
    """Extrahiert Client-Zertifikat aus WS-Handshake-Headern.

    Sucht zuerst den konfigurierten Forwarding-Header, danach gängige
    Fallbacks wie ``SSL_CLIENT_CERT`` oder ``X-SSL-Client-Cert``.

    Args:
        headers: Header-Mapping der WS-Verbindung

    Returns:
        Zertifikatsinformationen oder None, wenn nicht vorhanden/parsbar
    """
    # 1) Konfigurierter Header (z. B. "X-Client-Cert")
    header_name = MTLS_SETTINGS.inbound.cert_header_name
    header_val = headers.get(header_name) or headers.get(header_name.lower())
    # 2) Gängige Fallbacks
    header_val = (
        header_val
        or headers.get("SSL_CLIENT_CERT")
        or headers.get("ssl_client_cert")
        or headers.get("X-SSL-Client-Cert")
        or headers.get("x-ssl-client-cert")
    )

    cert = _parse_pem_from_header_value(header_val) if header_val else None
    if not cert:
        return None
    return WSClientCertificateInfo(cert)


def validate_client_certificate(cert_info: WSClientCertificateInfo | None) -> dict[str, Any]:
    """Validiert ein (optional) extrahiertes Client-Zertifikat.

    - Prüft zeitliche Gültigkeit
    - Optional: Validiert gegen konfigurierte Client-CA
    - Optional: Prüft Subject-Whitelist

    Args:
        cert_info: Zertifikatsinformationen oder None

    Returns:
        Validierungsresultat mit ``valid``-Flag und optionalen Details
    """
    if cert_info is None:
        return {"valid": False, "error": "Kein Client-Zertifikat bereitgestellt"}

    # Zeitliche Gültigkeit
    if not cert_info.is_temporally_valid():
        return {
            "valid": False,
            "error": (
                f"Zertifikat zeitlich ungültig (gültig von "
                f"{cert_info.not_valid_before} bis {cert_info.not_valid_after})"
            ),
        }

    # Optional: CA-Validierung
    if MTLS_SETTINGS.inbound.verify_client_certs and MTLS_SETTINGS.inbound.client_ca_path:
        try:
            with open(MTLS_SETTINGS.inbound.client_ca_path, "rb") as f:
                ca_data = f.read()
            ca_cert = x509.load_pem_x509_certificate(ca_data, default_backend())
            store = StoreBuilder().add_certs([ca_cert]).build()
            verifier = PolicyBuilder().store(store).build()
            _chain = verifier.verify(cert_info.certificate, [])
        except Exception as exc:  # pragma: no cover - defensiv
            return {"valid": False, "error": f"CA-Validierung fehlgeschlagen: {exc}"}

    # Optional: Subject-Whitelist
    allowed = MTLS_SETTINGS.inbound.allowed_client_subjects or []
    if allowed and cert_info.subject not in allowed:
        return {"valid": False, "error": f"Client-Subject nicht in Whitelist: {cert_info.subject}"}

    # Erfolg
    return {
        "valid": True,
        "subject": cert_info.subject,
        "issuer": cert_info.issuer,
        "serial_number": cert_info.serial_number,
        "fingerprint": cert_info.fingerprint,
    }


__all__ = [
    "WSClientCertificateInfo",
    "extract_client_certificate_from_ws_headers",
    "validate_client_certificate",
]
