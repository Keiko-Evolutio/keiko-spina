"""mTLS-Middleware für KEI-MCP API.

Diese Middleware behandelt eingehende Client-Zertifikat-Authentifizierung
für die KEI-MCP Management-API.
"""

import base64
from datetime import datetime
from typing import Any

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Optional: Neuere `cryptography` Versionen können diese Builder nicht bereitstellen
try:  # pragma: no cover - optional dependency
    from cryptography.x509.verification import PolicyBuilder, StoreBuilder  # type: ignore
except Exception:  # Fallback-Placebo Builder, NO-OP Überprüfung
    class PolicyBuilder:  # type: ignore
        def store(self, _):
            return self
        def build(self):
            class _Verifier:
                def verify(self, cert, _chain):  # noqa: ARG002
                    return []
            return _Verifier()
    class StoreBuilder:  # type: ignore
        def add_certs(self, _):
            return self
        def build(self):
            return object()

from config.mtls_config import MTLS_SETTINGS, MTLSMode
from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class ClientCertificateInfo:
    """Informationen über ein Client-Zertifikat."""

    def __init__(self, cert: x509.Certificate):
        """Initialisiert Client-Zertifikat-Informationen.

        Args:
            cert: X.509-Zertifikat
        """
        self.certificate = cert
        self.subject = cert.subject.rfc4514_string()
        self.issuer = cert.issuer.rfc4514_string()
        self.serial_number = str(cert.serial_number)
        self.not_valid_before = cert.not_valid_before
        self.not_valid_after = cert.not_valid_after
        self.fingerprint = cert.fingerprint(cert.signature_hash_algorithm).hex()

        # Subject Alternative Names extrahieren
        self.san_dns_names = []
        self.san_email_addresses = []

        try:
            san_ext = cert.extensions.get_extension_for_oid(x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            for name in san_ext.value:
                if isinstance(name, x509.DNSName):
                    self.san_dns_names.append(name.value)
                elif isinstance(name, x509.RFC822Name):
                    self.san_email_addresses.append(name.value)
        except x509.ExtensionNotFound:
            pass

    def is_valid(self) -> bool:
        """Prüft ob das Zertifikat zeitlich gültig ist."""
        now = datetime.now()
        return self.not_valid_before <= now <= self.not_valid_after

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "subject": self.subject,
            "issuer": self.issuer,
            "serial_number": self.serial_number,
            "not_valid_before": self.not_valid_before.isoformat(),
            "not_valid_after": self.not_valid_after.isoformat(),
            "fingerprint": self.fingerprint,
            "san_dns_names": self.san_dns_names,
            "san_email_addresses": self.san_email_addresses,
            "is_valid": self.is_valid()
        }


class MTLSMiddleware(BaseHTTPMiddleware):
    """FastAPI-Middleware für mTLS-Client-Authentifizierung."""

    def __init__(self, app):
        """Initialisiert die mTLS-Middleware.

        Args:
            app: FastAPI-Anwendung
        """
        super().__init__(app)
        self.client_ca_store = None

        # Client-CA laden falls konfiguriert
        if (MTLS_SETTINGS.inbound.enabled and
            MTLS_SETTINGS.inbound.verify_client_certs and
            MTLS_SETTINGS.inbound.client_ca_path):
            self._load_client_ca()

    def _load_client_ca(self):
        """Lädt die Client-CA für Zertifikat-Validierung."""
        try:
            with open(MTLS_SETTINGS.inbound.client_ca_path, "rb") as f:
                ca_data = f.read()

            # CA-Zertifikat parsen
            ca_cert = x509.load_pem_x509_certificate(ca_data, default_backend())

            # Certificate Store erstellen
            builder = StoreBuilder()
            self.client_ca_store = builder.add_certs([ca_cert]).build()

            logger.info(f"Client-CA geladen: {ca_cert.subject.rfc4514_string()}")

        except Exception as e:
            logger.exception(f"Fehler beim Laden der Client-CA: {e}")
            if MTLS_SETTINGS.inbound.mode == MTLSMode.REQUIRED:
                raise

    @trace_function("mtls.middleware.dispatch")
    async def dispatch(self, request: Request, call_next):
        """Verarbeitet eingehende Requests mit mTLS-Validierung.

        Args:
            request: Eingehender HTTP-Request
            call_next: Nächste Middleware/Handler

        Returns:
            HTTP-Response
        """
        # mTLS nur für aktivierte Inbound-Konfiguration
        if not MTLS_SETTINGS.inbound.enabled:
            return await call_next(request)

        # Client-Zertifikat extrahieren
        client_cert_info = await self._extract_client_certificate(request)

        # mTLS-Validierung
        validation_result = await self._validate_client_certificate(client_cert_info)

        # Request-State mit Zertifikat-Informationen erweitern
        request.state.mtls_enabled = True
        request.state.client_cert_info = client_cert_info
        request.state.mtls_validation_result = validation_result

        # Bei erforderlicher mTLS und fehlgeschlagener Validierung
        if (MTLS_SETTINGS.inbound.mode == MTLSMode.REQUIRED and
            not validation_result.get("valid", False)):

            error_detail = {
                "error": "Client Certificate Required",
                "message": validation_result.get("error", "Gültiges Client-Zertifikat erforderlich"),
                "type": "mtls_authentication_error"
            }

            if MTLS_SETTINGS.enable_mtls_logging:
                logger.warning(f"mTLS-Authentifizierung fehlgeschlagen: {validation_result.get('error')}")

            raise HTTPException(status_code=403, detail=error_detail)

        # Request weiterleiten
        response = await call_next(request)

        # mTLS-Informationen zu Response-Headers hinzufügen (optional)
        if client_cert_info and MTLS_SETTINGS.log_cert_details:
            response.headers["X-Client-Cert-Subject"] = client_cert_info.subject
            response.headers["X-Client-Cert-Valid"] = str(validation_result.get("valid", False))

        return response

    async def _extract_client_certificate(self, request: Request) -> ClientCertificateInfo | None:
        """Extrahiert Client-Zertifikat aus Request.

        Args:
            request: HTTP-Request

        Returns:
            Client-Zertifikat-Informationen oder None
        """
        try:
            # 1. Direkte TLS-Verbindung (wenn FastAPI TLS-Termination macht)
            if hasattr(request, "scope") and "client_cert" in request.scope:
                cert_data = request.scope["client_cert"]
                if cert_data:
                    cert = x509.load_der_x509_certificate(cert_data, default_backend())
                    return ClientCertificateInfo(cert)

            # 2. Proxy-übertragenes Zertifikat (z.B. nginx, HAProxy)
            cert_header = request.headers.get(MTLS_SETTINGS.inbound.cert_header_name)
            if cert_header:
                # Base64-dekodiertes PEM-Zertifikat
                try:
                    cert_pem = base64.b64decode(cert_header).decode("utf-8")
                    cert = x509.load_pem_x509_certificate(cert_pem.encode("utf-8"), default_backend())
                    return ClientCertificateInfo(cert)
                except Exception as e:
                    logger.warning(f"Fehler beim Parsen des Proxy-übertragenen Zertifikats: {e}")

            # 3. Standard SSL-Umgebungsvariablen (falls verfügbar)
            ssl_client_cert = request.headers.get("SSL_CLIENT_CERT")
            if ssl_client_cert:
                cert = x509.load_pem_x509_certificate(ssl_client_cert.encode("utf-8"), default_backend())
                return ClientCertificateInfo(cert)

            return None

        except Exception as e:
            logger.exception(f"Fehler beim Extrahieren des Client-Zertifikats: {e}")
            return None

    async def _validate_client_certificate(self, cert_info: ClientCertificateInfo | None) -> dict[str, Any]:
        """Validiert Client-Zertifikat.

        Args:
            cert_info: Client-Zertifikat-Informationen

        Returns:
            Validierungsergebnis
        """
        if not cert_info:
            return {
                "valid": False,
                "error": "Kein Client-Zertifikat bereitgestellt"
            }

        try:
            # 1. Zeitliche Gültigkeit prüfen
            if not cert_info.is_valid():
                return {
                    "valid": False,
                    "error": f"Zertifikat zeitlich ungültig (gültig von {cert_info.not_valid_before} bis {cert_info.not_valid_after})"
                }

            # 2. CA-Validierung (falls konfiguriert)
            if MTLS_SETTINGS.inbound.verify_client_certs and self.client_ca_store:
                try:
                    # Certificate Chain Validation
                    builder = PolicyBuilder().store(self.client_ca_store)
                    verifier = builder.build()

                    # Validierung durchführen
                    verifier.verify(cert_info.certificate, [])

                    if MTLS_SETTINGS.enable_mtls_logging:
                        logger.info(f"Client-Zertifikat erfolgreich validiert: {cert_info.subject}")

                except Exception as e:
                    return {
                        "valid": False,
                        "error": f"CA-Validierung fehlgeschlagen: {e!s}"
                    }

            # 3. Subject-Whitelist prüfen (falls konfiguriert)
            if MTLS_SETTINGS.inbound.allowed_client_subjects:
                if cert_info.subject not in MTLS_SETTINGS.inbound.allowed_client_subjects:
                    return {
                        "valid": False,
                        "error": f"Client-Subject nicht in Whitelist: {cert_info.subject}"
                    }

            # Validierung erfolgreich
            result = {
                "valid": True,
                "subject": cert_info.subject,
                "issuer": cert_info.issuer,
                "serial_number": cert_info.serial_number,
                "fingerprint": cert_info.fingerprint
            }

            if MTLS_SETTINGS.log_cert_details:
                result.update(cert_info.to_dict())

            return result

        except Exception as e:
            logger.exception(f"Fehler bei Client-Zertifikat-Validierung: {e}")
            return {
                "valid": False,
                "error": f"Validierungsfehler: {e!s}"
            }


__all__ = [
    "ClientCertificateInfo",
    "MTLSMiddleware"
]
