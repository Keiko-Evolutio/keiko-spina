# backend/security/mtls_manager.py
"""mTLS-Manager für Keiko Personal Assistant

Implementiert mutual TLS (mTLS) für hochsichere Umgebungen mit
Certificate-Management, -Rotation und Client-Certificate-Validierung.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import ssl
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.x509.oid import ExtensionOID, NameOID

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class CertificateStatus(str, Enum):
    """Status eines Zertifikats."""
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"
    ERROR = "error"


class CertificateType(str, Enum):
    """Typ eines Zertifikats."""
    CLIENT = "client"
    SERVER = "server"
    CA = "ca"
    INTERMEDIATE = "intermediate"


@dataclass
class CertificateInfo:
    """Informationen über ein Zertifikat."""
    certificate: x509.Certificate
    private_key: Any | None = None
    certificate_type: CertificateType = CertificateType.CLIENT
    status: CertificateStatus = CertificateStatus.VALID

    # Metadaten
    subject: str = ""
    issuer: str = ""
    serial_number: str = ""
    fingerprint: str = ""
    not_before: datetime | None = None
    not_after: datetime | None = None

    # Extensions
    subject_alt_names: list[str] = field(default_factory=list)
    key_usage: set[str] = field(default_factory=set)
    extended_key_usage: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Extrahiert Zertifikat-Informationen."""
        if self.certificate:
            self._extract_certificate_info()

    def _extract_certificate_info(self) -> None:
        """Extrahiert Informationen aus dem Zertifikat."""
        cert = self.certificate

        # Basis-Informationen
        self.subject = cert.subject.rfc4514_string()
        self.issuer = cert.issuer.rfc4514_string()
        self.serial_number = str(cert.serial_number)
        self.not_before = cert.not_valid_before.replace(tzinfo=UTC)
        self.not_after = cert.not_valid_after.replace(tzinfo=UTC)

        # Fingerprint
        fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).digest()
        self.fingerprint = base64.b64encode(fingerprint).decode("ascii")

        # Subject Alternative Names
        try:
            san_ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            self.subject_alt_names = [name.value for name in san_ext.value]
        except x509.ExtensionNotFound:
            pass

        # Key Usage
        try:
            key_usage_ext = cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE)
            ku = key_usage_ext.value

            if hasattr(ku, "digital_signature") and ku.digital_signature:
                self.key_usage.add("digital_signature")
            if hasattr(ku, "key_encipherment") and ku.key_encipherment:
                self.key_usage.add("key_encipherment")
            if hasattr(ku, "key_agreement") and ku.key_agreement:
                self.key_usage.add("key_agreement")
            if hasattr(ku, "key_cert_sign") and ku.key_cert_sign:
                self.key_usage.add("key_cert_sign")
            if hasattr(ku, "crl_sign") and ku.crl_sign:
                self.key_usage.add("crl_sign")
        except x509.ExtensionNotFound:
            pass

        # Extended Key Usage
        try:
            eku_ext = cert.extensions.get_extension_for_oid(ExtensionOID.EXTENDED_KEY_USAGE)
            for usage in eku_ext.value:
                if usage == x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH:
                    self.extended_key_usage.add("client_auth")
                elif usage == x509.oid.ExtendedKeyUsageOID.SERVER_AUTH:
                    self.extended_key_usage.add("server_auth")
        except x509.ExtensionNotFound:
            pass

    @property
    def is_expired(self) -> bool:
        """Prüft, ob Zertifikat abgelaufen ist."""
        if not self.not_after:
            return False
        return datetime.now(UTC) >= self.not_after

    def expires_soon(self, days: int = 30) -> bool:
        """Prüft, ob Zertifikat bald abläuft."""
        if not self.not_after:
            return False
        threshold = datetime.now(UTC) + timedelta(days=days)
        return threshold >= self.not_after

    @property
    def common_name(self) -> str | None:
        """Gibt Common Name zurück."""
        try:
            cn_attributes = self.certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            if cn_attributes:
                return cn_attributes[0].value
        except Exception:
            pass
        return None

    def get_agent_id(self) -> str | None:
        """Extrahiert Agent-ID aus Zertifikat."""
        # Versuche Agent-ID aus Common Name zu extrahieren
        cn = self.common_name
        if cn and cn.startswith("agent:"):
            return cn[6:]

        # Versuche Agent-ID aus Subject Alternative Names
        for san in self.subject_alt_names:
            if san.startswith("agent:"):
                return san[6:]

        return None


@dataclass
class MTLSConfig:
    """Konfiguration für mTLS."""
    enabled: bool = False

    # Certificate Paths
    ca_cert_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None
    server_cert_path: str | None = None
    server_key_path: str | None = None

    # Validation Settings
    verify_client_cert: bool = True
    verify_server_cert: bool = True
    check_hostname: bool = True

    # Certificate Rotation
    auto_rotation_enabled: bool = True
    rotation_threshold_days: int = 30
    rotation_check_interval_hours: int = 24

    # Revocation Checking
    crl_check_enabled: bool = False
    ocsp_check_enabled: bool = False

    def validate(self) -> bool:
        """Validiert mTLS-Konfiguration."""
        if not self.enabled:
            return True

        # CA-Zertifikat ist erforderlich
        if not self.ca_cert_path or not Path(self.ca_cert_path).exists():
            return False

        # Client-Zertifikat für Client-Authentifizierung
        if self.verify_client_cert:
            if not self.client_cert_path or not Path(self.client_cert_path).exists():
                return False
            if not self.client_key_path or not Path(self.client_key_path).exists():
                return False

        return True


class CertificateStore:
    """Store für Zertifikat-Management."""

    def __init__(self) -> None:
        """Initialisiert Certificate Store."""
        self._certificates: dict[str, CertificateInfo] = {}
        self._ca_certificates: dict[str, CertificateInfo] = {}
        self._revoked_certificates: set[str] = set()

    def add_certificate(self, cert_id: str, cert_info: CertificateInfo) -> None:
        """Fügt Zertifikat zum Store hinzu."""
        self._certificates[cert_id] = cert_info

        if cert_info.certificate_type == CertificateType.CA:
            self._ca_certificates[cert_id] = cert_info

        logger.info(f"Zertifikat {cert_id} zum Store hinzugefügt")

    def get_certificate(self, cert_id: str) -> CertificateInfo | None:
        """Gibt Zertifikat zurück."""
        return self._certificates.get(cert_id)

    def get_certificate_by_fingerprint(self, fingerprint: str) -> CertificateInfo | None:
        """Gibt Zertifikat anhand Fingerprint zurück."""
        for cert_info in self._certificates.values():
            if cert_info.fingerprint == fingerprint:
                return cert_info
        return None

    def get_certificates_by_subject(self, subject: str) -> list[CertificateInfo]:
        """Gibt Zertifikate anhand Subject zurück."""
        return [cert for cert in self._certificates.values() if cert.subject == subject]

    def get_expiring_certificates(self, days: int = 30) -> list[CertificateInfo]:
        """Gibt bald ablaufende Zertifikate zurück."""
        return [cert for cert in self._certificates.values() if cert.expires_soon(days)]

    def revoke_certificate(self, cert_id: str) -> bool:
        """Widerruft Zertifikat."""
        cert_info = self._certificates.get(cert_id)
        if cert_info:
            cert_info.status = CertificateStatus.REVOKED
            self._revoked_certificates.add(cert_info.serial_number)
            logger.info(f"Zertifikat {cert_id} widerrufen")
            return True
        return False

    def is_revoked(self, cert_info: CertificateInfo) -> bool:
        """Prüft, ob Zertifikat widerrufen ist."""
        return cert_info.serial_number in self._revoked_certificates

    def get_ca_certificates(self) -> list[CertificateInfo]:
        """Gibt alle CA-Zertifikate zurück."""
        return list(self._ca_certificates.values())


class MTLSManager:
    """Manager für mTLS-Operationen."""

    def __init__(self, config: MTLSConfig) -> None:
        """Initialisiert mTLS Manager.

        Args:
            config: mTLS-Konfiguration
        """
        self.config = config
        self.certificate_store = CertificateStore()
        self._rotation_task: asyncio.Task | None = None
        self._ssl_context_cache: dict[str, ssl.SSLContext] = {}

        if not config.validate():
            raise ValueError("Ungültige mTLS-Konfiguration")

    async def initialize(self) -> None:
        """Initialisiert mTLS Manager."""
        if not self.config.enabled:
            logger.info("mTLS ist deaktiviert")
            return

        logger.info("Initialisiere mTLS Manager")

        try:
            # Lade Zertifikate
            await self._load_certificates()

            # Starte automatische Rotation
            if self.config.auto_rotation_enabled:
                await self._start_certificate_rotation()

            logger.info("mTLS Manager erfolgreich initialisiert")

        except Exception as e:
            logger.exception(f"mTLS Manager Initialisierung fehlgeschlagen: {e}")
            raise

    async def shutdown(self) -> None:
        """Fährt mTLS Manager herunter."""
        if self._rotation_task:
            self._rotation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._rotation_task

        logger.info("mTLS Manager heruntergefahren")

    async def _load_certificates(self) -> None:
        """Lädt Zertifikate aus Dateien."""
        # CA-Zertifikat laden
        if self.config.ca_cert_path:
            ca_cert = await self._load_certificate_from_file(self.config.ca_cert_path)
            if ca_cert:
                ca_info = CertificateInfo(
                    certificate=ca_cert,
                    certificate_type=CertificateType.CA
                )
                self.certificate_store.add_certificate("ca", ca_info)

        # Client-Zertifikat laden
        if self.config.client_cert_path and self.config.client_key_path:
            client_cert = await self._load_certificate_from_file(self.config.client_cert_path)
            client_key = await self._load_private_key_from_file(self.config.client_key_path)

            if client_cert and client_key:
                client_info = CertificateInfo(
                    certificate=client_cert,
                    private_key=client_key,
                    certificate_type=CertificateType.CLIENT
                )
                self.certificate_store.add_certificate("client", client_info)

        # Server-Zertifikat laden
        if self.config.server_cert_path and self.config.server_key_path:
            server_cert = await self._load_certificate_from_file(self.config.server_cert_path)
            server_key = await self._load_private_key_from_file(self.config.server_key_path)

            if server_cert and server_key:
                server_info = CertificateInfo(
                    certificate=server_cert,
                    private_key=server_key,
                    certificate_type=CertificateType.SERVER
                )
                self.certificate_store.add_certificate("server", server_info)

    async def _load_certificate_from_file(self, cert_path: str) -> x509.Certificate | None:
        """Lädt Zertifikat aus Datei."""
        try:
            with open(cert_path, "rb") as f:
                cert_data = f.read()

            # Versuche PEM-Format
            try:
                return x509.load_pem_x509_certificate(cert_data)
            except ValueError:
                # Versuche DER-Format
                return x509.load_der_x509_certificate(cert_data)

        except Exception as e:
            logger.exception(f"Fehler beim Laden des Zertifikats {cert_path}: {e}")
            return None

    async def _load_private_key_from_file(self, key_path: str) -> Any | None:
        """Lädt Private Key aus Datei."""
        try:
            with open(key_path, "rb") as f:
                key_data = f.read()

            # Versuche PEM-Format
            try:
                return serialization.load_pem_private_key(key_data, password=None)
            except ValueError:
                # Versuche DER-Format
                return serialization.load_der_private_key(key_data, password=None)

        except Exception as e:
            logger.exception(f"Fehler beim Laden des Private Keys {key_path}: {e}")
            return None

    @trace_function("mtls.validate_client_certificate")
    def validate_client_certificate(self, cert_data: bytes) -> tuple[bool, CertificateInfo | None]:
        """Validiert Client-Zertifikat.

        Args:
            cert_data: Zertifikat-Daten

        Returns:
            Tuple (is_valid, certificate_info)
        """
        try:
            # Zertifikat parsen
            certificate = x509.load_der_x509_certificate(cert_data)
            cert_info = CertificateInfo(certificate=certificate)

            # Basis-Validierungen
            if cert_info.is_expired:
                logger.warning(f"Client-Zertifikat abgelaufen: {cert_info.subject}")
                return False, cert_info

            # Prüfe Widerruf
            if self.certificate_store.is_revoked(cert_info):
                logger.warning(f"Client-Zertifikat widerrufen: {cert_info.subject}")
                return False, cert_info

            # Prüfe Client-Auth Extended Key Usage
            if "client_auth" not in cert_info.extended_key_usage:
                logger.warning(f"Client-Zertifikat ohne Client-Auth EKU: {cert_info.subject}")
                return False, cert_info

            # Prüfe Zertifikatskette (vereinfacht)
            if not self._verify_certificate_chain(certificate):
                logger.warning(f"Client-Zertifikat Ketten-Validierung fehlgeschlagen: {cert_info.subject}")
                return False, cert_info

            logger.info(f"Client-Zertifikat erfolgreich validiert: {cert_info.subject}")
            return True, cert_info

        except Exception as e:
            logger.exception(f"Client-Zertifikat Validierung fehlgeschlagen: {e}")
            return False, None

    def _verify_certificate_chain(self, certificate: x509.Certificate) -> bool:
        """Verifiziert Zertifikatskette (vereinfachte Implementierung)."""
        try:
            # Hole CA-Zertifikate
            ca_certs = self.certificate_store.get_ca_certificates()

            for ca_cert_info in ca_certs:
                ca_cert = ca_cert_info.certificate

                # Prüfe, ob Zertifikat von dieser CA signiert wurde
                try:
                    ca_public_key = ca_cert.public_key()
                    ca_public_key.verify(
                        certificate.signature,
                        certificate.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        certificate.signature_hash_algorithm
                    )
                    return True
                except Exception:
                    continue

            return False

        except Exception as e:
            logger.exception(f"Zertifikatsketten-Verifikation fehlgeschlagen: {e}")
            return False

    def create_ssl_context(self, context_type: str = "client") -> ssl.SSLContext:
        """Erstellt SSL-Context für mTLS.

        Args:
            context_type: "client" oder "server"

        Returns:
            SSL-Context
        """
        if context_type in self._ssl_context_cache:
            return self._ssl_context_cache[context_type]

        if context_type == "client":
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            # Client-Zertifikat laden
            client_cert = self.certificate_store.get_certificate("client")
            if client_cert and client_cert.private_key:
                context.load_cert_chain(
                    self.config.client_cert_path,
                    self.config.client_key_path
                )

            # CA-Zertifikat für Server-Verifikation
            if self.config.ca_cert_path:
                context.load_verify_locations(self.config.ca_cert_path)

            context.check_hostname = self.config.check_hostname
            context.verify_mode = ssl.CERT_REQUIRED if self.config.verify_server_cert else ssl.CERT_NONE

        else:  # server
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

            # Server-Zertifikat laden
            server_cert = self.certificate_store.get_certificate("server")
            if server_cert and server_cert.private_key:
                context.load_cert_chain(
                    self.config.server_cert_path,
                    self.config.server_key_path
                )

            # CA-Zertifikat für Client-Verifikation
            if self.config.ca_cert_path:
                context.load_verify_locations(self.config.ca_cert_path)

            context.verify_mode = ssl.CERT_REQUIRED if self.config.verify_client_cert else ssl.CERT_NONE

        self._ssl_context_cache[context_type] = context
        return context

    async def _start_certificate_rotation(self) -> None:
        """Startet automatische Zertifikat-Rotation."""
        self._rotation_task = asyncio.create_task(self._certificate_rotation_loop())

    async def _certificate_rotation_loop(self) -> None:
        """Background-Loop für Zertifikat-Rotation."""
        while True:
            try:
                await asyncio.sleep(self.config.rotation_check_interval_hours * 3600)
                await self._check_certificate_expiration()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Zertifikat-Rotation-Loop Fehler: {e}")
                await asyncio.sleep(3600)  # 1 Stunde warten bei Fehler

    async def _check_certificate_expiration(self) -> None:
        """Prüft Zertifikat-Ablauf und initiiert Rotation."""
        expiring_certs = self.certificate_store.get_expiring_certificates(
            self.config.rotation_threshold_days
        )

        for cert_info in expiring_certs:
            logger.warning(
                f"Zertifikat läuft bald ab: {cert_info.subject} "
                f"(Ablauf: {cert_info.not_after})"
            )

            # Hier würde automatische Rotation implementiert werden
            # Für Demo-Zwecke nur Logging

    def get_certificate_info(self, cert_id: str) -> CertificateInfo | None:
        """Gibt Zertifikat-Informationen zurück."""
        return self.certificate_store.get_certificate(cert_id)

    def get_all_certificates(self) -> dict[str, CertificateInfo]:
        """Gibt alle Zertifikate zurück."""
        return self.certificate_store._certificates.copy()

    def get_mtls_stats(self) -> dict[str, Any]:
        """Gibt mTLS-Statistiken zurück."""
        all_certs = self.certificate_store._certificates

        valid_certs = sum(1 for cert in all_certs.values() if not cert.is_expired)
        expired_certs = len(all_certs) - valid_certs
        expiring_soon = len(self.certificate_store.get_expiring_certificates(30))
        revoked_certs = len(self.certificate_store._revoked_certificates)

        return {
            "mtls_enabled": self.config.enabled,
            "total_certificates": len(all_certs),
            "valid_certificates": valid_certs,
            "expired_certificates": expired_certs,
            "expiring_soon": expiring_soon,
            "revoked_certificates": revoked_certs,
            "ca_certificates": len(self.certificate_store._ca_certificates),
            "auto_rotation_enabled": self.config.auto_rotation_enabled
        }
