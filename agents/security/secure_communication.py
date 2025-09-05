# backend/kei_agents/security/secure_communication.py
"""Secure Communication zwischen KEI-Agents.

Implementiert End-to-End-Verschlüsselung:
- Message Encryption und Authentication
- Certificate Management und PKI
- Secure Channel Establishment
- Message Integrity und Verification
"""

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from kei_logging import get_logger

from .encryption import AsymmetricEncryption, EncryptionConfig, EncryptionManager
from .exceptions import (
    CertificateError,
    ChannelEstablishmentError,
    CommunicationSecurityError,
    KeyNotFoundError,
    MessageIntegrityError,
)

logger = get_logger(__name__)


@dataclass
class SecureMessage:
    """Secure Message Data Structure."""
    message_id: str
    sender_id: str
    recipient_id: str
    encrypted_content: str
    signature: str
    timestamp: datetime
    message_type: str = "data"
    metadata: dict[str, Any] | None = None


@dataclass
class Certificate:
    """Certificate Data Structure."""
    certificate_id: str
    subject: str
    issuer: str
    public_key: bytes
    private_key: bytes | None
    valid_from: datetime
    valid_until: datetime
    certificate_pem: bytes
    is_ca: bool = False


@dataclass
class SecureChannel:
    """Secure Channel Data Structure."""
    channel_id: str
    agent_a: str
    agent_b: str
    shared_secret: bytes
    established_at: datetime
    expires_at: datetime | None
    is_active: bool = True


class MessageEncryption:
    """Message Encryption und Authentication."""

    def __init__(self, encryption_manager: EncryptionManager):
        """Initialisiert MessageEncryption.

        Args:
            encryption_manager: EncryptionManager-Instanz
        """
        self.encryption_manager = encryption_manager
        self.asymmetric_encryption = AsymmetricEncryption()

        logger.info("MessageEncryption initialisiert")

    async def encrypt_message(self, sender_id: str, recipient_id: str,
                            message: dict[str, Any],
                            recipient_public_key: bytes) -> SecureMessage:
        """Verschlüsselt Nachricht für Empfänger.

        Args:
            sender_id: Sender-ID
            recipient_id: Empfänger-ID
            message: Nachricht
            recipient_public_key: Public Key des Empfängers

        Returns:
            Verschlüsselte SecureMessage
        """
        try:
            # Serialisiere Nachricht
            message_json = json.dumps(message, default=str)

            # Hybrid-Verschlüsselung für große Nachrichten
            encrypted_data = self.asymmetric_encryption.encrypt_hybrid(
                message_json, recipient_public_key
            )

            # Erstelle Message-ID
            import secrets
            message_id = secrets.token_hex(16)

            # Signiere Nachricht
            sender_private_key = await self._get_agent_private_key(sender_id)
            signature = self.asymmetric_encryption.sign(
                message_json, sender_private_key
            )

            return SecureMessage(
                message_id=message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                encrypted_content=json.dumps({
                    "encrypted_data": {
                        "encrypted_data": encrypted_data["encrypted_data"],
                        "encrypted_key": encrypted_data["encrypted_key"].hex(),
                        "algorithm": encrypted_data["algorithm"]
                    }
                }),
                signature=signature.hex(),
                timestamp=datetime.now(UTC),
                message_type="encrypted"
            )

        except Exception as e:
            logger.error(f"Message-Verschlüsselung fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Message-Verschlüsselung fehlgeschlagen: {e}")

    async def decrypt_message(self, secure_message: SecureMessage,
                            recipient_private_key: bytes,
                            sender_public_key: bytes) -> dict[str, Any]:
        """Entschlüsselt SecureMessage.

        Args:
            secure_message: Verschlüsselte Nachricht
            recipient_private_key: Private Key des Empfängers
            sender_public_key: Public Key des Senders

        Returns:
            Entschlüsselte Nachricht
        """
        try:
            # Parse encrypted content
            encrypted_content = json.loads(secure_message.encrypted_content)
            encrypted_data = encrypted_content["encrypted_data"]

            # Rekonstruiere encrypted_data für Decryption
            hybrid_data = {
                "encrypted_data": encrypted_data["encrypted_data"],
                "encrypted_key": bytes.fromhex(encrypted_data["encrypted_key"]),
                "algorithm": encrypted_data["algorithm"]
            }

            # Entschlüssele Nachricht
            decrypted_json = self.asymmetric_encryption.decrypt_hybrid(
                hybrid_data, recipient_private_key
            )

            # Verifiziere Signatur
            signature = bytes.fromhex(secure_message.signature)
            signature_valid = self.asymmetric_encryption.verify(
                decrypted_json, signature, sender_public_key
            )

            if not signature_valid:
                raise MessageIntegrityError("Nachrichtensignatur ungültig")

            # Deserialisiere Nachricht
            return json.loads(decrypted_json)

        except Exception as e:
            logger.error(f"Message-Entschlüsselung fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Message-Entschlüsselung fehlgeschlagen: {e}")

    async def _get_agent_private_key(self, agent_id: str) -> bytes:
        """Holt Private Key für Agent (Mock-Implementierung).

        Args:
            agent_id: Agent-ID

        Returns:
            Private Key als PEM bytes
        """
        # In echter Implementierung würde hier der Private Key
        # aus sicherem Storage geholt werden
        key_id = f"agent_{agent_id}_private_key"

        try:
            # Versuche Key aus KeyManager zu holen
            private_key_data = self.encryption_manager.key_manager.get_key(key_id)
            return private_key_data
        except KeyNotFoundError:
            # Generiere neuen Key wenn nicht vorhanden
            private_key, public_key = self.asymmetric_encryption.generate_key_pair()
            self.encryption_manager.key_manager.store_key(key_id, private_key)
            return private_key
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Private Keys für Agent {agent_id}: {e}")
            raise CommunicationSecurityError(f"Private Key konnte nicht abgerufen werden: {e}")


class CertificateManager:
    """Certificate Management und PKI."""

    def __init__(self, ca_certificate: Certificate | None = None):
        """Initialisiert CertificateManager.

        Args:
            ca_certificate: Optional CA-Certificate
        """
        self.ca_certificate = ca_certificate
        self._certificates: dict[str, Certificate] = {}
        self._trusted_cas: dict[str, Certificate] = {}

        if ca_certificate:
            self._trusted_cas[ca_certificate.certificate_id] = ca_certificate

        logger.info("CertificateManager initialisiert")

    async def generate_certificate(self, subject_name: str, agent_id: str,
                                 valid_days: int = 365) -> Certificate:
        """Generiert neues Certificate für Agent.

        Args:
            subject_name: Subject Name
            agent_id: Agent-ID
            valid_days: Gültigkeitsdauer in Tagen

        Returns:
            Neues Certificate
        """
        try:
            # Generiere Key Pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )

            # Erstelle Certificate
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "DE"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Bayern"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "München"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "KEI-Agent System"),
                x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
            ])

            # Self-signed wenn keine CA vorhanden
            issuer = subject if not self.ca_certificate else x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, self.ca_certificate.subject)
            ])

            valid_from = datetime.now(UTC)
            valid_until = valid_from + timedelta(days=valid_days)

            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                valid_from
            ).not_valid_after(
                valid_until
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(f"agent-{agent_id}"),
                    x509.DNSName(f"{agent_id}.kei-agents.local"),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256(), default_backend())

            # Serialisiere Certificate und Key
            cert_pem = cert.public_bytes(serialization.Encoding.PEM)
            private_key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_key_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            # Erstelle Certificate-Objekt
            certificate = Certificate(
                certificate_id=f"cert_{agent_id}",
                subject=subject_name,
                issuer=self.ca_certificate.subject if self.ca_certificate else subject_name,
                public_key=public_key_pem,
                private_key=private_key_pem,
                valid_from=valid_from,
                valid_until=valid_until,
                certificate_pem=cert_pem
            )

            # Speichere Certificate
            self._certificates[certificate.certificate_id] = certificate

            logger.info(f"Certificate für Agent {agent_id} generiert")
            return certificate

        except Exception as e:
            logger.error(f"Certificate-Generierung fehlgeschlagen: {e}")
            raise CertificateError(f"Certificate-Generierung fehlgeschlagen: {e}")

    async def verify_certificate(self, certificate: Certificate) -> bool:
        """Verifiziert Certificate.

        Args:
            certificate: Zu verifizierendes Certificate

        Returns:
            True wenn gültig
        """
        try:
            # Parse Certificate
            cert = x509.load_pem_x509_certificate(
                certificate.certificate_pem, default_backend()
            )

            # Prüfe Gültigkeitsdauer
            now = datetime.now(UTC)
            if now < cert.not_valid_before or now > cert.not_valid_after:
                logger.warning(f"Certificate {certificate.certificate_id} abgelaufen")
                return False

            # Prüfe Issuer wenn CA vorhanden
            if self.ca_certificate:
                ca_cert = x509.load_pem_x509_certificate(
                    self.ca_certificate.certificate_pem, default_backend()
                )

                # Verifiziere Signatur
                try:
                    ca_cert.public_key().verify(
                        cert.signature,
                        cert.tbs_certificate_bytes,
                        cert.signature_algorithm_oid._name
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Certificate {certificate.certificate_id} Signatur ungültig - Parameter-Fehler: {e}")
                    return False
                except Exception as e:
                    logger.warning(f"Certificate {certificate.certificate_id} Signatur ungültig - Unerwarteter Fehler: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Certificate-Verifikation fehlgeschlagen: {e}")
            return False

    async def get_certificate(self, certificate_id: str) -> Certificate | None:
        """Ruft Certificate ab.

        Args:
            certificate_id: Certificate-ID

        Returns:
            Certificate oder None
        """
        return self._certificates.get(certificate_id)

    async def revoke_certificate(self, certificate_id: str) -> bool:
        """Widerruft Certificate.

        Args:
            certificate_id: Certificate-ID

        Returns:
            True wenn erfolgreich
        """
        try:
            if certificate_id in self._certificates:
                del self._certificates[certificate_id]
                logger.info(f"Certificate {certificate_id} widerrufen")
                return True
            return False

        except Exception as e:
            logger.error(f"Certificate-Widerruf fehlgeschlagen: {e}")
            return False


class SecureCommunicationManager:
    """Zentraler Secure Communication Manager."""

    def __init__(self, encryption_config: EncryptionConfig):
        """Initialisiert SecureCommunicationManager.

        Args:
            encryption_config: Encryption-Konfiguration
        """
        self.encryption_config = encryption_config
        self.encryption_manager = EncryptionManager(encryption_config)
        self.message_encryption = MessageEncryption(self.encryption_manager)
        self.certificate_manager = CertificateManager()

        # Secure Channels
        self._channels: dict[str, SecureChannel] = {}

        logger.info("SecureCommunicationManager initialisiert")

    async def establish_secure_channel(self, agent_a: str, agent_b: str) -> SecureChannel:
        """Etabliert Secure Channel zwischen Agents.

        Args:
            agent_a: Agent A ID
            agent_b: Agent B ID

        Returns:
            Etablierter SecureChannel
        """
        try:
            # Generiere Channel-ID
            import secrets
            channel_id = f"channel_{agent_a}_{agent_b}_{secrets.token_hex(8)}"

            # Generiere Shared Secret
            shared_secret = secrets.token_bytes(32)  # 256-bit shared secret

            # Erstelle Channel
            channel = SecureChannel(
                channel_id=channel_id,
                agent_a=agent_a,
                agent_b=agent_b,
                shared_secret=shared_secret,
                established_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=24)
            )

            self._channels[channel_id] = channel

            logger.info(f"Secure Channel {channel_id} zwischen {agent_a} und {agent_b} etabliert")
            return channel

        except Exception as e:
            logger.error(f"Secure Channel Establishment fehlgeschlagen: {e}")
            raise ChannelEstablishmentError(f"Channel Establishment fehlgeschlagen: {e}")

    async def encrypt_communication(self, sender_id: str, recipient_id: str,
                                  message: dict[str, Any]) -> SecureMessage:
        """Verschlüsselt Kommunikation zwischen Agents.

        Args:
            sender_id: Sender-ID
            recipient_id: Empfänger-ID
            message: Nachricht

        Returns:
            Verschlüsselte SecureMessage
        """
        try:
            # Hole oder generiere Certificate für Empfänger
            recipient_cert = await self.certificate_manager.get_certificate(f"cert_{recipient_id}")
            if not recipient_cert:
                recipient_cert = await self.certificate_manager.generate_certificate(
                    f"Agent {recipient_id}", recipient_id
                )

            # Verschlüssele Nachricht
            secure_message = await self.message_encryption.encrypt_message(
                sender_id, recipient_id, message, recipient_cert.public_key
            )

            return secure_message

        except Exception as e:
            logger.error(f"Communication-Verschlüsselung fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Communication-Verschlüsselung fehlgeschlagen: {e}")

    async def decrypt_communication(self, secure_message: SecureMessage,
                                  recipient_id: str) -> dict[str, Any]:
        """Entschlüsselt Kommunikation.

        Args:
            secure_message: Verschlüsselte Nachricht
            recipient_id: Empfänger-ID

        Returns:
            Entschlüsselte Nachricht
        """
        try:
            # Hole Certificates
            recipient_cert = await self.certificate_manager.get_certificate(f"cert_{recipient_id}")
            sender_cert = await self.certificate_manager.get_certificate(f"cert_{secure_message.sender_id}")

            if not recipient_cert or not sender_cert:
                raise CommunicationSecurityError("Certificates nicht verfügbar")

            # Entschlüssele Nachricht
            decrypted_message = await self.message_encryption.decrypt_message(
                secure_message, recipient_cert.private_key, sender_cert.public_key
            )

            return decrypted_message

        except Exception as e:
            logger.error(f"Communication-Entschlüsselung fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Communication-Entschlüsselung fehlgeschlagen: {e}")

    async def verify_message_integrity(self, secure_message: SecureMessage) -> bool:
        """Verifiziert Message Integrity.

        Args:
            secure_message: Zu verifizierende Nachricht

        Returns:
            True wenn Integrität gewährleistet
        """
        try:
            # Hole Sender Certificate
            sender_cert = await self.certificate_manager.get_certificate(f"cert_{secure_message.sender_id}")
            if not sender_cert:
                return False

            # Verifiziere Certificate
            cert_valid = await self.certificate_manager.verify_certificate(sender_cert)
            if not cert_valid:
                return False

            # Message Integrity wird bei Decryption geprüft
            # Hier zusätzliche Checks möglich

            return True

        except Exception as e:
            logger.error(f"Message Integrity Verification fehlgeschlagen: {e}")
            return False

    async def get_secure_channel(self, channel_id: str) -> SecureChannel | None:
        """Ruft Secure Channel ab.

        Args:
            channel_id: Channel-ID

        Returns:
            SecureChannel oder None
        """
        channel = self._channels.get(channel_id)

        if channel and channel.expires_at and datetime.now(UTC) > channel.expires_at:
            # Channel abgelaufen
            channel.is_active = False
            return None

        return channel

    async def close_secure_channel(self, channel_id: str) -> bool:
        """Schließt Secure Channel.

        Args:
            channel_id: Channel-ID

        Returns:
            True wenn erfolgreich
        """
        try:
            if channel_id in self._channels:
                self._channels[channel_id].is_active = False
                del self._channels[channel_id]
                logger.info(f"Secure Channel {channel_id} geschlossen")
                return True
            return False

        except Exception as e:
            logger.error(f"Secure Channel schließen fehlgeschlagen: {e}")
            return False


class MessageAuthentication:
    """Message Authentication System."""

    def __init__(self, algorithm: str = "HMAC-SHA256"):
        """Initialisiert MessageAuthentication.

        Args:
            algorithm: Authentication-Algorithmus
        """
        self.algorithm = algorithm

        logger.info(f"MessageAuthentication initialisiert mit {algorithm}")

    @staticmethod
    async def authenticate_message(message: bytes, key: bytes) -> bytes:
        """Authentifiziert Nachricht mit HMAC.

        Args:
            message: Nachricht
            key: Authentication-Key

        Returns:
            HMAC-Tag
        """
        try:
            mac = hmac.new(key, message, hashlib.sha256)
            return mac.digest()

        except Exception as e:
            logger.error(f"Message Authentication fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Message Authentication fehlgeschlagen: {e}")

    @staticmethod
    async def verify_message(message: bytes, key: bytes, tag: bytes) -> bool:
        """Verifiziert Nachrichten-Authentizität.

        Args:
            message: Nachricht
            key: Authentication-Key
            tag: HMAC-Tag

        Returns:
            True wenn authentisch
        """
        try:
            expected_tag = await MessageAuthentication.authenticate_message(message, key)
            return hmac.compare_digest(tag, expected_tag)

        except Exception as e:
            logger.error(f"Message Verification fehlgeschlagen: {e}")
            return False


class SecurityLevel(Enum):
    """Security Level Enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class KeyExchange:
    """Key Exchange System."""

    def __init__(self, algorithm: str = "ECDH"):
        """Initialisiert KeyExchange.

        Args:
            algorithm: Key Exchange Algorithmus
        """
        self.algorithm = algorithm

        logger.info(f"KeyExchange initialisiert mit {algorithm}")

    @staticmethod
    async def generate_key_pair() -> tuple[bytes, bytes]:
        """Generiert Key Pair für Key Exchange.

        Returns:
            Tuple (private_key, public_key)
        """
        try:
            # Mock Key Pair Generation
            private_key = os.urandom(32)
            public_key = os.urandom(32)

            return private_key, public_key

        except Exception as e:
            logger.error(f"Key Pair Generation fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Key Pair Generation fehlgeschlagen: {e}")

    @staticmethod
    async def derive_shared_secret(private_key: bytes, public_key: bytes) -> bytes:
        """Leitet Shared Secret ab.

        Args:
            private_key: Private Key
            public_key: Public Key der Gegenstelle

        Returns:
            Shared Secret
        """
        try:
            # Mock Shared Secret Derivation
            combined = private_key + public_key
            shared_secret = hashlib.sha256(combined).digest()

            return shared_secret

        except Exception as e:
            logger.error(f"Shared Secret Derivation fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Shared Secret Derivation fehlgeschlagen: {e}")


class CommunicationProtocol:
    """Communication Protocol Manager."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        """Initialisiert CommunicationProtocol.

        Args:
            security_level: Security Level
        """
        self.security_level = security_level
        self.key_exchange = KeyExchange()
        self.message_auth = MessageAuthentication()

        logger.info(f"CommunicationProtocol initialisiert mit {security_level.value}")

    async def establish_secure_session(self, agent_a: str, agent_b: str) -> dict[str, Any]:
        """Etabliert Secure Session zwischen Agents.

        Args:
            agent_a: Agent A ID
            agent_b: Agent B ID

        Returns:
            Session-Informationen
        """
        try:
            # Key Exchange
            _private_key, _public_key = await self.key_exchange.generate_key_pair()

            # Mock Session Establishment
            session_id = secrets.token_hex(16)

            return {
                "session_id": session_id,
                "agent_a": agent_a,
                "agent_b": agent_b,
                "security_level": self.security_level.value,
                "established_at": datetime.now(UTC)
            }

        except Exception as e:
            logger.error(f"Secure Session Establishment fehlgeschlagen: {e}")
            raise ChannelEstablishmentError(f"Secure Session Establishment fehlgeschlagen: {e}")

    async def send_secure_message(self, session_id: str, message: dict[str, Any]) -> dict[str, Any]:
        """Sendet Secure Message.

        Args:
            session_id: Session-ID
            message: Nachricht

        Returns:
            Versand-Ergebnis
        """
        try:
            # Mock Secure Message Sending
            message_id = secrets.token_hex(8)

            return {
                "message_id": message_id,
                "session_id": session_id,
                "sent_at": datetime.now(UTC),
                "encrypted": True,
                "authenticated": True
            }

        except Exception as e:
            logger.error(f"Secure Message Sending fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Secure Message Sending fehlgeschlagen: {e}")

    async def receive_secure_message(self, session_id: str,
                                   encrypted_message: dict[str, Any]) -> dict[str, Any]:
        """Empfängt Secure Message.

        Args:
            session_id: Session-ID
            encrypted_message: Verschlüsselte Nachricht

        Returns:
            Entschlüsselte Nachricht
        """
        try:
            # Mock Secure Message Receiving
            return {
                "message_id": encrypted_message.get("message_id"),
                "session_id": session_id,
                "content": "Decrypted message content",
                "received_at": datetime.now(UTC),
                "verified": True
            }

        except Exception as e:
            logger.error(f"Secure Message Receiving fehlgeschlagen: {e}")
            raise CommunicationSecurityError(f"Secure Message Receiving fehlgeschlagen: {e}")
