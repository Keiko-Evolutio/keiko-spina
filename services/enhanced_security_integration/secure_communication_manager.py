# backend/services/enhanced_security_integration/secure_communication_manager.py
"""Secure Communication Manager für Service-to-Service-Kommunikation.

Implementiert mTLS, End-to-End-Encryption und sichere Communication Channels
zwischen allen Services mit Performance-Optimierung.
"""

from __future__ import annotations

import asyncio
import ssl
import time
from typing import Any

import aiohttp
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kei_logging import get_logger

from .data_models import (
    EncryptionAlgorithm,
    SecureCommunicationChannel,
    SecurityContext,
    SecurityEvent,
    SecurityEventType,
    ThreatLevel,
)

logger = get_logger(__name__)


class SecureCommunicationManager:
    """Secure Communication Manager für Service-to-Service-Kommunikation."""

    def __init__(self):
        """Initialisiert Secure Communication Manager."""
        # mTLS-Konfiguration
        self.enable_mtls = True
        self.require_client_certificates = True
        self.certificate_validation_strict = True

        # Encryption-Konfiguration
        self.default_encryption_algorithm = EncryptionAlgorithm.AES_256_GCM
        self.enable_end_to_end_encryption = True
        self.key_rotation_interval_hours = 24

        # Communication-Channels
        self._communication_channels: dict[str, SecureCommunicationChannel] = {}
        self._channel_encryption_keys: dict[str, bytes] = {}

        # SSL-Context-Cache
        self._ssl_contexts: dict[str, ssl.SSLContext] = {}

        # Certificate-Management
        self._service_certificates: dict[str, dict[str, Any]] = {}
        self._trusted_ca_certificates: list[Any] = []

        # Performance-Tracking
        self._communication_count = 0
        self._total_communication_time_ms = 0.0
        self._encryption_overhead_ms = 0.0
        self._mtls_handshake_time_ms = 0.0

        # Security-Events
        self._security_events: list[SecurityEvent] = []

        # Background-Tasks
        self._key_rotation_task: asyncio.Task | None = None
        self._certificate_renewal_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Secure Communication Manager initialisiert")

    async def start(self) -> None:
        """Startet Secure Communication Manager."""
        if self._is_running:
            return

        self._is_running = True

        # Initialisiere SSL-Contexts
        await self._initialize_ssl_contexts()

        # Lade Service-Certificates
        await self._load_service_certificates()

        # Starte Background-Tasks
        self._key_rotation_task = asyncio.create_task(self._key_rotation_loop())
        self._certificate_renewal_task = asyncio.create_task(self._certificate_renewal_loop())

        logger.info("Secure Communication Manager gestartet")

    async def stop(self) -> None:
        """Stoppt Secure Communication Manager."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._key_rotation_task:
            self._key_rotation_task.cancel()
        if self._certificate_renewal_task:
            self._certificate_renewal_task.cancel()

        await asyncio.gather(
            self._key_rotation_task,
            self._certificate_renewal_task,
            return_exceptions=True
        )

        logger.info("Secure Communication Manager gestoppt")

    async def establish_secure_channel(
        self,
        source_service: str,
        target_service: str,
        channel_config: dict[str, Any] | None = None
    ) -> SecureCommunicationChannel:
        """Etabliert sicheren Communication Channel.

        Args:
            source_service: Source Service Name
            target_service: Target Service Name
            channel_config: Channel-Konfiguration

        Returns:
            Secure Communication Channel
        """
        try:
            channel_id = f"{source_service}_to_{target_service}"

            logger.debug({
                "event": "establish_secure_channel_started",
                "channel_id": channel_id,
                "source_service": source_service,
                "target_service": target_service
            })

            # Erstelle Channel-Konfiguration
            config = channel_config or {}

            channel = SecureCommunicationChannel(
                channel_id=channel_id,
                source_service=source_service,
                target_service=target_service,
                encryption_enabled=config.get("encryption_enabled", True),
                encryption_algorithm=EncryptionAlgorithm(
                    config.get("encryption_algorithm", self.default_encryption_algorithm.value)
                ),
                mtls_enabled=config.get("mtls_enabled", self.enable_mtls),
                client_certificate_required=config.get("client_certificate_required", self.require_client_certificates),
                server_certificate_validation=config.get("server_certificate_validation", self.certificate_validation_strict),
                allowed_operations=config.get("allowed_operations", []),
                rate_limits=config.get("rate_limits", {}),
                security_monitoring_enabled=config.get("security_monitoring_enabled", True),
                threat_detection_enabled=config.get("threat_detection_enabled", True)
            )

            # Generiere Encryption-Key für Channel
            if channel.encryption_enabled:
                encryption_key = self._generate_channel_encryption_key()
                self._channel_encryption_keys[channel_id] = encryption_key

            # Speichere Channel
            self._communication_channels[channel_id] = channel

            logger.info({
                "event": "secure_channel_established",
                "channel_id": channel_id,
                "encryption_enabled": channel.encryption_enabled,
                "mtls_enabled": channel.mtls_enabled
            })

            return channel

        except Exception as e:
            logger.error(f"Secure channel establishment fehlgeschlagen: {e}")
            raise

    async def send_secure_message(
        self,
        channel_id: str,
        message_data: dict[str, Any],
        target_url: str,
        security_context: SecurityContext | None = None
    ) -> dict[str, Any]:
        """Sendet sichere Nachricht über Channel.

        Args:
            channel_id: Channel-ID
            message_data: Nachrichtendaten
            target_url: Target-URL
            security_context: Security-Kontext

        Returns:
            Response-Daten
        """
        start_time = time.time()

        try:
            # Hole Channel
            channel = self._communication_channels.get(channel_id)
            if not channel:
                raise ValueError(f"Channel {channel_id} nicht gefunden")

            logger.debug({
                "event": "send_secure_message_started",
                "channel_id": channel_id,
                "target_url": target_url,
                "encryption_enabled": channel.encryption_enabled
            })

            # Verschlüssele Message (falls aktiviert)
            if channel.encryption_enabled:
                encrypted_message = await self._encrypt_message(channel_id, message_data)
                payload = {
                    "encrypted": True,
                    "algorithm": channel.encryption_algorithm.value,
                    "data": encrypted_message
                }
            else:
                payload = {
                    "encrypted": False,
                    "data": message_data
                }

            # Erstelle SSL-Context für mTLS
            ssl_context = None
            if channel.mtls_enabled:
                ssl_context = await self._get_ssl_context_for_channel(channel)

            # Sende HTTP-Request
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "X-Channel-ID": channel_id,
                    "X-Source-Service": channel.source_service
                }

                # Füge Security-Context-Headers hinzu
                if security_context:
                    headers.update(self._create_security_headers(security_context))

                mtls_start_time = time.time()

                async with session.post(
                    target_url,
                    json=payload,
                    headers=headers,
                    ssl=ssl_context,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    mtls_time_ms = (time.time() - mtls_start_time) * 1000
                    self._mtls_handshake_time_ms += mtls_time_ms

                    response.raise_for_status()
                    response_data = await response.json()

            # Entschlüssele Response (falls verschlüsselt)
            if response_data.get("encrypted", False):
                decrypted_response = await self._decrypt_message(channel_id, response_data["data"])
                final_response = decrypted_response
            else:
                final_response = response_data.get("data", response_data)

            # Performance-Tracking
            communication_time_ms = (time.time() - start_time) * 1000
            self._update_communication_performance_stats(communication_time_ms)

            # Update Channel-Usage
            from datetime import datetime
            channel.last_used_at = datetime.utcnow()

            logger.debug({
                "event": "send_secure_message_completed",
                "channel_id": channel_id,
                "communication_time_ms": communication_time_ms,
                "mtls_time_ms": mtls_time_ms
            })

            return final_response

        except Exception as e:
            logger.error(f"Secure message sending fehlgeschlagen für Channel {channel_id}: {e}")

            # Logge Security-Event
            if security_context:
                await self._log_security_event(
                    SecurityEventType.ENCRYPTION_FAILURE,
                    ThreatLevel.MEDIUM,
                    security_context,
                    f"Secure communication failed for channel {channel_id}: {e}"
                )

            raise

    async def _encrypt_message(self, channel_id: str, message_data: dict[str, Any]) -> str:
        """Verschlüsselt Nachricht für Channel."""
        try:
            encryption_key = self._channel_encryption_keys.get(channel_id)
            if not encryption_key:
                raise ValueError(f"Encryption key für Channel {channel_id} nicht gefunden")

            # Serialisiere Message
            import json
            message_json = json.dumps(message_data, sort_keys=True, ensure_ascii=False)
            message_bytes = message_json.encode("utf-8")

            # Verschlüssele mit AES-GCM
            import os
            iv = os.urandom(12)  # 96-bit IV für GCM
            aesgcm = AESGCM(encryption_key)
            encrypted_data = aesgcm.encrypt(iv, message_bytes, None)

            # Kombiniere IV und encrypted data
            import base64
            combined_data = iv + encrypted_data
            encrypted_message = base64.b64encode(combined_data).decode("ascii")

            return encrypted_message

        except Exception as e:
            logger.error(f"Message encryption fehlgeschlagen: {e}")
            raise

    async def _decrypt_message(self, channel_id: str, encrypted_message: str) -> dict[str, Any]:
        """Entschlüsselt Nachricht für Channel."""
        try:
            encryption_key = self._channel_encryption_keys.get(channel_id)
            if not encryption_key:
                raise ValueError(f"Encryption key für Channel {channel_id} nicht gefunden")

            # Dekodiere Base64
            import base64
            combined_data = base64.b64decode(encrypted_message.encode("ascii"))

            # Separiere IV und encrypted data
            iv = combined_data[:12]  # 96-bit IV
            encrypted_data = combined_data[12:]

            # Entschlüssele mit AES-GCM
            aesgcm = AESGCM(encryption_key)
            decrypted_bytes = aesgcm.decrypt(iv, encrypted_data, None)

            # Deserialisiere Message
            import json
            message_json = decrypted_bytes.decode("utf-8")
            message_data = json.loads(message_json)

            return message_data

        except Exception as e:
            logger.error(f"Message decryption fehlgeschlagen: {e}")
            raise

    async def _get_ssl_context_for_channel(self, channel: SecureCommunicationChannel) -> ssl.SSLContext:
        """Erstellt SSL-Context für Channel."""
        try:
            context_key = f"{channel.source_service}_{channel.target_service}"

            # Prüfe Cache
            if context_key in self._ssl_contexts:
                return self._ssl_contexts[context_key]

            # Erstelle neuen SSL-Context
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            # Konfiguriere für mTLS
            if channel.mtls_enabled:
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED

                # Lade Client-Certificate
                service_cert = self._service_certificates.get(channel.source_service)
                if service_cert:
                    context.load_cert_chain(
                        service_cert["cert_file"],
                        service_cert["key_file"]
                    )

                # Lade CA-Certificates
                if self._trusted_ca_certificates:
                    # TODO: Lade CA-Certificates in Context - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111
                    pass

            # Cache Context
            self._ssl_contexts[context_key] = context

            return context

        except Exception as e:
            logger.error(f"SSL context creation fehlgeschlagen: {e}")
            raise

    def _create_security_headers(self, security_context: SecurityContext) -> dict[str, str]:
        """Erstellt Security-Headers für Request."""
        headers = {}

        if security_context.user_id:
            headers["X-User-ID"] = security_context.user_id

        if security_context.tenant_id:
            headers["X-Tenant-ID"] = security_context.tenant_id

        if security_context.request_id:
            headers["X-Request-ID"] = security_context.request_id

        headers["X-Security-Level"] = security_context.security_level.value

        return headers

    def _generate_channel_encryption_key(self) -> bytes:
        """Generiert Encryption-Key für Channel."""
        import os
        return os.urandom(32)  # 256-bit key

    async def _initialize_ssl_contexts(self) -> None:
        """Initialisiert SSL-Contexts."""
        try:
            # TODO: Initialisiere SSL-Contexts für Services - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111
            logger.debug("SSL contexts initialisiert")

        except Exception as e:
            logger.error(f"SSL context initialization fehlgeschlagen: {e}")

    async def _load_service_certificates(self) -> None:
        """Lädt Service-Certificates."""
        try:
            # TODO: Lade Service-Certificates aus Konfiguration - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111 - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111
            logger.debug("Service certificates geladen")

        except Exception as e:
            logger.error(f"Service certificate loading fehlgeschlagen: {e}")

    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        security_context: SecurityContext,
        description: str
    ) -> None:
        """Loggt Security-Event."""
        try:
            import uuid

            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                threat_level=threat_level,
                security_context=security_context,
                description=description
            )

            self._security_events.append(event)

            # Memory-Limit prüfen
            if len(self._security_events) > 10000:
                self._security_events = self._security_events[-10000:]

            logger.warning({
                "event": "security_event",
                "event_id": event.event_id,
                "event_type": event_type.value,
                "threat_level": threat_level.value,
                "description": description
            })

        except Exception as e:
            logger.error(f"Security event logging fehlgeschlagen: {e}")

    async def _key_rotation_loop(self) -> None:
        """Background-Loop für Key-Rotation."""
        while self._is_running:
            try:
                await asyncio.sleep(self.key_rotation_interval_hours * 3600)

                if self._is_running:
                    await self._rotate_channel_keys()

            except Exception as e:
                logger.error(f"Key rotation fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _certificate_renewal_loop(self) -> None:
        """Background-Loop für Certificate-Renewal."""
        while self._is_running:
            try:
                await asyncio.sleep(24 * 3600)  # Prüfe täglich

                if self._is_running:
                    await self._check_certificate_expiry()

            except Exception as e:
                logger.error(f"Certificate renewal check fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _rotate_channel_keys(self) -> None:
        """Rotiert Channel-Encryption-Keys."""
        try:
            rotated_count = 0

            for channel_id in self._channel_encryption_keys.keys():
                new_key = self._generate_channel_encryption_key()
                self._channel_encryption_keys[channel_id] = new_key
                rotated_count += 1

            if rotated_count > 0:
                logger.info(f"Channel keys rotiert: {rotated_count} channels")

        except Exception as e:
            logger.error(f"Channel key rotation fehlgeschlagen: {e}")

    async def _check_certificate_expiry(self) -> None:
        """Prüft Certificate-Expiry."""
        try:
            # TODO: Implementiere Certificate-Expiry-Check - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111
            logger.debug("Certificate expiry geprüft")

        except Exception as e:
            logger.error(f"Certificate expiry check fehlgeschlagen: {e}")

    def _update_communication_performance_stats(self, communication_time_ms: float) -> None:
        """Aktualisiert Communication-Performance-Statistiken."""
        self._communication_count += 1
        self._total_communication_time_ms += communication_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_communication_time = (
            self._total_communication_time_ms / self._communication_count
            if self._communication_count > 0 else 0.0
        )

        avg_mtls_handshake_time = (
            self._mtls_handshake_time_ms / self._communication_count
            if self._communication_count > 0 else 0.0
        )

        return {
            "total_communications": self._communication_count,
            "avg_communication_time_ms": avg_communication_time,
            "avg_mtls_handshake_time_ms": avg_mtls_handshake_time,
            "encryption_overhead_ms": self._encryption_overhead_ms,
            "active_channels": len(self._communication_channels),
            "cached_ssl_contexts": len(self._ssl_contexts),
            "security_events": len(self._security_events),
            "mtls_enabled": self.enable_mtls,
            "end_to_end_encryption_enabled": self.enable_end_to_end_encryption
        }
