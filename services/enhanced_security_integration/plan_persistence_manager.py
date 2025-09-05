# backend/services/enhanced_security_integration/plan_persistence_manager.py
"""Plan Persistence Manager für sichere State-Verwaltung.

Implementiert verschlüsselte State-Speicherung mit Tamper-Protection
und Integrity-Verification für Enterprise-Grade Security.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kei_logging import get_logger

from .data_models import (
    EncryptedState,
    EncryptionAlgorithm,
    SecurityLevel,
    StateIntegrityCheck,
    StateIntegrityStatus,
)

logger = get_logger(__name__)


class PlanPersistenceManager:
    """Plan Persistence Manager für sichere State-Verwaltung."""

    def __init__(self, encryption_key: bytes | None = None):
        """Initialisiert Plan Persistence Manager.

        Args:
            encryption_key: Encryption-Key (wird generiert falls None)
        """
        # Encryption-Konfiguration
        self.encryption_algorithm = EncryptionAlgorithm.AES_256_GCM
        self.key_rotation_interval_hours = 24
        self.enable_tamper_protection = True
        self.enable_integrity_verification = True

        # Encryption-Keys
        self._master_key = encryption_key or self._generate_master_key()
        self._current_key_id = "key_001"
        self._encryption_keys: dict[str, bytes] = {
            self._current_key_id: self._master_key
        }

        # State-Storage
        self._encrypted_states: dict[str, EncryptedState] = {}
        self._state_access_log: list[dict[str, Any]] = []

        # Performance-Tracking
        self._encryption_count = 0
        self._decryption_count = 0
        self._total_encryption_time_ms = 0.0
        self._total_decryption_time_ms = 0.0
        self._integrity_check_count = 0

        # Background-Tasks
        self._key_rotation_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Plan Persistence Manager initialisiert")

    async def start(self) -> None:
        """Startet Plan Persistence Manager."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Key-Rotation-Task
        self._key_rotation_task = asyncio.create_task(self._key_rotation_loop())

        logger.info("Plan Persistence Manager gestartet")

    async def stop(self) -> None:
        """Stoppt Plan Persistence Manager."""
        self._is_running = False

        if self._key_rotation_task:
            self._key_rotation_task.cancel()
            try:
                await self._key_rotation_task
            except asyncio.CancelledError:
                pass

        logger.info("Plan Persistence Manager gestoppt")

    async def encrypt_and_store_state(
        self,
        state_id: str,
        state_data: dict[str, Any],
        tenant_id: str | None = None,
        security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL
    ) -> EncryptedState:
        """Verschlüsselt und speichert State.

        Args:
            state_id: State-ID
            state_data: State-Daten
            tenant_id: Tenant-ID
            security_level: Security-Level

        Returns:
            Encrypted State
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "encrypt_store_state_started",
                "state_id": state_id,
                "tenant_id": tenant_id,
                "security_level": security_level.value
            })

            # Serialisiere State-Daten
            state_json = json.dumps(state_data, sort_keys=True, ensure_ascii=False)
            state_bytes = state_json.encode("utf-8")

            # Verschlüssele State
            encrypted_data, iv, auth_tag = await self._encrypt_data(state_bytes)

            # Erstelle Integrity-Hash
            integrity_hash = self._calculate_integrity_hash(state_bytes)

            # Erstelle Tamper-Protection-Signature
            tamper_signature = self._create_tamper_protection_signature(
                state_id, encrypted_data, iv, auth_tag
            )

            # Erstelle EncryptedState
            encrypted_state = EncryptedState(
                state_id=state_id,
                encryption_algorithm=self.encryption_algorithm,
                key_id=self._current_key_id,
                encrypted_data=encrypted_data,
                initialization_vector=iv,
                authentication_tag=auth_tag,
                integrity_hash=integrity_hash,
                tamper_protection_signature=tamper_signature,
                tenant_id=tenant_id,
                security_level=security_level
            )

            # Speichere State
            self._encrypted_states[state_id] = encrypted_state

            # Logge Access
            await self._log_state_access(state_id, "encrypt_store", tenant_id)

            # Performance-Tracking
            encryption_time_ms = (time.time() - start_time) * 1000
            self._update_encryption_performance_stats(encryption_time_ms)

            logger.debug({
                "event": "encrypt_store_state_completed",
                "state_id": state_id,
                "encryption_time_ms": encryption_time_ms,
                "data_size_bytes": len(state_bytes)
            })

            return encrypted_state

        except Exception as e:
            logger.error(f"State encryption/storage fehlgeschlagen für {state_id}: {e}")
            raise

    async def retrieve_and_decrypt_state(
        self,
        state_id: str,
        tenant_id: str | None = None,
        verify_integrity: bool = True
    ) -> dict[str, Any] | None:
        """Holt und entschlüsselt State.

        Args:
            state_id: State-ID
            tenant_id: Tenant-ID für Access-Control
            verify_integrity: Integrity-Verification durchführen

        Returns:
            Entschlüsselte State-Daten oder None
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "retrieve_decrypt_state_started",
                "state_id": state_id,
                "tenant_id": tenant_id,
                "verify_integrity": verify_integrity
            })

            # Hole Encrypted State
            encrypted_state = self._encrypted_states.get(state_id)
            if not encrypted_state:
                logger.warning(f"State {state_id} nicht gefunden")
                return None

            # Prüfe Tenant-Access
            if tenant_id and encrypted_state.tenant_id != tenant_id:
                logger.warning(f"Tenant-Access verweigert für State {state_id}")
                return None

            # Integrity-Verification
            if verify_integrity:
                integrity_check = await self.verify_state_integrity(state_id)
                if not integrity_check.is_valid:
                    logger.error(f"State integrity check fehlgeschlagen für {state_id}: {integrity_check.violations}")
                    return None

            # Entschlüssele State
            decrypted_data = await self._decrypt_data(
                encrypted_state.encrypted_data,
                encrypted_state.initialization_vector,
                encrypted_state.authentication_tag,
                encrypted_state.key_id
            )

            # Deserialisiere State-Daten
            state_json = decrypted_data.decode("utf-8")
            state_data = json.loads(state_json)

            # Logge Access
            await self._log_state_access(state_id, "retrieve_decrypt", tenant_id)

            # Performance-Tracking
            decryption_time_ms = (time.time() - start_time) * 1000
            self._update_decryption_performance_stats(decryption_time_ms)

            logger.debug({
                "event": "retrieve_decrypt_state_completed",
                "state_id": state_id,
                "decryption_time_ms": decryption_time_ms,
                "data_size_bytes": len(decrypted_data)
            })

            return state_data

        except Exception as e:
            logger.error(f"State retrieval/decryption fehlgeschlagen für {state_id}: {e}")
            return None

    async def verify_state_integrity(self, state_id: str) -> StateIntegrityCheck:
        """Verifiziert State-Integrität.

        Args:
            state_id: State-ID

        Returns:
            State Integrity Check Result
        """
        start_time = time.time()

        try:
            encrypted_state = self._encrypted_states.get(state_id)
            if not encrypted_state:
                return StateIntegrityCheck(
                    status=StateIntegrityStatus.UNKNOWN,
                    is_valid=False,
                    integrity_hash_valid=False,
                    tamper_protection_valid=False,
                    encryption_valid=False,
                    violations=["state_not_found"]
                )

            violations = []

            # 1. Prüfe Tamper-Protection-Signature
            expected_signature = self._create_tamper_protection_signature(
                encrypted_state.state_id,
                encrypted_state.encrypted_data,
                encrypted_state.initialization_vector,
                encrypted_state.authentication_tag
            )

            tamper_protection_valid = hmac.compare_digest(
                encrypted_state.tamper_protection_signature,
                expected_signature
            )

            if not tamper_protection_valid:
                violations.append("tamper_protection_signature_invalid")

            # 2. Prüfe Encryption-Validity (durch Decryption-Test)
            encryption_valid = True
            try:
                await self._decrypt_data(
                    encrypted_state.encrypted_data,
                    encrypted_state.initialization_vector,
                    encrypted_state.authentication_tag,
                    encrypted_state.key_id
                )
            except Exception:
                encryption_valid = False
                violations.append("encryption_invalid")

            # 3. Prüfe Integrity-Hash (falls Decryption erfolgreich)
            integrity_hash_valid = True
            if encryption_valid:
                try:
                    decrypted_data = await self._decrypt_data(
                        encrypted_state.encrypted_data,
                        encrypted_state.initialization_vector,
                        encrypted_state.authentication_tag,
                        encrypted_state.key_id
                    )

                    expected_hash = self._calculate_integrity_hash(decrypted_data)
                    integrity_hash_valid = hmac.compare_digest(
                        encrypted_state.integrity_hash,
                        expected_hash
                    )

                    if not integrity_hash_valid:
                        violations.append("integrity_hash_invalid")

                except Exception:
                    integrity_hash_valid = False
                    violations.append("integrity_hash_check_failed")

            # Bestimme Overall-Status
            is_valid = len(violations) == 0

            if violations:
                status = StateIntegrityStatus.TAMPERED
            else:
                status = StateIntegrityStatus.VALID

            # Performance-Tracking
            check_duration_ms = (time.time() - start_time) * 1000
            self._integrity_check_count += 1

            result = StateIntegrityCheck(
                status=status,
                is_valid=is_valid,
                integrity_hash_valid=integrity_hash_valid,
                tamper_protection_valid=tamper_protection_valid,
                encryption_valid=encryption_valid,
                violations=violations,
                check_duration_ms=check_duration_ms
            )

            logger.debug({
                "event": "state_integrity_check_completed",
                "state_id": state_id,
                "is_valid": is_valid,
                "violations": violations,
                "check_duration_ms": check_duration_ms
            })

            return result

        except Exception as e:
            logger.error(f"State integrity check fehlgeschlagen für {state_id}: {e}")

            return StateIntegrityCheck(
                status=StateIntegrityStatus.UNKNOWN,
                is_valid=False,
                integrity_hash_valid=False,
                tamper_protection_valid=False,
                encryption_valid=False,
                violations=["integrity_check_error"],
                check_duration_ms=(time.time() - start_time) * 1000
            )

    async def _encrypt_data(self, data: bytes) -> tuple[bytes, bytes, bytes]:
        """Verschlüsselt Daten mit AES-256-GCM.

        Args:
            data: Zu verschlüsselnde Daten

        Returns:
            Tuple (encrypted_data, iv, auth_tag)
        """
        try:
            # Generiere IV
            iv = os.urandom(12)  # 96-bit IV für GCM

            # Verschlüssele mit AES-GCM
            aesgcm = AESGCM(self._master_key)
            encrypted_data = aesgcm.encrypt(iv, data, None)

            # Separiere Auth-Tag (letzten 16 Bytes)
            ciphertext = encrypted_data[:-16]
            auth_tag = encrypted_data[-16:]

            return ciphertext, iv, auth_tag

        except Exception as e:
            logger.error(f"Data encryption fehlgeschlagen: {e}")
            raise

    async def _decrypt_data(self, encrypted_data: bytes, iv: bytes, auth_tag: bytes, key_id: str) -> bytes:
        """Entschlüsselt Daten mit AES-256-GCM.

        Args:
            encrypted_data: Verschlüsselte Daten
            iv: Initialization Vector
            auth_tag: Authentication Tag
            key_id: Key-ID

        Returns:
            Entschlüsselte Daten
        """
        try:
            # Hole Encryption-Key
            encryption_key = self._encryption_keys.get(key_id)
            if not encryption_key:
                raise ValueError(f"Encryption key {key_id} nicht gefunden")

            # Kombiniere Ciphertext und Auth-Tag
            ciphertext_with_tag = encrypted_data + auth_tag

            # Entschlüssele mit AES-GCM
            aesgcm = AESGCM(encryption_key)
            decrypted_data = aesgcm.decrypt(iv, ciphertext_with_tag, None)

            return decrypted_data

        except Exception as e:
            logger.error(f"Data decryption fehlgeschlagen: {e}")
            raise

    def _calculate_integrity_hash(self, data: bytes) -> str:
        """Berechnet Integrity-Hash für Daten."""
        return hashlib.sha256(data).hexdigest()

    def _create_tamper_protection_signature(
        self,
        state_id: str,
        encrypted_data: bytes,
        iv: bytes,
        auth_tag: bytes
    ) -> str:
        """Erstellt Tamper-Protection-Signature."""
        # Kombiniere alle relevanten Daten
        signature_data = f"{state_id}".encode() + encrypted_data + iv + auth_tag

        # Erstelle HMAC-Signature
        signature = hmac.new(
            self._master_key,
            signature_data,
            hashlib.sha256
        ).hexdigest()

        return signature

    def _generate_master_key(self) -> bytes:
        """Generiert Master-Key für Encryption."""
        return os.urandom(32)  # 256-bit key

    async def _log_state_access(self, state_id: str, operation: str, tenant_id: str | None) -> None:
        """Loggt State-Access für Audit-Trail."""
        try:
            access_log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "state_id": state_id,
                "operation": operation,
                "tenant_id": tenant_id
            }

            self._state_access_log.append(access_log_entry)

            # Limitiere Log-Größe
            if len(self._state_access_log) > 10000:
                self._state_access_log = self._state_access_log[-10000:]

        except Exception as e:
            logger.error(f"State access logging fehlgeschlagen: {e}")

    async def _key_rotation_loop(self) -> None:
        """Background-Loop für Key-Rotation."""
        while self._is_running:
            try:
                await asyncio.sleep(self.key_rotation_interval_hours * 3600)

                if self._is_running:
                    await self._rotate_encryption_keys()

            except Exception as e:
                logger.error(f"Key rotation fehlgeschlagen: {e}")
                await asyncio.sleep(3600)  # Retry nach 1 Stunde

    async def _rotate_encryption_keys(self) -> None:
        """Rotiert Encryption-Keys."""
        try:
            # Generiere neuen Key
            new_key = self._generate_master_key()
            new_key_id = f"key_{len(self._encryption_keys) + 1:03d}"

            # Füge neuen Key hinzu
            self._encryption_keys[new_key_id] = new_key
            self._current_key_id = new_key_id
            self._master_key = new_key

            logger.info(f"Encryption key rotiert zu {new_key_id}")

            # TODO: Re-encrypt existing states with new key (optional) - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111

        except Exception as e:
            logger.error(f"Key rotation fehlgeschlagen: {e}")

    def _update_encryption_performance_stats(self, encryption_time_ms: float) -> None:
        """Aktualisiert Encryption-Performance-Statistiken."""
        self._encryption_count += 1
        self._total_encryption_time_ms += encryption_time_ms

    def _update_decryption_performance_stats(self, decryption_time_ms: float) -> None:
        """Aktualisiert Decryption-Performance-Statistiken."""
        self._decryption_count += 1
        self._total_decryption_time_ms += decryption_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_encryption_time = (
            self._total_encryption_time_ms / self._encryption_count
            if self._encryption_count > 0 else 0.0
        )

        avg_decryption_time = (
            self._total_decryption_time_ms / self._decryption_count
            if self._decryption_count > 0 else 0.0
        )

        return {
            "total_encryptions": self._encryption_count,
            "total_decryptions": self._decryption_count,
            "avg_encryption_time_ms": avg_encryption_time,
            "avg_decryption_time_ms": avg_decryption_time,
            "integrity_checks": self._integrity_check_count,
            "stored_states": len(self._encrypted_states),
            "active_keys": len(self._encryption_keys),
            "current_key_id": self._current_key_id,
            "tamper_protection_enabled": self.enable_tamper_protection,
            "integrity_verification_enabled": self.enable_integrity_verification
        }
