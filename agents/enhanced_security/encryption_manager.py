# backend/agents/enhanced_security/encryption_manager.py
"""Encryption Manager

Enterprise-Grade Verschlüsselungsmanagement mit:
- Symmetrische und asymmetrische Verschlüsselung
- Key Management und Rotation
- Datenintegrität und Authentifizierung
- Compliance-konforme Algorithmen
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class EncryptionAlgorithm(Enum):
    """Unterstützte Verschlüsselungsalgorithmen."""

    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    FERNET = "fernet"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"


@dataclass
class EncryptionConfig:
    """Konfiguration für Encryption Manager."""

    # Standard-Algorithmus
    default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM

    # Key Management
    key_rotation_interval_hours: int = 24
    key_derivation_iterations: int = 100000
    master_key_length: int = 32

    # Sicherheitseinstellungen
    require_authentication: bool = True
    enable_compression: bool = False
    secure_random_source: str = "os.urandom"

    # Compliance
    fips_mode: bool = False
    audit_encryption_operations: bool = True

    # Erweiterte Konfiguration
    default_symmetric_algorithm: str = "AES-256-GCM"
    default_asymmetric_algorithm: str = "RSA-2048"


@dataclass
class EncryptionKey:
    """Verschlüsselungsschlüssel mit Metadaten."""

    key_id: str
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class CryptoError(Exception):
    """Verschlüsselungs-spezifische Exception."""


class KeyManager:
    """Verwaltet Verschlüsselungsschlüssel."""

    def __init__(self, config: EncryptionConfig):
        """Initialisiert Key Manager.

        Args:
            config: Encryption-Konfiguration
        """
        self.config = config
        self._keys: dict[str, EncryptionKey] = {}
        self._master_key: bytes | None = None

    def generate_key(
        self,
        algorithm: EncryptionAlgorithm,
        key_id: str | None = None
    ) -> EncryptionKey:
        """Generiert neuen Verschlüsselungsschlüssel.

        Args:
            algorithm: Verschlüsselungsalgorithmus
            key_id: Optionale Key-ID

        Returns:
            Generierter Schlüssel
        """
        if not key_id:
            key_id = self._generate_key_id()

        if algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
            key_data = os.urandom(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.RSA_2048:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        elif algorithm == EncryptionAlgorithm.RSA_4096:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise CryptoError(f"Nicht unterstützter Algorithmus: {algorithm}")

        # Ablaufzeit berechnen
        expires_at = None
        if self.config.key_rotation_interval_hours > 0:
            expires_at = time.time() + (self.config.key_rotation_interval_hours * 3600)

        encryption_key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            key_data=key_data,
            expires_at=expires_at
        )

        self._keys[key_id] = encryption_key
        logger.debug(f"Schlüssel generiert: {key_id} ({algorithm.value})")

        return encryption_key

    def get_key(self, key_id: str) -> EncryptionKey | None:
        """Ruft Schlüssel ab.

        Args:
            key_id: Schlüssel-ID

        Returns:
            Schlüssel oder None
        """
        key = self._keys.get(key_id)
        if key and self._is_key_valid(key):
            return key
        return None

    def rotate_key(self, key_id: str) -> EncryptionKey:
        """Rotiert einen Schlüssel.

        Args:
            key_id: Schlüssel-ID

        Returns:
            Neuer Schlüssel
        """
        old_key = self._keys.get(key_id)
        if not old_key:
            raise CryptoError(f"Schlüssel nicht gefunden: {key_id}")

        # Alten Schlüssel deaktivieren
        old_key.is_active = False

        # Neuen Schlüssel generieren
        new_key = self.generate_key(old_key.algorithm, key_id)

        logger.info(f"Schlüssel rotiert: {key_id}")
        return new_key

    @staticmethod
    def _generate_key_id() -> str:
        """Generiert eindeutige Schlüssel-ID.

        Returns:
            Eindeutige Schlüssel-ID im Format 'key_<timestamp>_<random>'
        """
        return f"key_{int(time.time())}_{os.urandom(4).hex()}"

    @staticmethod
    def _is_key_valid(key: EncryptionKey) -> bool:
        """Prüft ob Schlüssel gültig ist.

        Args:
            key: Zu prüfender Encryption-Key

        Returns:
            True wenn Schlüssel gültig und aktiv ist
        """
        if not key.is_active:
            return False

        if key.expires_at and time.time() > key.expires_at:
            return False

        return True

    def get_key_statistics(self) -> dict[str, int]:
        """Gibt Statistiken über verwaltete Schlüssel zurück.

        Returns:
            Dictionary mit active_keys und total_keys Anzahl
        """
        active_count = len([k for k in self._keys.values() if k.is_active])
        total_count = len(self._keys)
        return {
            "active_keys": active_count,
            "total_keys": total_count
        }


class EncryptionManager:
    """Enterprise Encryption Manager für Keiko Personal Assistant"""

    def __init__(self, config: EncryptionConfig):
        """Initialisiert Encryption Manager.

        Args:
            config: Encryption-Konfiguration
        """
        self.config = config
        self.key_manager = KeyManager(config)
        self._default_key: EncryptionKey | None = None

        # Standard-Schlüssel generieren
        self._initialize_default_key()

        logger.info("Encryption Manager initialisiert")

    def _initialize_default_key(self) -> None:
        """Initialisiert Standard-Verschlüsselungsschlüssel."""
        self._default_key = self.key_manager.generate_key(
            self.config.default_algorithm,
            "default"
        )

    @trace_function("encryption.encrypt")
    def encrypt(
        self,
        data: str | bytes,
        key_id: str | None = None,
        _algorithm: EncryptionAlgorithm | None = None
    ) -> tuple[bytes, str]:
        """Verschlüsselt Daten.

        Args:
            data: Zu verschlüsselnde Daten
            key_id: Optionale Schlüssel-ID
            _algorithm: Optionaler Algorithmus (ungenutzt in aktueller Implementation)

        Returns:
            Tuple aus verschlüsselten Daten und Schlüssel-ID
        """
        try:
            # Daten vorbereiten
            if isinstance(data, str):
                data = data.encode("utf-8")

            # Schlüssel bestimmen
            if key_id:
                key = self.key_manager.get_key(key_id)
                if not key:
                    raise CryptoError(f"Schlüssel nicht gefunden: {key_id}")
            else:
                key = self._default_key
                if not key:
                    raise CryptoError("Kein Standard-Schlüssel verfügbar")

            # Verschlüsselung durchführen
            if key.algorithm == EncryptionAlgorithm.FERNET:
                encrypted_data = self._encrypt_fernet(data, key.key_data)
            elif key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                encrypted_data = self._encrypt_aes_gcm(data, key.key_data)
            elif key.algorithm == EncryptionAlgorithm.AES_256_CBC:
                encrypted_data = self._encrypt_aes_cbc(data, key.key_data)
            else:
                raise CryptoError(f"Verschlüsselung nicht unterstützt: {key.algorithm}")

            logger.debug(f"Daten verschlüsselt mit Schlüssel: {key.key_id}")
            return encrypted_data, key.key_id

        except Exception as e:
            logger.error(f"Verschlüsselung fehlgeschlagen: {e}")
            raise CryptoError(f"Verschlüsselung fehlgeschlagen: {e!s}")

    @trace_function("encryption.decrypt")
    def decrypt(
        self,
        encrypted_data: bytes,
        key_id: str
    ) -> bytes:
        """Entschlüsselt Daten.

        Args:
            encrypted_data: Verschlüsselte Daten
            key_id: Schlüssel-ID

        Returns:
            Entschlüsselte Daten
        """
        try:
            # Schlüssel abrufen
            key = self.key_manager.get_key(key_id)
            if not key:
                raise CryptoError(f"Schlüssel nicht gefunden: {key_id}")

            # Entschlüsselung durchführen
            if key.algorithm == EncryptionAlgorithm.FERNET:
                decrypted_data = self._decrypt_fernet(encrypted_data, key.key_data)
            elif key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                decrypted_data = self._decrypt_aes_gcm(encrypted_data, key.key_data)
            elif key.algorithm == EncryptionAlgorithm.AES_256_CBC:
                decrypted_data = self._decrypt_aes_cbc(encrypted_data, key.key_data)
            else:
                raise CryptoError(f"Entschlüsselung nicht unterstützt: {key.algorithm}")

            logger.debug(f"Daten entschlüsselt mit Schlüssel: {key_id}")
            return decrypted_data

        except Exception as e:
            logger.error(f"Entschlüsselung fehlgeschlagen: {e}")
            raise CryptoError(f"Entschlüsselung fehlgeschlagen: {e!s}")

    @staticmethod
    def _encrypt_fernet(data: bytes, key: bytes) -> bytes:
        """Verschlüsselt mit Fernet.

        Args:
            data: Zu verschlüsselnde Daten
            key: Fernet-Schlüssel

        Returns:
            Verschlüsselte Daten
        """
        f = Fernet(key)
        return f.encrypt(data)

    @staticmethod
    def _decrypt_fernet(encrypted_data: bytes, key: bytes) -> bytes:
        """Entschlüsselt mit Fernet.

        Args:
            encrypted_data: Verschlüsselte Daten
            key: Fernet-Schlüssel

        Returns:
            Entschlüsselte Daten
        """
        f = Fernet(key)
        return f.decrypt(encrypted_data)

    @staticmethod
    def _encrypt_aes_gcm(data: bytes, key: bytes) -> bytes:
        """Verschlüsselt mit AES-256-GCM.

        Args:
            data: Zu verschlüsselnde Daten
            key: AES-256-Schlüssel (32 Bytes)

        Returns:
            Verschlüsselte Daten (IV + Tag + Ciphertext)
        """
        # IV generieren
        iv = os.urandom(12)  # 96 bits für GCM

        # Cipher erstellen
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()

        # Verschlüsseln
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # IV + Tag + Ciphertext kombinieren
        return iv + encryptor.tag + ciphertext

    @staticmethod
    def _decrypt_aes_gcm(encrypted_data: bytes, key: bytes) -> bytes:
        """Entschlüsselt mit AES-256-GCM.

        Args:
            encrypted_data: Verschlüsselte Daten (IV + Tag + Ciphertext)
            key: AES-256-Schlüssel (32 Bytes)

        Returns:
            Entschlüsselte Daten
        """
        # IV, Tag und Ciphertext extrahieren
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]

        # Cipher erstellen
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()

        # Entschlüsseln
        return decryptor.update(ciphertext) + decryptor.finalize()

    @staticmethod
    def _encrypt_aes_cbc(data: bytes, key: bytes) -> bytes:
        """Verschlüsselt mit AES-256-CBC.

        Args:
            data: Zu verschlüsselnde Daten
            key: AES-256-Schlüssel (32 Bytes)

        Returns:
            Verschlüsselte Daten (IV + Ciphertext)
        """
        # Padding hinzufügen (PKCS7)
        pad_length = 16 - (len(data) % 16)
        padded_data = data + bytes([pad_length] * pad_length)

        # IV generieren
        iv = os.urandom(16)

        # Cipher erstellen
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()

        # Verschlüsseln
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # IV + Ciphertext kombinieren
        return iv + ciphertext

    @staticmethod
    def _decrypt_aes_cbc(encrypted_data: bytes, key: bytes) -> bytes:
        """Entschlüsselt mit AES-256-CBC.

        Args:
            encrypted_data: Verschlüsselte Daten (IV + Ciphertext)
            key: AES-256-Schlüssel (32 Bytes)

        Returns:
            Entschlüsselte Daten
        """
        # IV und Ciphertext extrahieren
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        # Cipher erstellen
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()

        # Entschlüsseln
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Padding entfernen (PKCS7)
        pad_length = padded_data[-1]
        return padded_data[:-pad_length]

    @staticmethod
    def generate_hash(
        data: str | bytes,
        algorithm: str = "sha256"
    ) -> str:
        """Generiert Hash von Daten.

        Args:
            data: Zu hashende Daten
            algorithm: Hash-Algorithmus (sha256, sha512, md5)

        Returns:
            Hex-kodierter Hash

        Raises:
            CryptoError: Bei nicht unterstütztem Algorithmus
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        if algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        if algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        raise CryptoError(f"Nicht unterstützter Hash-Algorithmus: {algorithm}")

    @staticmethod
    def verify_hmac(
        data: str | bytes,
        signature: str,
        key: str | bytes,
        algorithm: str = "sha256"
    ) -> bool:
        """Verifiziert HMAC-Signatur.

        Args:
            data: Originaldaten
            signature: HMAC-Signatur
            key: HMAC-Schlüssel
            algorithm: Hash-Algorithmus (sha256, sha512)

        Returns:
            True wenn Signatur gültig

        Raises:
            CryptoError: Bei nicht unterstütztem Algorithmus
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        if isinstance(key, str):
            key = key.encode("utf-8")

        if algorithm == "sha256":
            expected = hmac.new(key, data, hashlib.sha256).hexdigest()
        elif algorithm == "sha512":
            expected = hmac.new(key, data, hashlib.sha512).hexdigest()
        else:
            raise CryptoError(f"Nicht unterstützter HMAC-Algorithmus: {algorithm}")

        return hmac.compare_digest(signature, expected)

    @trace_function("encryption.encrypt_agent_data")
    async def encrypt_agent_data(self, agent_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Verschlüsselt Agent-Daten.

        Args:
            agent_id: Agent-ID
            data: Zu verschlüsselnde Daten

        Returns:
            Verschlüsselte Daten mit Metadaten
        """
        try:
            import base64
            import json

            # Hole oder erstelle Agent-Schlüssel
            key_id = f"agent_{agent_id}"
            key = self.key_manager.get_key(key_id)
            if not key:
                key = self.key_manager.generate_key(self.config.default_algorithm, key_id)

            # Serialisiere Daten
            json_data = json.dumps(data, default=str)

            # Verschlüssele
            encrypted_data, _ = self.encrypt(json_data, key_id)

            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "key_id": key_id,
                "algorithm": key.algorithm.value,
                "agent_id": agent_id,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Agent-Daten-Verschlüsselung fehlgeschlagen: {e}")
            raise CryptoError(f"Agent-Daten-Verschlüsselung fehlgeschlagen: {e!s}")

    @trace_function("encryption.decrypt_agent_data")
    async def decrypt_agent_data(self, agent_id: str, encrypted_data: dict[str, Any]) -> dict[str, Any]:
        """Entschlüsselt Agent-Daten.

        Args:
            agent_id: Agent-ID
            encrypted_data: Verschlüsselte Daten

        Returns:
            Entschlüsselte Daten
        """
        try:
            import base64
            import json

            # Validiere Agent-ID für Sicherheit
            stored_agent_id = encrypted_data.get("agent_id")
            if stored_agent_id and stored_agent_id != agent_id:
                raise CryptoError(f"Agent-ID-Mismatch: erwartet {stored_agent_id}, erhalten {agent_id}")

            # Daten extrahieren
            ciphertext = base64.b64decode(encrypted_data["encrypted_data"])
            key_id = encrypted_data["key_id"]

            # Entschlüssele
            decrypted_data = self.decrypt(ciphertext, key_id)

            # Deserialisiere JSON
            return json.loads(decrypted_data.decode("utf-8"))

        except Exception as e:
            logger.error(f"Agent-Daten-Entschlüsselung fehlgeschlagen: {e}")
            raise CryptoError(f"Agent-Daten-Entschlüsselung fehlgeschlagen: {e!s}")

    def get_encryption_status(self) -> dict[str, Any]:
        """Gibt aktuellen Encryption-Status zurück."""
        key_stats = self.key_manager.get_key_statistics()
        return {
            "default_algorithm": self.config.default_algorithm.value,
            "active_keys": key_stats["active_keys"],
            "total_keys": key_stats["total_keys"],
            "key_rotation_interval": self.config.key_rotation_interval_hours,
            "fips_mode": self.config.fips_mode,
            "audit_enabled": self.config.audit_encryption_operations
        }
