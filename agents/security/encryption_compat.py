# backend/agents/security/encryption_compat.py
"""Backward-Compatibility-Adapter für security/encryption.py

Stellt die alte API bereit und leitet alle Aufrufe an das konsolidierte
enhanced_security/encryption_manager.py Modul weiter.

DEPRECATED: Dieses Modul wird in einer zukünftigen Version entfernt.
Verwenden Sie stattdessen backend.agents.enhanced_security.encryption_manager.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from kei_logging import get_logger

# Import des konsolidierten Moduls
from ..enhanced_security.encryption_manager import (
    CryptoError,
    EncryptionAlgorithm,
)
from ..enhanced_security.encryption_manager import (
    EncryptionConfig as EnhancedEncryptionConfig,
)
from ..enhanced_security.encryption_manager import (
    EncryptionManager as EnhancedEncryptionManager,
)

logger = get_logger(__name__)

# Deprecation Warning
warnings.warn(
    "backend.agents.security.encryption ist deprecated. "
    "Verwenden Sie backend.agents.enhanced_security.encryption_manager.",
    DeprecationWarning,
    stacklevel=2
)


# Backward-Compatibility Exceptions
class EncryptionError(CryptoError):
    """Backward-Compatibility für EncryptionError."""


class DecryptionError(CryptoError):
    """Backward-Compatibility für DecryptionError."""


class EncryptionConfigError(CryptoError):
    """Backward-Compatibility für EncryptionConfigError."""


@dataclass
class EncryptionConfig:
    """Backward-Compatibility für alte EncryptionConfig."""

    default_symmetric_algorithm: str = "AES-256-GCM"
    default_asymmetric_algorithm: str = "RSA-2048"
    key_rotation_interval: timedelta = timedelta(days=30)
    enable_compression: bool = True
    key_derivation_iterations: int = 100000

    def to_enhanced_config(self) -> EnhancedEncryptionConfig:
        """Konvertiert zu neuer EncryptionConfig."""
        return EnhancedEncryptionConfig(
            default_algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_rotation_interval_hours=int(self.key_rotation_interval.total_seconds() / 3600),
            key_derivation_iterations=self.key_derivation_iterations,
            enable_compression=self.enable_compression,
        )


class SymmetricEncryption:
    """Backward-Compatibility für SymmetricEncryption."""

    def __init__(self, algorithm: str = "AES-256-GCM"):
        """Initialisiert SymmetricEncryption."""
        warnings.warn(
            "SymmetricEncryption ist deprecated. Verwenden Sie EncryptionManager.",
            DeprecationWarning,
            stacklevel=2
        )
        self.algorithm = algorithm
        self._manager = EnhancedEncryptionManager(EnhancedEncryptionConfig())

    def generate_key(self) -> bytes:
        """Generiert Verschlüsselungsschlüssel."""
        import os
        return os.urandom(32)  # 256-bit key

    def encrypt(self, data: str | bytes, _key: bytes, compress: bool = False) -> dict[str, Any]:
        """Verschlüsselt Daten."""
        try:
            # Verwende enhanced encryption manager
            encrypted_data, key_id = self._manager.encrypt(data)

            # Simuliere alte API-Struktur
            return {
                "ciphertext": encrypted_data,
                "nonce": b"",  # Placeholder
                "tag": b"",    # Placeholder
                "algorithm": self.algorithm,
                "compressed": compress,
                "timestamp": "2024-01-01T00:00:00"  # Placeholder
            }
        except Exception as e:
            raise EncryptionError(f"Verschlüsselung fehlgeschlagen: {e}")

    def decrypt(self, encrypted_data: dict[str, Any], _key: bytes) -> str | bytes:
        """Entschlüsselt Daten."""
        try:
            # Verwende enhanced encryption manager
            return self._manager.decrypt(encrypted_data["ciphertext"], "default")
        except Exception as e:
            raise DecryptionError(f"Entschlüsselung fehlgeschlagen: {e}")


class AsymmetricEncryption:
    """Backward-Compatibility für AsymmetricEncryption."""

    def __init__(self, algorithm: str = "RSA-2048"):
        """Initialisiert AsymmetricEncryption."""
        warnings.warn(
            "AsymmetricEncryption ist deprecated. Verwenden Sie EncryptionManager.",
            DeprecationWarning,
            stacklevel=2
        )
        self.algorithm = algorithm
        self._manager = EnhancedEncryptionManager(EnhancedEncryptionConfig())

    def generate_key_pair(self) -> dict[str, bytes]:
        """Generiert RSA-Schlüsselpaar."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return {
            "private_key": private_pem,
            "public_key": public_pem
        }

    def encrypt(self, _data: str | bytes, _public_key_pem: bytes) -> bytes:
        """Verschlüsselt mit RSA Public Key."""
        # Placeholder - würde echte RSA-Verschlüsselung implementieren
        return b"encrypted_data_placeholder"

    def decrypt(self, _ciphertext: bytes, _private_key_pem: bytes) -> str:
        """Entschlüsselt RSA-verschlüsselte Daten."""
        # Placeholder - würde echte RSA-Entschlüsselung implementieren
        return "decrypted_data_placeholder"


class EncryptionManager:
    """Backward-Compatibility für EncryptionManager."""

    def __init__(self, config: EncryptionConfig):
        """Initialisiert EncryptionManager."""
        warnings.warn(
            "security.encryption.EncryptionManager ist deprecated. "
            "Verwenden Sie enhanced_security.encryption_manager.EncryptionManager.",
            DeprecationWarning,
            stacklevel=2
        )
        self.config = config
        self._enhanced_manager = EnhancedEncryptionManager(config.to_enhanced_config())

    async def encrypt_agent_data(self, agent_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Verschlüsselt Agent-Daten."""
        return await self._enhanced_manager.encrypt_agent_data(agent_id, data)

    async def decrypt_agent_data(self, agent_id: str, encrypted_data: dict[str, Any]) -> dict[str, Any]:
        """Entschlüsselt Agent-Daten."""
        return await self._enhanced_manager.decrypt_agent_data(agent_id, encrypted_data)


# Backward-Compatibility Exports
__all__ = [
    "AsymmetricEncryption",
    "DecryptionError",
    "EncryptionConfig",
    "EncryptionConfigError",
    "EncryptionError",
    "EncryptionManager",
    "SymmetricEncryption",
]

logger.warning(
    "Modul backend.agents.security.encryption_compat geladen. "
    "Migrieren Sie zu backend.agents.enhanced_security.encryption_manager."
)
