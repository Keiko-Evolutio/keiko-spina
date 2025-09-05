 # backend/kei_agents/security/encryption.py
"""Encryption und Key Management für KEI-Agents.

Implementiert umfassende Verschlüsselungsfunktionalitäten:
- Symmetrische Verschlüsselung (AES-256-GCM)
- Asymmetrische Verschlüsselung (RSA-2048)
- Key Management und Key Rotation
- Hybrid-Verschlüsselung für große Daten
"""

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Dict, Any, Optional, Tuple, Union, List

from kei_logging import get_logger

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    # Fallback-Definitionen
    Cipher = algorithms = modes = hashes = serialization = None
    rsa = padding = default_backend = None

from .exceptions import (
    EncryptionError, DecryptionError, KeyNotFoundError,
    EncryptionConfigError
)

logger = get_logger(__name__)


@dataclass
class EncryptionConfig:
    """Konfiguration für Encryption Manager."""
    default_symmetric_algorithm: str = "AES-256-GCM"
    default_asymmetric_algorithm: str = "RSA-2048"
    key_rotation_interval: timedelta = timedelta(days=30)
    enable_compression: bool = True
    key_derivation_iterations: int = 100000


class SymmetricEncryption:
    """Symmetrische Verschlüsselung mit AES-256-GCM."""

    def __init__(self, algorithm: str = "AES-256-GCM"):
        """Initialisiert SymmetricEncryption.

        Args:
            algorithm: Verschlüsselungsalgorithmus
        """
        self.algorithm = algorithm
        self.key_size = SymmetricEncryption._get_key_size(algorithm)
        self.block_size = 16  # AES block size

    @staticmethod
    def _get_key_size(algorithm: str) -> int:
        """Ermittelt Schlüsselgröße basierend auf Algorithmus."""
        if "256" in algorithm:
            return 256 // 8  # 32 bytes
        elif "192" in algorithm:
            return 192 // 8  # 24 bytes
        elif "128" in algorithm:
            return 128 // 8  # 16 bytes
        else:
            raise EncryptionConfigError(f"Unbekannter Algorithmus: {algorithm}")

    def generate_key(self) -> bytes:
        """Generiert neuen symmetrischen Schlüssel.

        Returns:
            Zufälliger Schlüssel
        """
        return os.urandom(self.key_size)

    def encrypt(self, data: Union[str, bytes], key: bytes,
                compress: bool = False) -> Dict[str, Any]:
        """Verschlüsselt Daten mit AES-GCM.

        Args:
            data: Zu verschlüsselnde Daten
            key: Verschlüsselungsschlüssel
            compress: Optional Kompression vor Verschlüsselung

        Returns:
            Dict mit verschlüsselten Daten und Metadaten
        """
        try:
            # Konvertiere zu bytes wenn nötig
            if isinstance(data, str):
                plaintext = data.encode('utf-8')
            else:
                plaintext = data

            # Optional: Kompression
            if compress:
                import gzip
                plaintext = gzip.compress(plaintext)

            # Generiere Nonce
            nonce = os.urandom(12)  # 96-bit nonce für GCM

            # Erstelle Cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            # Verschlüssele
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()

            return {
                "ciphertext": ciphertext,
                "nonce": nonce,
                "tag": encryptor.tag,
                "algorithm": self.algorithm,
                "compressed": compress,
                "timestamp": datetime.now(UTC).isoformat()
            }

        except Exception as e:
            logger.error(f"Encryption-Fehler: {e}")
            raise EncryptionError(f"Verschlüsselung fehlgeschlagen: {e}")

    def decrypt(self, encrypted_data: Dict[str, Any], key: bytes) -> Union[str, bytes]:
        """Entschlüsselt AES-GCM verschlüsselte Daten.

        Args:
            encrypted_data: Verschlüsselte Daten mit Metadaten
            key: Entschlüsselungsschlüssel

        Returns:
            Entschlüsselte Daten
        """
        try:
            # Erstelle Cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(encrypted_data["nonce"], encrypted_data["tag"]),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            # Entschlüssele
            plaintext = decryptor.update(encrypted_data["ciphertext"]) + decryptor.finalize()

            # Optional: Dekompression
            if encrypted_data.get("compressed", False):
                import gzip
                plaintext = gzip.decompress(plaintext)

            # Versuche als UTF-8 String zu dekodieren
            try:
                return plaintext.decode('utf-8')
            except UnicodeDecodeError:
                return plaintext

        except Exception as e:
            logger.error(f"Decryption-Fehler: {e}")
            raise DecryptionError(f"Entschlüsselung fehlgeschlagen: {e}")


class AsymmetricEncryption:
    """Asymmetrische Verschlüsselung mit RSA."""

    def __init__(self, algorithm: str = "RSA-2048"):
        """Initialisiert AsymmetricEncryption.

        Args:
            algorithm: Asymmetrischer Algorithmus
        """
        self.algorithm = algorithm
        self.key_size = int(algorithm.split("-")[1])

    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """Generiert RSA-Schlüsselpaar.

        Returns:
            Tuple (private_key, public_key) als PEM-encoded bytes
        """
        try:
            # Generiere Private Key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size,
                backend=default_backend()
            )

            # Serialisiere Private Key
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            # Serialisiere Public Key
            public_key = private_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            return private_pem, public_pem

        except Exception as e:
            logger.error(f"Key-Generierung fehlgeschlagen: {e}")
            raise EncryptionError(f"Schlüsselpaar-Generierung fehlgeschlagen: {e}")

    def encrypt(self, data: Union[str, bytes], public_key_pem: bytes) -> bytes:
        """Verschlüsselt Daten mit RSA Public Key.

        Args:
            data: Zu verschlüsselnde Daten
            public_key_pem: Public Key als PEM bytes

        Returns:
            Verschlüsselte Daten
        """
        try:
            # Konvertiere zu bytes
            if isinstance(data, str):
                plaintext = data.encode('utf-8')
            else:
                plaintext = data

            # Lade Public Key
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=default_backend()
            )

            # Verschlüssele mit OAEP Padding
            ciphertext = public_key.encrypt(
                plaintext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return ciphertext

        except Exception as e:
            logger.error(f"RSA-Encryption-Fehler: {e}")
            raise EncryptionError(f"RSA-Verschlüsselung fehlgeschlagen: {e}")

    def decrypt(self, ciphertext: bytes, private_key_pem: bytes) -> str:
        """Entschlüsselt RSA-verschlüsselte Daten.

        Args:
            ciphertext: Verschlüsselte Daten
            private_key_pem: Private Key als PEM bytes

        Returns:
            Entschlüsselte Daten als String
        """
        try:
            # Lade Private Key
            private_key = serialization.load_pem_private_key(
                private_key_pem, password=None, backend=default_backend()
            )

            # Entschlüssele
            plaintext = private_key.decrypt(
                ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return plaintext.decode('utf-8')

        except Exception as e:
            logger.error(f"RSA-Decryption-Fehler: {e}")
            raise DecryptionError(f"RSA-Entschlüsselung fehlgeschlagen: {e}")

    def sign(self, message: Union[str, bytes], private_key_pem: bytes) -> bytes:
        """Signiert Nachricht mit RSA Private Key.

        Args:
            message: Zu signierende Nachricht
            private_key_pem: Private Key als PEM bytes

        Returns:
            Digitale Signatur
        """
        try:
            if isinstance(message, str):
                message_bytes = message.encode('utf-8')
            else:
                message_bytes = message

            # Lade Private Key
            private_key = serialization.load_pem_private_key(
                private_key_pem, password=None, backend=default_backend()
            )

            # Signiere
            signature = private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return signature

        except Exception as e:
            logger.error(f"Signierung fehlgeschlagen: {e}")
            raise EncryptionError(f"Signierung fehlgeschlagen: {e}")

    def verify(self, message: Union[str, bytes], signature: bytes,
               public_key_pem: bytes) -> bool:
        """Verifiziert digitale Signatur.

        Args:
            message: Original-Nachricht
            signature: Digitale Signatur
            public_key_pem: Public Key als PEM bytes

        Returns:
            True wenn Signatur gültig, sonst False
        """
        try:
            if isinstance(message, str):
                message_bytes = message.encode('utf-8')
            else:
                message_bytes = message

            # Lade Public Key
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=default_backend()
            )

            # Verifiziere
            public_key.verify(
                signature,
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except (ValueError, TypeError) as e:
            logger.error(f"RSA-Signatur-Verifikation fehlgeschlagen - Ungültige Parameter: {e}")
            return False
        except Exception as e:
            logger.error(f"RSA-Signatur-Verifikation fehlgeschlagen - Unerwarteter Fehler: {e}")
            return False

    def encrypt_hybrid(self, data: Union[str, bytes], public_key_pem: bytes) -> Dict[str, Any]:
        """Hybrid-Verschlüsselung für große Daten.

        Args:
            data: Zu verschlüsselnde Daten
            public_key_pem: Public Key als PEM bytes

        Returns:
            Dict mit hybrid-verschlüsselten Daten
        """
        try:
            # Generiere AES-Schlüssel
            symmetric_encryption = SymmetricEncryption()
            aes_key = symmetric_encryption.generate_key()

            # Verschlüssele Daten mit AES
            encrypted_data = symmetric_encryption.encrypt(data, aes_key)

            # Verschlüssele AES-Schlüssel mit RSA
            encrypted_key = self.encrypt(aes_key, public_key_pem)

            return {
                "encrypted_data": encrypted_data,
                "encrypted_key": encrypted_key,
                "algorithm": "hybrid_rsa_aes"
            }

        except Exception as e:
            logger.error(f"Hybrid-Encryption-Fehler: {e}")
            raise EncryptionError(f"Hybrid-Verschlüsselung fehlgeschlagen: {e}")

    def decrypt_hybrid(self, hybrid_data: Dict[str, Any],
                      private_key_pem: bytes) -> Union[str, bytes]:
        """Hybrid-Entschlüsselung.

        Args:
            hybrid_data: Hybrid-verschlüsselte Daten
            private_key_pem: Private Key als PEM bytes

        Returns:
            Entschlüsselte Daten
        """
        try:
            # Entschlüssele AES-Schlüssel mit RSA
            aes_key = self.decrypt(hybrid_data["encrypted_key"], private_key_pem).encode()

            # Entschlüssele Daten mit AES
            symmetric_encryption = SymmetricEncryption()
            decrypted_data = symmetric_encryption.decrypt(
                hybrid_data["encrypted_data"], aes_key
            )

            return decrypted_data

        except Exception as e:
            logger.error(f"Hybrid-Decryption-Fehler: {e}")
            raise DecryptionError(f"Hybrid-Entschlüsselung fehlgeschlagen: {e}")


class KeyManager:
    """Key Management System für KEI-Agents."""

    def __init__(self, storage_backend: str = "memory"):
        """Initialisiert KeyManager.

        Args:
            storage_backend: Storage-Backend für Schlüssel
        """
        self.storage_backend = storage_backend
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._key_metadata: Dict[str, Dict[str, Any]] = {}

    def store_key(self, key_id: str, key_data: bytes,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Speichert Schlüssel mit Metadaten.

        Args:
            key_id: Eindeutige Schlüssel-ID
            key_data: Schlüsseldaten
            metadata: Optional Metadaten
        """
        try:
            self._keys[key_id] = {
                "data": key_data,
                "created_at": datetime.now(UTC),
                "version": 1
            }

            self._key_metadata[key_id] = metadata or {}
            logger.info(f"Schlüssel {key_id} erfolgreich gespeichert")

        except Exception as e:
            logger.error(f"Fehler beim Speichern von Schlüssel {key_id}: {e}")
            raise EncryptionError(f"Schlüssel-Speicherung fehlgeschlagen: {e}")

    def get_key(self, key_id: str, version: Optional[int] = None) -> bytes:
        """Ruft Schlüssel ab.

        Args:
            key_id: Schlüssel-ID
            version: Optional spezifische Version

        Returns:
            Schlüsseldaten
        """
        if key_id not in self._keys:
            raise KeyNotFoundError(f"Schlüssel {key_id} nicht gefunden")

        key_info = self._keys[key_id]

        # Prüfe Gültigkeit
        if not self.is_key_valid(key_id):
            raise KeyNotFoundError(f"Schlüssel {key_id} ist abgelaufen")

        return key_info["data"]

    def get_key_metadata(self, key_id: str) -> Dict[str, Any]:
        """Ruft Schlüssel-Metadaten ab.

        Args:
            key_id: Schlüssel-ID

        Returns:
            Metadaten-Dict
        """
        if key_id not in self._key_metadata:
            raise KeyNotFoundError(f"Metadaten für Schlüssel {key_id} nicht gefunden")

        return self._key_metadata[key_id].copy()

    def list_keys(self) -> List[str]:
        """Listet alle verfügbaren Schlüssel-IDs auf.

        Returns:
            Liste der Schlüssel-IDs
        """
        return list(self._keys.keys())

    def delete_key(self, key_id: str) -> None:
        """Löscht Schlüssel sicher.

        Args:
            key_id: Zu löschende Schlüssel-ID
        """
        if key_id in self._keys:
            # Überschreibe Schlüsseldaten vor Löschung
            key_data = self._keys[key_id]["data"]
            if isinstance(key_data, bytes):
                # Überschreibe mit Zufallsdaten
                self._keys[key_id]["data"] = os.urandom(len(key_data))

            del self._keys[key_id]

        if key_id in self._key_metadata:
            del self._key_metadata[key_id]

        logger.info(f"Schlüssel {key_id} sicher gelöscht")

    def rotate_key(self, key_id: str, new_key_data: bytes,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Rotiert Schlüssel (behält alte Version).

        Args:
            key_id: Schlüssel-ID
            new_key_data: Neue Schlüsseldaten
            metadata: Optional neue Metadaten
        """
        if key_id not in self._keys:
            raise KeyNotFoundError(f"Schlüssel {key_id} für Rotation nicht gefunden")

        # Archiviere alte Version
        old_key = self._keys[key_id].copy()
        old_version = old_key.get("version", 1)

        # Speichere neue Version
        self._keys[key_id] = {
            "data": new_key_data,
            "created_at": datetime.now(UTC),
            "version": old_version + 1,
            "previous_version": old_key
        }

        if metadata:
            self._key_metadata[key_id].update(metadata)

        logger.info(f"Schlüssel {key_id} rotiert zu Version {old_version + 1}")

    def is_key_valid(self, key_id: str) -> bool:
        """Prüft ob Schlüssel gültig ist.

        Args:
            key_id: Schlüssel-ID

        Returns:
            True wenn gültig, sonst False
        """
        if key_id not in self._key_metadata:
            return True  # Keine Ablaufzeit definiert

        metadata = self._key_metadata[key_id]
        expires_at_str = metadata.get("expires_at")

        if expires_at_str:
            try:
                expires_at = datetime.fromisoformat(expires_at_str)
                return datetime.now(UTC) < expires_at
            except ValueError:
                logger.warning(f"Ungültiges Ablaufdatum für Schlüssel {key_id}")
                return True

        return True

    def create_backup(self) -> Dict[str, Any]:
        """Erstellt Backup aller Schlüssel.

        Returns:
            Backup-Daten
        """
        backup = {
            "keys": {},
            "metadata": self._key_metadata.copy(),
            "created_at": datetime.now(UTC).isoformat(),
            "version": "1.0"
        }

        # Verschlüssele Schlüsseldaten für Backup
        for key_id, key_info in self._keys.items():
            backup["keys"][key_id] = {
                "data": base64.b64encode(key_info["data"]).decode(),
                "created_at": key_info["created_at"].isoformat(),
                "version": key_info["version"]
            }

        return backup

    def restore_backup(self, backup_data: Dict[str, Any]) -> None:
        """Stellt Schlüssel aus Backup wieder her.

        Args:
            backup_data: Backup-Daten
        """
        try:
            # Restore Schlüssel
            for key_id, key_info in backup_data["keys"].items():
                key_data = base64.b64decode(key_info["data"])
                self._keys[key_id] = {
                    "data": key_data,
                    "created_at": datetime.fromisoformat(key_info["created_at"]),
                    "version": key_info["version"]
                }

            # Restore Metadaten
            self._key_metadata = backup_data["metadata"].copy()

            logger.info(f"Backup mit {len(backup_data['keys'])} Schlüsseln wiederhergestellt")

        except Exception as e:
            logger.error(f"Backup-Wiederherstellung fehlgeschlagen: {e}")
            raise EncryptionError(f"Backup-Wiederherstellung fehlgeschlagen: {e}")


class EncryptionManager:
    """Zentraler Encryption Manager für KEI-Agents."""

    def __init__(self, config: EncryptionConfig):
        """Initialisiert EncryptionManager.

        Args:
            config: Encryption-Konfiguration
        """
        self.config = config
        self.symmetric_encryption = SymmetricEncryption(config.default_symmetric_algorithm)
        self.asymmetric_encryption = AsymmetricEncryption(config.default_asymmetric_algorithm)
        self.key_manager = KeyManager()

        logger.info(f"EncryptionManager initialisiert mit {config.default_symmetric_algorithm}")

    async def encrypt_agent_data(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Verschlüsselt Agent-Daten.

        Args:
            agent_id: Agent-ID
            data: Zu verschlüsselnde Daten

        Returns:
            Verschlüsselte Daten mit Metadaten
        """
        try:
            # Hole oder erstelle Agent-Schlüssel
            key_id = await self.get_or_create_agent_key(agent_id)
            encryption_key = self.key_manager.get_key(key_id)

            # Serialisiere Daten
            json_data = json.dumps(data, default=str)

            # Verschlüssele
            encrypted_result = self.symmetric_encryption.encrypt(
                json_data, encryption_key, compress=self.config.enable_compression
            )

            return {
                "encrypted_data": base64.b64encode(encrypted_result["ciphertext"]).decode(),
                "nonce": base64.b64encode(encrypted_result["nonce"]).decode(),
                "tag": base64.b64encode(encrypted_result["tag"]).decode(),
                "key_id": key_id,
                "algorithm": encrypted_result["algorithm"],
                "compressed": encrypted_result["compressed"],
                "timestamp": encrypted_result["timestamp"]
            }

        except Exception as e:
            logger.error(f"Agent-Daten-Verschlüsselung fehlgeschlagen: {e}")
            raise EncryptionError(f"Agent-Daten-Verschlüsselung fehlgeschlagen: {e}")

    async def decrypt_agent_data(self, agent_id: str,
                                encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Entschlüsselt Agent-Daten.

        Args:
            agent_id: Agent-ID
            encrypted_data: Verschlüsselte Daten

        Returns:
            Entschlüsselte Daten
        """
        try:
            # Hole Schlüssel
            key_id = encrypted_data["key_id"]
            decryption_key = self.key_manager.get_key(key_id)

            # Rekonstruiere encrypted_data für Decryption
            decryption_input = {
                "ciphertext": base64.b64decode(encrypted_data["encrypted_data"]),
                "nonce": base64.b64decode(encrypted_data["nonce"]),
                "tag": base64.b64decode(encrypted_data["tag"]),
                "compressed": encrypted_data.get("compressed", False)
            }

            # Entschlüssele
            decrypted_json = self.symmetric_encryption.decrypt(decryption_input, decryption_key)

            # Deserialisiere
            return json.loads(decrypted_json)

        except Exception as e:
            logger.error(f"Agent-Daten-Entschlüsselung fehlgeschlagen: {e}")
            raise DecryptionError(f"Agent-Daten-Entschlüsselung fehlgeschlagen: {e}")

    async def get_or_create_agent_key(self, agent_id: str) -> str:
        """Holt oder erstellt Schlüssel für Agent.

        Args:
            agent_id: Agent-ID

        Returns:
            Schlüssel-ID
        """
        key_id = f"agent_{agent_id}_key"

        try:
            # Prüfe ob Schlüssel existiert und gültig ist
            if key_id in self.key_manager.list_keys() and self.key_manager.is_key_valid(key_id):
                # Prüfe ob Key-Rotation nötig ist
                metadata = self.key_manager.get_key_metadata(key_id)
                created_at_str = metadata.get("created_at")

                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)
                    if datetime.now(UTC) - created_at > self.config.key_rotation_interval:
                        # Rotiere Schlüssel
                        new_key = self.symmetric_encryption.generate_key()
                        self.key_manager.rotate_key(key_id, new_key, {
                            "created_at": datetime.now(UTC).isoformat(),
                            "agent_id": agent_id
                        })
                        logger.info(f"Schlüssel für Agent {agent_id} rotiert")

                return key_id
            else:
                # Erstelle neuen Schlüssel
                new_key = self.symmetric_encryption.generate_key()
                self.key_manager.store_key(key_id, new_key, {
                    "created_at": datetime.now(UTC).isoformat(),
                    "agent_id": agent_id,
                    "algorithm": self.config.default_symmetric_algorithm
                })
                logger.info(f"Neuer Schlüssel für Agent {agent_id} erstellt")
                return key_id

        except Exception as e:
            logger.error(f"Schlüssel-Management für Agent {agent_id} fehlgeschlagen: {e}")
            raise EncryptionError(f"Schlüssel-Management fehlgeschlagen: {e}")
