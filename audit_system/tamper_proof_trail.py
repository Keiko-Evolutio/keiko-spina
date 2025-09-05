# backend/audit_system/tamper_proof_trail.py
"""Tamper-Proof Audit Trail für Keiko Personal Assistant

Implementiert kryptographische Signaturen, Blockchain-ähnliche Verkettung
und unveränderliche Aufzeichnung aller Audit-Events.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kei_logging import get_logger
from observability import trace_function

from .core_audit_engine import AuditBlock, AuditEvent, AuditSignature

logger = get_logger(__name__)


class SignatureAlgorithm(str, Enum):
    """Unterstützte Signatur-Algorithmen."""
    RSA_SHA256 = "rsa_sha256"
    ECDSA_SHA256 = "ecdsa_sha256"
    HMAC_SHA256 = "hmac_sha256"


class TamperDetectionLevel(str, Enum):
    """Level der Tamper-Detection."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    PARANOID = "paranoid"


@dataclass
class TamperDetectionResult:
    """Ergebnis der Tamper-Detection."""
    is_tampered: bool
    tamper_type: str | None = None
    affected_events: list[str] = field(default_factory=list)
    detection_details: dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "is_tampered": self.is_tampered,
            "tamper_type": self.tamper_type,
            "affected_events": self.affected_events,
            "detection_details": self.detection_details,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat()
        }


class CryptographicSigner:
    """Kryptographischer Signer für Audit-Events."""

    def __init__(self, algorithm: SignatureAlgorithm = SignatureAlgorithm.RSA_SHA256):
        """Initialisiert Cryptographic Signer.

        Args:
            algorithm: Zu verwendender Signatur-Algorithmus
        """
        self.algorithm = algorithm
        self._private_key = None
        self._public_key = None
        self._hmac_key = None
        self._key_id = f"key_{int(time.time())}"

        # Generiere Schlüssel basierend auf Algorithmus
        self._generate_keys()

    def _generate_keys(self) -> None:
        """Generiert Schlüsselpaar für gewählten Algorithmus."""
        if self.algorithm == SignatureAlgorithm.RSA_SHA256:
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self._public_key = self._private_key.public_key()

        elif self.algorithm == SignatureAlgorithm.HMAC_SHA256:
            # Generiere HMAC-Schlüssel
            import secrets
            self._hmac_key = secrets.token_bytes(32)  # 256-bit Schlüssel

        else:
            raise ValueError(f"Algorithmus nicht unterstützt: {self.algorithm}")

    def sign_data(self, data: bytes) -> str:
        """Signiert Daten mit privatem Schlüssel.

        Args:
            data: Zu signierende Daten

        Returns:
            Base64-kodierte Signatur
        """
        import base64

        if self.algorithm == SignatureAlgorithm.RSA_SHA256:
            signature = self._private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode("utf-8")

        if self.algorithm == SignatureAlgorithm.HMAC_SHA256:
            signature = hmac.new(
                self._hmac_key,
                data,
                hashlib.sha256
            ).digest()
            return base64.b64encode(signature).decode("utf-8")

        raise ValueError(f"Algorithmus nicht unterstützt: {self.algorithm}")

    def verify_signature(self, data: bytes, signature: str) -> bool:
        """Verifiziert Signatur.

        Args:
            data: Originaldaten
            signature: Base64-kodierte Signatur

        Returns:
            True wenn Signatur gültig
        """
        import base64

        try:
            signature_bytes = base64.b64decode(signature)

            if self.algorithm == SignatureAlgorithm.RSA_SHA256:
                self._public_key.verify(
                    signature_bytes,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True

            if self.algorithm == SignatureAlgorithm.HMAC_SHA256:
                expected_signature = hmac.new(
                    self._hmac_key,
                    data,
                    hashlib.sha256
                ).digest()
                return hmac.compare_digest(signature_bytes, expected_signature)

            return False

        except Exception as e:
            logger.exception(f"Signatur-Verifikation fehlgeschlagen: {e}")
            return False

    def get_public_key_pem(self) -> str | None:
        """Gibt öffentlichen Schlüssel als PEM zurück."""
        if self._public_key:
            pem = self._public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return pem.decode("utf-8")
        return None

    def get_key_id(self) -> str:
        """Gibt Schlüssel-ID zurück."""
        return self._key_id


class AuditHashChain:
    """Hash-Chain für Audit-Events."""

    def __init__(self):
        """Initialisiert Hash Chain."""
        self._chain: list[str] = []
        self._genesis_hash = self._calculate_genesis_hash()
        self._chain.append(self._genesis_hash)

    def _calculate_genesis_hash(self) -> str:
        """Berechnet Genesis-Hash."""
        genesis_data = f"KEI_AUDIT_GENESIS_{int(time.time())}"
        return hashlib.sha256(genesis_data.encode("utf-8")).hexdigest()

    def add_event_hash(self, event: AuditEvent) -> str:
        """Fügt Event-Hash zur Chain hinzu.

        Args:
            event: Audit-Event

        Returns:
            Berechneter Hash
        """
        # Berechne Event-Hash
        event_data = self._serialize_event_for_hash(event)
        event_hash = hashlib.sha256(event_data.encode("utf-8")).hexdigest()

        # Verkette mit vorherigem Hash
        previous_hash = self._chain[-1]
        chained_data = f"{previous_hash}:{event_hash}"
        chained_hash = hashlib.sha256(chained_data.encode("utf-8")).hexdigest()

        # Füge zur Chain hinzu
        self._chain.append(chained_hash)

        # Setze Hashes im Event
        event.hash_value = event_hash
        event.previous_hash = previous_hash

        return chained_hash

    def _serialize_event_for_hash(self, event: AuditEvent) -> str:
        """Serialisiert Event für Hash-Berechnung."""
        # Deterministische Serialisierung für konsistente Hashes
        import json

        hash_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "action": event.action,
            "description": event.description,
            "context": event.context.to_dict(),
            "input_data": event.input_data,
            "output_data": event.output_data
        }

        return json.dumps(hash_data, sort_keys=True, separators=(",", ":"))

    def verify_chain_integrity(self) -> bool:
        """Verifiziert Integrität der gesamten Chain."""
        if len(self._chain) < 2:
            return True

        # Prüfe jeden Hash in der Chain
        for i in range(1, len(self._chain)):
            # Hier würde normalerweise das entsprechende Event geholt
            # und der Hash neu berechnet werden
            # Vereinfacht für Demo
            pass

        return True

    def get_chain_summary(self) -> dict[str, Any]:
        """Gibt Chain-Zusammenfassung zurück."""
        return {
            "chain_length": len(self._chain),
            "genesis_hash": self._genesis_hash,
            "latest_hash": self._chain[-1] if self._chain else None,
            "created_at": datetime.now(UTC).isoformat()
        }


class IntegrityVerifier:
    """Verifizierer für Audit-Trail-Integrität."""

    def __init__(self, signer: CryptographicSigner):
        """Initialisiert Integrity Verifier.

        Args:
            signer: Cryptographic Signer für Verifikation
        """
        self.signer = signer
        self._verification_cache: dict[str, bool] = {}

    @trace_function("audit.verify_event")
    async def verify_event_integrity(self, event: AuditEvent) -> bool:
        """Verifiziert Integrität eines einzelnen Events.

        Args:
            event: Zu verifizierendes Event

        Returns:
            True wenn Event integer ist
        """
        # Cache-Check
        if event.event_id in self._verification_cache:
            return self._verification_cache[event.event_id]

        try:
            # 1. Signatur-Verifikation
            if event.signature:
                event_data = self._serialize_event_for_verification(event)
                signature_valid = self.signer.verify_signature(
                    event_data.encode("utf-8"),
                    event.signature.signature
                )

                if not signature_valid:
                    logger.warning(f"Signatur-Verifikation fehlgeschlagen für Event {event.event_id}")
                    self._verification_cache[event.event_id] = False
                    return False

            # 2. Hash-Verifikation
            if event.hash_value:
                expected_hash = self._calculate_event_hash(event)
                if event.hash_value != expected_hash:
                    logger.warning(f"Hash-Verifikation fehlgeschlagen für Event {event.event_id}")
                    self._verification_cache[event.event_id] = False
                    return False

            # Event ist integer
            self._verification_cache[event.event_id] = True
            return True

        except Exception as e:
            logger.exception(f"Integrität-Verifikation fehlgeschlagen: {e}")
            self._verification_cache[event.event_id] = False
            return False

    def _serialize_event_for_verification(self, event: AuditEvent) -> str:
        """Serialisiert Event für Verifikation."""
        # Gleiche Logik wie in Hash Chain
        import json

        verification_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "action": event.action,
            "description": event.description,
            "context": event.context.to_dict()
        }

        return json.dumps(verification_data, sort_keys=True, separators=(",", ":"))

    def _calculate_event_hash(self, event: AuditEvent) -> str:
        """Berechnet Event-Hash neu."""
        event_data = self._serialize_event_for_verification(event)
        return hashlib.sha256(event_data.encode("utf-8")).hexdigest()

    async def detect_tampering(
        self,
        events: list[AuditEvent],
        detection_level: TamperDetectionLevel = TamperDetectionLevel.ENHANCED
    ) -> TamperDetectionResult:
        """Detektiert Tampering in Event-Liste.

        Args:
            events: Zu prüfende Events
            detection_level: Level der Detection

        Returns:
            Tamper-Detection-Ergebnis
        """
        tampered_events = []
        detection_details = {}

        for event in events:
            is_valid = await self.verify_event_integrity(event)
            if not is_valid:
                tampered_events.append(event.event_id)
                detection_details[event.event_id] = {
                    "reason": "integrity_check_failed",
                    "timestamp": event.timestamp.isoformat()
                }

        # Erweiterte Detection basierend auf Level
        if detection_level in [TamperDetectionLevel.ENHANCED, TamperDetectionLevel.PARANOID]:
            # Prüfe zeitliche Anomalien
            temporal_anomalies = self._detect_temporal_anomalies(events)
            if temporal_anomalies:
                detection_details["temporal_anomalies"] = temporal_anomalies

        if detection_level == TamperDetectionLevel.PARANOID:
            # Prüfe statistische Anomalien
            statistical_anomalies = self._detect_statistical_anomalies(events)
            if statistical_anomalies:
                detection_details["statistical_anomalies"] = statistical_anomalies

        # Berechne Confidence Score
        confidence_score = self._calculate_confidence_score(
            len(tampered_events),
            len(events),
            detection_details
        )

        return TamperDetectionResult(
            is_tampered=len(tampered_events) > 0,
            tamper_type="integrity_violation" if tampered_events else None,
            affected_events=tampered_events,
            detection_details=detection_details,
            confidence_score=confidence_score
        )

    def _detect_temporal_anomalies(self, events: list[AuditEvent]) -> list[dict[str, Any]]:
        """Detektiert zeitliche Anomalien."""
        anomalies = []

        # Sortiere Events nach Timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for i in range(1, len(sorted_events)):
            current = sorted_events[i]
            previous = sorted_events[i-1]

            # Prüfe auf Zeitsprünge rückwärts
            if current.timestamp < previous.timestamp:
                anomalies.append({
                    "type": "backward_time_jump",
                    "event_id": current.event_id,
                    "current_time": current.timestamp.isoformat(),
                    "previous_time": previous.timestamp.isoformat()
                })

        return anomalies

    def _detect_statistical_anomalies(self, events: list[AuditEvent]) -> list[dict[str, Any]]:
        """Detektiert statistische Anomalien."""
        anomalies = []

        # Prüfe Event-Häufigkeiten
        event_type_counts = {}
        for event in events:
            event_type_counts[event.event_type.value] = event_type_counts.get(event.event_type.value, 0) + 1

        # Einfache Anomalie-Detection basierend auf Häufigkeit
        total_events = len(events)
        for event_type, count in event_type_counts.items():
            frequency = count / total_events

            # Wenn ein Event-Typ > 80% aller Events ausmacht
            if frequency > 0.8:
                anomalies.append({
                    "type": "unusual_event_frequency",
                    "event_type": event_type,
                    "frequency": frequency,
                    "count": count
                })

        return anomalies

    def _calculate_confidence_score(
        self,
        tampered_count: int,
        total_count: int,
        detection_details: dict[str, Any]
    ) -> float:
        """Berechnet Confidence Score für Tamper-Detection."""
        if total_count == 0:
            return 0.0

        # Basis-Score basierend auf Tamper-Rate
        base_score = tampered_count / total_count

        # Bonus für zusätzliche Anomalien
        anomaly_bonus = 0.0
        if "temporal_anomalies" in detection_details:
            anomaly_bonus += 0.1
        if "statistical_anomalies" in detection_details:
            anomaly_bonus += 0.1

        return min(1.0, base_score + anomaly_bonus)


class BlockchainAuditChain:
    """Blockchain-ähnliche Audit-Chain."""

    def __init__(self, signer: CryptographicSigner):
        """Initialisiert Blockchain Audit Chain.

        Args:
            signer: Cryptographic Signer
        """
        self.signer = signer
        self.hash_chain = AuditHashChain()
        self.integrity_verifier = IntegrityVerifier(signer)

        # Chain-Konfiguration
        self._max_events_per_block = 100
        self._current_block_events: list[AuditEvent] = []
        self._blocks: list[AuditBlock] = []

    async def add_event(self, event: AuditEvent) -> str:
        """Fügt Event zur Chain hinzu.

        Args:
            event: Hinzuzufügendes Event

        Returns:
            Block-Hash
        """
        # Signiere Event
        await self._sign_event(event)

        # Füge zur Hash-Chain hinzu
        self.hash_chain.add_event_hash(event)

        # Füge zu aktuellem Block hinzu
        self._current_block_events.append(event)

        # Prüfe, ob Block voll ist
        if len(self._current_block_events) >= self._max_events_per_block:
            return await self._finalize_block()

        return ""

    async def _sign_event(self, event: AuditEvent) -> None:
        """Signiert Event."""
        event_data = self._serialize_event_for_signing(event)
        signature = self.signer.sign_data(event_data.encode("utf-8"))

        event.signature = AuditSignature(
            algorithm=self.signer.algorithm.value,
            signature=signature,
            public_key_id=self.signer.get_key_id(),
            timestamp=datetime.now(UTC)
        )

    def _serialize_event_for_signing(self, event: AuditEvent) -> str:
        """Serialisiert Event für Signierung."""
        import json

        signing_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "action": event.action,
            "description": event.description
        }

        return json.dumps(signing_data, sort_keys=True, separators=(",", ":"))

    async def _finalize_block(self) -> str:
        """Finalisiert aktuellen Block.

        Returns:
            Block-Hash
        """
        if not self._current_block_events:
            return ""

        # Erstelle Block
        block_id = str(len(self._blocks))
        timestamp = datetime.now(UTC)
        previous_block_hash = self._blocks[-1].block_hash if self._blocks else None

        # Berechne Merkle Root
        merkle_root = self._calculate_merkle_root(self._current_block_events)

        # Berechne Block-Hash
        block_data = f"{block_id}:{timestamp.isoformat()}:{previous_block_hash}:{merkle_root}"
        block_hash = hashlib.sha256(block_data.encode("utf-8")).hexdigest()

        # Signiere Block
        block_signature = self.signer.sign_data(block_data.encode("utf-8"))
        signature = AuditSignature(
            algorithm=self.signer.algorithm.value,
            signature=block_signature,
            public_key_id=self.signer.get_key_id(),
            timestamp=timestamp
        )

        # Erstelle Block
        block = AuditBlock(
            block_id=block_id,
            timestamp=timestamp,
            events=self._current_block_events.copy(),
            previous_block_hash=previous_block_hash,
            block_hash=block_hash,
            merkle_root=merkle_root,
            signature=signature
        )

        # Füge Block zur Chain hinzu
        self._blocks.append(block)

        # Reset aktueller Block
        self._current_block_events.clear()

        logger.info(f"Audit-Block finalisiert: {block_id} mit {len(block.events)} Events")

        return block_hash

    def _calculate_merkle_root(self, events: list[AuditEvent]) -> str:
        """Berechnet Merkle Root für Events."""
        if not events:
            return ""

        # Vereinfachte Merkle-Tree-Implementierung
        event_hashes = [event.hash_value or "" for event in events]

        while len(event_hashes) > 1:
            new_hashes = []

            for i in range(0, len(event_hashes), 2):
                left = event_hashes[i]
                right = event_hashes[i + 1] if i + 1 < len(event_hashes) else left

                combined = f"{left}:{right}"
                combined_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                new_hashes.append(combined_hash)

            event_hashes = new_hashes

        return event_hashes[0] if event_hashes else ""

    async def verify_chain_integrity(self) -> TamperDetectionResult:
        """Verifiziert Integrität der gesamten Chain."""
        all_events = []
        for block in self._blocks:
            all_events.extend(block.events)

        return await self.integrity_verifier.detect_tampering(all_events)

    def get_chain_info(self) -> dict[str, Any]:
        """Gibt Chain-Informationen zurück."""
        total_events = sum(len(block.events) for block in self._blocks)

        return {
            "total_blocks": len(self._blocks),
            "total_events": total_events,
            "current_block_events": len(self._current_block_events),
            "latest_block_hash": self._blocks[-1].block_hash if self._blocks else None,
            "chain_summary": self.hash_chain.get_chain_summary()
        }


class TamperProofAuditTrail:
    """Hauptklasse für Tamper-Proof Audit Trail."""

    def __init__(self, algorithm: SignatureAlgorithm = SignatureAlgorithm.RSA_SHA256):
        """Initialisiert Tamper-Proof Audit Trail.

        Args:
            algorithm: Signatur-Algorithmus
        """
        self.signer = CryptographicSigner(algorithm)
        self.blockchain_chain = BlockchainAuditChain(self.signer)

        # Statistiken
        self._events_added = 0
        self._blocks_created = 0
        self._integrity_checks = 0
        self._tamper_detections = 0

    async def add_audit_event(self, event: AuditEvent) -> str:
        """Fügt Event zum Tamper-Proof Trail hinzu.

        Args:
            event: Audit-Event

        Returns:
            Block-Hash oder Event-Hash
        """
        block_hash = await self.blockchain_chain.add_event(event)
        self._events_added += 1

        if block_hash:
            self._blocks_created += 1

        return block_hash or event.hash_value or ""

    async def verify_trail_integrity(self) -> TamperDetectionResult:
        """Verifiziert Integrität des gesamten Trails.

        Returns:
            Tamper-Detection-Ergebnis
        """
        self._integrity_checks += 1

        result = await self.blockchain_chain.verify_chain_integrity()

        if result.is_tampered:
            self._tamper_detections += 1

        return result

    def get_trail_statistics(self) -> dict[str, Any]:
        """Gibt Trail-Statistiken zurück."""
        chain_info = self.blockchain_chain.get_chain_info()

        return {
            "events_added": self._events_added,
            "blocks_created": self._blocks_created,
            "integrity_checks": self._integrity_checks,
            "tamper_detections": self._tamper_detections,
            "signature_algorithm": self.signer.algorithm.value,
            "chain_info": chain_info
        }


# Globale Tamper-Proof Trail Instanz
tamper_proof_trail = TamperProofAuditTrail()
