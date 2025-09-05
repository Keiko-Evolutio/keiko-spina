#!/usr/bin/env python3
"""Voice Configuration Migration Test Script.

Testet die Backward-Compatibility und Funktionalität der neuen Voice Configuration.
Kann verwendet werden um sicherzustellen, dass die Migration erfolgreich war.
"""

import os
import sys
from pathlib import Path

# Füge Backend-Pfad hinzu
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from config.voice_config import (
    VoiceDetectionConfig,
    VoiceServiceSettings,
    reload_voice_config
)
from kei_logging import get_logger

logger = get_logger(__name__)


def test_default_configuration():
    """Testet Default-Konfiguration ohne Environment-Variablen."""
    print("🔧 Teste Default-Konfiguration...")
    
    # Entferne alle Voice-Environment-Variablen
    voice_env_vars = [key for key in os.environ.keys() if key.startswith("VOICE_")]
    for var in voice_env_vars:
        del os.environ[var]
    
    # Lade neue Konfiguration
    config = reload_voice_config("development")
    
    # Teste Default-Werte
    assert config.threshold == 0.8, f"Erwarteter threshold: 0.8, erhalten: {config.threshold}"
    assert config.silence_duration_ms == 1000, f"Erwartete silence_duration_ms: 1000, erhalten: {config.silence_duration_ms}"
    assert config.voice == "echo", f"Erwartete voice: echo, erhalten: {config.voice}"
    
    print("✅ Default-Konfiguration erfolgreich")
    return config


def test_environment_override():
    """Testet Überschreibung durch Environment-Variablen."""
    print("🔧 Teste Environment-Variable-Überschreibung...")
    
    # Setze Test-Environment-Variablen
    test_env = {
        "VOICE_THRESHOLD": "0.9",
        "VOICE_SILENCE_DURATION_MS": "1500",
        "VOICE_TEMPERATURE": "0.5",
        "VOICE_SYNTHESIS_VOICE": "alloy"
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    # Lade neue Konfiguration
    config = reload_voice_config()
    
    # Teste überschriebene Werte
    assert config.threshold == 0.9, f"Erwarteter threshold: 0.9, erhalten: {config.threshold}"
    assert config.silence_duration_ms == 1500, f"Erwartete silence_duration_ms: 1500, erhalten: {config.silence_duration_ms}"
    assert config.temperature == 0.5, f"Erwartete temperature: 0.5, erhalten: {config.temperature}"
    assert config.voice == "alloy", f"Erwartete voice: alloy, erhalten: {config.voice}"
    
    print("✅ Environment-Variable-Überschreibung erfolgreich")
    return config


def test_environment_specific_configs():
    """Testet umgebungsspezifische Konfigurationen."""
    print("🔧 Teste umgebungsspezifische Konfigurationen...")
    
    # Entferne Environment-Variablen
    voice_env_vars = [key for key in os.environ.keys() if key.startswith("VOICE_")]
    for var in voice_env_vars:
        del os.environ[var]
    
    # Teste Development
    dev_config = VoiceDetectionConfig.for_environment("development")
    assert dev_config.threshold == 0.8
    assert dev_config.temperature == 0.7
    print("  ✅ Development-Konfiguration korrekt")
    
    # Teste Staging
    staging_config = VoiceDetectionConfig.for_environment("staging")
    assert staging_config.threshold == 0.85
    assert staging_config.temperature == 0.6
    print("  ✅ Staging-Konfiguration korrekt")
    
    # Teste Production
    prod_config = VoiceDetectionConfig.for_environment("production")
    assert prod_config.threshold == 0.9
    assert prod_config.temperature == 0.5
    print("  ✅ Production-Konfiguration korrekt")
    
    print("✅ Umgebungsspezifische Konfigurationen erfolgreich")


def test_validation():
    """Testet Konfigurationsvalidierung."""
    print("🔧 Teste Konfigurationsvalidierung...")
    
    # Teste gültige Konfiguration
    config = VoiceDetectionConfig()
    config._validate_config()  # Sollte nicht fehlschlagen
    print("  ✅ Gültige Konfiguration validiert")
    
    # Teste ungültige Threshold
    config.threshold = 1.5
    try:
        config._validate_config()
        assert False, "Validierung sollte fehlschlagen"
    except ValueError as e:
        assert "Voice threshold muss zwischen 0.0 und 1.0 liegen" in str(e)
        print("  ✅ Ungültige Threshold korrekt abgefangen")
    
    # Reset für weitere Tests
    config.threshold = 0.8
    
    # Teste ungültige Silence Duration
    config.silence_duration_ms = 50
    try:
        config._validate_config()
        assert False, "Validierung sollte fehlschlagen"
    except ValueError as e:
        assert "Silence duration muss mindestens 100ms sein" in str(e)
        print("  ✅ Ungültige Silence Duration korrekt abgefangen")
    
    print("✅ Konfigurationsvalidierung erfolgreich")


def test_dictionary_conversion():
    """Testet Konvertierung zu Dictionary-Formaten."""
    print("🔧 Teste Dictionary-Konvertierung...")
    
    config = VoiceDetectionConfig()
    
    # Teste turn_detection Dictionary
    turn_detection = config.to_turn_detection_dict()
    expected_turn_detection = {
        "type": "server_vad",
        "threshold": 0.8,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 1000,
        "create_response": True
    }
    assert turn_detection == expected_turn_detection, f"Turn detection mismatch: {turn_detection}"
    print("  ✅ turn_detection Dictionary korrekt")
    
    # Teste transcription Dictionary
    transcription = config.to_transcription_dict()
    expected_transcription = {
        "model": "whisper-1",
        "language": "de"
    }
    assert transcription == expected_transcription, f"Transcription mismatch: {transcription}"
    print("  ✅ transcription Dictionary korrekt")
    
    print("✅ Dictionary-Konvertierung erfolgreich")


def test_voice_settings_integration():
    """Testet Integration mit VoiceSettings Klasse (vereinfacht)."""
    print("🔧 Teste VoiceSettings Integration...")

    # Teste nur die Voice Config direkt ohne FastAPI Import
    config = VoiceDetectionConfig()

    # Simuliere VoiceSettings Properties
    voice = config.voice
    assert isinstance(voice, str), "voice ist nicht String"
    assert voice == "echo", f"Erwartete voice: echo, erhalten: {voice}"

    turn_detection = config.to_turn_detection_dict()
    assert isinstance(turn_detection, dict), "turn_detection ist nicht Dictionary"
    assert "threshold" in turn_detection, "threshold fehlt in turn_detection"
    assert turn_detection["threshold"] == 0.8, f"Erwartete threshold: 0.8, erhalten: {turn_detection['threshold']}"

    transcription = config.to_transcription_dict()
    assert isinstance(transcription, dict), "transcription ist nicht Dictionary"
    assert "model" in transcription, "model fehlt in transcription"
    assert transcription["model"] == "whisper-1", f"Erwartetes model: whisper-1, erhalten: {transcription['model']}"

    print("✅ VoiceSettings Integration erfolgreich")


def test_pydantic_settings():
    """Testet Pydantic-basierte Settings."""
    print("🔧 Teste Pydantic Settings...")
    
    # Teste gültige Werte
    settings = VoiceServiceSettings(
        voice_threshold=0.8,
        voice_silence_duration_ms=1000,
        voice_temperature=0.7
    )
    assert settings.voice_threshold == 0.8
    print("  ✅ Gültige Pydantic Settings erstellt")
    
    # Teste Validierungsfehler
    try:
        VoiceServiceSettings(voice_threshold=1.5)
        assert False, "Pydantic Validierung sollte fehlschlagen"
    except ValueError:
        print("  ✅ Pydantic Validierungsfehler korrekt abgefangen")
    
    print("✅ Pydantic Settings erfolgreich")


def main():
    """Hauptfunktion für Migration Test."""
    print("🚀 Voice Configuration Migration Test")
    print("=" * 50)
    
    try:
        # Führe alle Tests durch
        test_default_configuration()
        test_environment_override()
        test_environment_specific_configs()
        test_validation()
        test_dictionary_conversion()
        test_voice_settings_integration()
        test_pydantic_settings()
        
        print("\n" + "=" * 50)
        print("🎉 Alle Tests erfolgreich! Voice Configuration Migration abgeschlossen.")
        print("\n📋 Nächste Schritte:")
        print("1. Starten Sie den Voice Service neu")
        print("2. Testen Sie Voice Detection Funktionalität")
        print("3. Überwachen Sie Logs auf Konfigurationswerte")
        print("4. Setzen Sie produktionsspezifische Environment-Variablen")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        logger.exception("Voice Configuration Migration Test fehlgeschlagen")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
