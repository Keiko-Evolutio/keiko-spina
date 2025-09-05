# backend/keiko_logging/clickable_logging_formatter.py
"""Vereinfachter ClickableLoggingFormatter mit Factory Pattern und Emoji-Support.
Ermöglicht einfache Nutzung: logging_formatter(__name__).info("message")
"""

import inspect
import logging
import logging.config
import os
from pathlib import Path
from threading import Lock

# Definiere custom TRAIN Log-Level
TRAIN_LEVEL = 25  # Zwischen INFO (20) und WARNING (30)
logging.addLevelName(TRAIN_LEVEL, "TRAIN")


def _find_caller_location() -> tuple[str, int, str] | None:
    """Findet die ursprüngliche Aufruf-Stelle im Code (überspringt Logger-interne Aufrufe).

    Returns:
        Tuple mit (filename, line_number, function_name) oder None
    """
    frame = inspect.currentframe()
    try:
        # Finde den ursprünglichen Aufrufer (überspringe Logger-Aufrufe)
        caller_frame = frame
        for _ in range(15):  # Max 15 Frames durchsuchen (erweitert für Factory Pattern)
            caller_frame = caller_frame.f_back
            if not caller_frame:
                break

            filename = caller_frame.f_code.co_filename

            # Überspringe Logger-interne und Factory-interne Aufrufe
            if (not any(skip_term in filename.lower() for skip_term in
                        ["logging", "keiko_logging", "clickable"]) and
                caller_frame.f_code.co_name not in ["__call__", "_create_configured_logger"]):
                return (
                    filename,
                    caller_frame.f_lineno,
                    caller_frame.f_code.co_name
                )
    finally:
        del frame

    return None


def _create_clickable_link(filename: str, line_number: int, rel_path: str) -> str:
    """Erstellt einen klickbaren Link für IDEs.

    Args:
        filename: Vollständiger Dateiname
        line_number: Zeilennummer
        rel_path: Relativer Pfad für Anzeige

    Returns:
        Formatierter klickbarer Link
    """
    # Prüfe ob Terminal Hyperlinks unterstützt
    if os.getenv("TERM_PROGRAM") in ["vscode", "code"] or os.getenv(
        "TERMINAL_EMULATOR") == "JetBrains-JediTerm":
        # ANSI Hyperlink für moderne Terminals
        file_uri = f"file://{filename}:{line_number}"
        # ANSI Escape Sequence: \033]8;;URI\033\\TEXT\033]8;;\033\\
        return f"\033]8;;{file_uri}\033\\{rel_path}:{line_number}\033]8;;\033\\"
    # Fallback: file:// URI für Copy/Paste
    return f"file://{filename}:{line_number}"


class EmojiClickableFormatter(logging.Formatter):
    """Erweitert den Standard-Formatter um klickbare Datei-Links, Emojis und Farben für besseres Debugging.
    Funktioniert in VS Code Terminal, PyCharm und modernen IDEs.
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None,
                 enable_links: bool = True, project_root: str | None = None):

        default_datefmt = "%d.%m.%y %H:%M:%S"
        super().__init__(fmt, datefmt or default_datefmt)
        self.enable_links = enable_links
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Emoji-Mapping für verschiedene Log-Level
        self.level_emojis = {
            "WARNING": "🟡",
            "ERROR": "🔴",
            "CRITICAL": "❌",
            "INFO": "🔵",
            "DEBUG": "⚪️",
            "TRAIN": "🤓"  # Training-Modus Emoji
        }

        # ANSI Farbcodes für Log-Level
        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[34m",  # Blau
            "TRAIN": "\033[95m",  # Helles Magenta für Training
            "WARNING": "\033[33m",  # Gelb
            "ERROR": "\033[31m",  # Rot
            "CRITICAL": "\033[35m",  # Magenta
            "RESET": "\033[0m"  # Reset
        }

    def format(self, record: logging.LogRecord) -> str:
        """Formatiert Log-Record mit Emojis, farbigem Level-Namen und klickbaren Links."""
        # Füge das entsprechende Emoji für den Log-Level hinzu
        record.level_emoji = self.level_emojis.get(record.levelname, "📝")

        # Speichere ursprünglichen Levelnamen für spätere Wiederherstellung
        original_levelname = record.levelname

        # Färbe nur den Level-Namen
        level_color = self.colors.get(record.levelname, "")
        reset_color = self.colors["RESET"]

        try:
            if level_color:
                # Ersetze mit farbigem Levelnamen
                record.levelname = f"{level_color}{record.levelname}{reset_color}"

            if self.enable_links:
                # Finde den ursprünglichen Aufruf-Ort (nicht Logger-interne Aufrufe)
                caller_info = _find_caller_location()

                if caller_info:
                    filename, line_number, function_name = caller_info

                    # Erstelle relativen Pfad vom Projekt-Root
                    try:
                        rel_path = os.path.relpath(filename, self.project_root)
                    except ValueError:
                        rel_path = filename

                    # Erstelle klickbaren Link (VS Code/PyCharm kompatibel)
                    clickable_link = _create_clickable_link(filename, line_number, rel_path)

                    # Erweitere Log-Record um klickbare Informationen
                    record.clickable_location = clickable_link
                    record.caller_function = function_name
                    record.caller_line = line_number
                    record.caller_file = rel_path
                else:
                    record.clickable_location = ""
            else:
                record.clickable_location = ""

            # Basis-Formatierung durchführen
            result = super().format(record)

        finally:
            # Stelle ursprünglichen Levelnamen immer wieder her (auch bei Exceptions)
            record.levelname = original_levelname

        return result


def _load_config_file():
    """Lädt logging.conf automatisch."""
    config_path = Path(__file__).parent.parent / "config" / "logging.conf"

    if config_path.exists():
        try:
            logging.config.fileConfig(config_path, disable_existing_loggers=False)
        except Exception:
            pass  # Ignoriere Config-Fehler


class ClickableLoggingFormatter:
    """Factory-Klasse für Logger mit Emoji-Support und klickbaren Links.
    """

    _instance_cache: dict[str, "ClickableLoggingFormatter"] = {}
    _logger_cache: dict[str, logging.Logger] = {}
    _cache_lock = Lock()

    def __init__(self,
                 enable_links: bool = True,
                 project_root: str | None = None,
                 log_level: str = os.getenv("LOG_LEVEL", "INFO"),
                 format_template: str | None = None,
                 auto_load_config: bool = True):

        if auto_load_config:
            _load_config_file()

        self.enable_links = enable_links
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Standard Format-Template mit dynamischem Emoji-Platzhalter
        self.format_template = format_template or (
            "%(level_emoji)s %(levelname)-17s [⏱️ %(asctime)s %(msecs)d] %(message)-100s %(clickable_location)s"
        )

        # Cache für konfigurierte Handler um Duplikate zu vermeiden
        self._configured_handlers: dict[str, bool] = {}

    def __call__(self, logger_name: str) -> logging.Logger:
        """Factory-Methode: Gibt einen konfigurierten Logger zurück.

        Args:
            logger_name: Name des Loggers (typischerweise __name__)

        Returns:
            Konfigurierter Logger mit EmojiClickableFormatter
        """
        # Threadsafe Cache-Check
        with self._cache_lock:
            if logger_name in self._logger_cache:
                return self._logger_cache[logger_name]

            # Neuen Logger erstellen und konfigurieren
            logger = self._create_configured_logger(logger_name)
            self._logger_cache[logger_name] = logger

            return logger

    def _create_configured_logger(self, logger_name: str) -> logging.Logger:
        """Erstellt und konfiguriert einen neuen Logger mit EmojiClickableFormatter.

        Args:
            logger_name: Name des zu erstellenden Loggers

        Returns:
            Vollständig konfigurierter Logger
        """
        configured_logger = logging.getLogger(logger_name)

        # Verhindere doppelte Handler für denselben Logger
        handler_key = f"{logger_name}_{id(self)}"
        if handler_key in self._configured_handlers:
            return configured_logger

        # Entferne bestehende Handler um Duplikate zu vermeiden
        for handler in configured_logger.handlers[:]:
            configured_logger.removeHandler(handler)

        # Logger-Konfiguration mit individuellen Level-Controls
        configured_logger.setLevel(logging.DEBUG)  # Niedrigster Level für Handler-Filterung
        configured_logger.propagate = False  # Verhindere doppelte Ausgaben

        # Erstelle Handler mit EmojiClickableFormatter und Level-Filter
        handler = logging.StreamHandler()
        formatter = self._create_emoji_clickable_formatter()
        handler.setFormatter(formatter)

        # Füge Level-Filter hinzu
        handler.addFilter(self._create_level_filter())

        # Handler zum Logger hinzufügen
        configured_logger.addHandler(handler)

        # Markiere als konfiguriert
        self._configured_handlers[handler_key] = True

        return configured_logger

    def _create_emoji_clickable_formatter(self) -> EmojiClickableFormatter:
        """Erstellt den EmojiClickableFormatter mit voller Link-Funktionalität.

        Returns:
            Konfigurierter EmojiClickableFormatter
        """
        return EmojiClickableFormatter(
            fmt=self.format_template,
            datefmt="%d.%m.%y %H:%M:%S",
            enable_links=self.enable_links,
            project_root=str(self.project_root)
        )

    def _create_level_filter(self) -> logging.Filter:
        """Erstellt Filter basierend auf Environment-Variablen."""

        class LevelFilter(logging.Filter):
            def filter(self, record):
                level_name = record.levelname

                if level_name == "DEBUG":
                    return os.getenv("LOGGING_DEBUG", "true").lower() in ("true", "1", "yes", "on")
                if level_name == "INFO":
                    return os.getenv("LOGGING_INFO", "true").lower() in ("true", "1", "yes", "on")
                if level_name == "TRAIN":
                    return os.getenv("LOGGING_TRAIN", "false").lower() in ("true", "1", "yes", "on")
                if level_name == "WARNING":
                    return os.getenv("LOGGING_WARNING", "true").lower() in ("true", "1", "yes", "on")
                if level_name == "ERROR":
                    return os.getenv("LOGGING_ERROR", "true").lower() in ("true", "1", "yes", "on")
                if level_name == "CRITICAL":
                    return os.getenv("LOGGING_CRITICAL", "true").lower() in ("true", "1", "yes", "on")

                return True

        return LevelFilter()

    @classmethod
    def get_default_instance(cls) -> "ClickableLoggingFormatter":
        """Gibt eine Standard-Instanz zurück (Singleton Pattern).

        Returns:
            Standard ClickableLoggingFormatter Instanz
        """
        cache_key = "default"

        with cls._cache_lock:
            if cache_key not in cls._instance_cache:
                cls._instance_cache[cache_key] = cls(project_root=str(Path.cwd()),
                                                     log_level=os.getenv("LOG_LEVEL", "INFO"))

            return cls._instance_cache[cache_key]

    def clear_cache(self):
        """Leert den Logger-Cache (nützlich für Tests).
        """
        with self._cache_lock:
            self._logger_cache.clear()
            self._configured_handlers.clear()

    def with_config(self, **kwargs) -> "ClickableLoggingFormatter":
        """Erstellt eine neue Instanz mit geänderten Konfigurationen.

        Args:
            **kwargs: Neue Konfigurationsparameter

        Returns:
            Neue ClickableLoggingFormatter Instanz
        """
        current_config = {
            "enable_links": self.enable_links,
            "project_root": str(self.project_root),
            "log_level": logging.getLevelName(self.log_level),
            "format_template": self.format_template
        }

        # Update mit neuen Parametern
        current_config.update(kwargs)

        return ClickableLoggingFormatter(**current_config)


def _add_train_method_to_logger(logger: logging.Logger) -> None:
    """Fügt train() Methode zu Logger hinzu."""
    def train(message, *args, **kwargs):
        """Loggt mit TRAIN Level."""
        if logger.isEnabledFor(TRAIN_LEVEL):
            logger._log(TRAIN_LEVEL, message, args, **kwargs)

    logger.train = train


def get_logger(logger_name: str, **config) -> logging.Logger:
    """Convenience-Funktion für schnelle Logger-Erstellung.

    Args:
        logger_name: Name des Loggers (typischerweise __name__)
        **config: Optionale Konfigurationsparameter

    Returns:
        Konfigurierter Logger mit train() Methode
    """
    if config:
        formatter = ClickableLoggingFormatter(**config)
    else:
        formatter = ClickableLoggingFormatter.get_default_instance()

    logger = formatter(logger_name)

    # Füge train() Methode hinzu
    _add_train_method_to_logger(logger)

    return logger


# Standard-Instanz für globale Verwendung
default_logging_formatter = ClickableLoggingFormatter.get_default_instance()

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Beispiel: Test aller Log-Level mit Emojis und klickbaren Links
    logging_formatter = ClickableLoggingFormatter(project_root=str(Path.cwd()))

    test_logger = logging_formatter(__name__)
    test_logger.info("✅ Info-Message mit Emoji und klickbarem Link")
    test_logger.debug("🐛 Debug Information")
    test_logger.warning("✴️ Warnung mit Emoji!")
    test_logger.error("🅰️ Fehler mit Emoji")
    test_logger.critical("🆘 Kritischer Fehler mit Emoji!")
