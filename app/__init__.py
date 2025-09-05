"""Keiko App-Paket mit Application, Startup und DI-Container.

Hinweis: Um Import-Seiteneffekte in Tests zu vermeiden, werden keine
Submodule eagerly importiert. Bitte direkt aus den Submodulen importieren:

    from app.service_container import ServiceContainer
    from app.application import KeikoApplication
"""

__all__ = ["application", "service_container"]
