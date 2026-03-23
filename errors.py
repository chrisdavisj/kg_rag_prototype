class KGRAGError(Exception):
    """Base exception for KG RAG pipeline failures."""


class ConfigError(KGRAGError):
    """Raised when required configuration is missing or invalid."""


class SPARQLQueryError(KGRAGError):
    """Raised when a SPARQL query fails or returns an invalid payload."""


class GraphExpansionError(KGRAGError):
    """Raised when graph context expansion or parsing fails."""
